import sys
import cv2
import numpy as np
import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from networks import schp_model
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from utils.agnostic_utils import generate_agnostic
from utils.parse_utils import generate_parse_agnostic

def create_dataset_structure(root):
    """
    Creates the following directory structure under 'root':

        root/
            ├── agnostic-v3.2
            ├── cloth
            ├── cloth-mask
            ├── image
            ├── image-densepose
            ├── image-parse-agnostic-v3.2
            ├── image-parse-v3
            ├── openpose-img
            ├── openpose-json
    """

    dirs = [
        "agnostic-v3.2",
        "cloth",
        "cloth-mask",
        "image",
        "image-densepose",
        "image-parse-agnostic-v3.2",
        "image-parse-v3",
        "openpose-img",
        "openpose-json"
    ]
    os.makedirs(root, exist_ok=True)
    for d in dirs:
        os.makedirs(os.path.join(root, d), exist_ok=True)




def generate_openpose_outputs(root, pil_image, image_name, openpose_root="/path/to/openpose"):
    """
    Generates OpenPose keypoints for a PIL image and saves the output image and json.

    Args:
        root (str): Root dataset directory.
        pil_image (PIL.Image): Input image.
        image_name (str): Filename (without extension).
        openpose_root (str): Path to CMU OpenPose.

    Saves:
        root/openpose-img/<image_name>.png
        root/openpose-json/<image_name>.json
    """
    sys.path.append(os.path.join(openpose_root, "build/python"))

    try:
        from openpose import pyopenpose as op
    except Exception as e:
        raise RuntimeError("OpenPose python module not found. Check openpose_root.") from e

    params = {
        "model_folder": os.path.join(openpose_root, "models/"),
        "hand": True,
        "face": True
    }

    op_wrapper = op.WrapperPython()
    op_wrapper.configure(params)
    op_wrapper.start()

    cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    datum = op.Datum()
    datum.cvInputData = cv_img
    op_wrapper.emplaceAndPop([datum])

    pose = datum.poseKeypoints
    face = datum.faceKeypoints
    hand_left = datum.handKeypoints[0] if datum.handKeypoints is not None else None
    hand_right = datum.handKeypoints[1] if datum.handKeypoints is not None else None
    rendered_img = datum.cvOutputData
    img_out_dir = os.path.join(root, "openpose-img")
    json_out_dir = os.path.join(root, "openpose-json")

    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(json_out_dir, exist_ok=True)

    img_out_path = os.path.join(img_out_dir, f"{image_name}.png")
    cv2.imwrite(img_out_path, rendered_img)

    def flatten(arr):
        return arr.reshape(-1).tolist() if arr is not None else []

    output_json = {
        "version": 1.3,
        "people": [
            {
                "person_id": [-1],
                "pose_keypoints_2d": flatten(pose),
                "face_keypoints_2d": flatten(face),
                "hand_left_keypoints_2d": flatten(hand_left),
                "hand_right_keypoints_2d": flatten(hand_right),
                "pose_keypoints_3d": [],
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": []
            }
        ]
    }

    json_out_path = os.path.join(json_out_dir, f"{image_name}.json")
    with open(json_out_path, "w") as f:
        json.dump(output_json, f, indent=2)

    return img_out_path, json_out_path


def load_schp():
    model = schp_model()
    ckpt = torch.load("schp.pth", map_location="cpu")
    model.load_state_dict(ckpt)
    model.eval()
    return model

schp = load_schp()

def load_densepose():
    cfg = get_cfg()
    from detectron2.projects.densepose import add_densepose_config
    add_densepose_config(cfg)
    cfg.merge_from_file("/content/VITON-HD/densepose_rcnn_R_50_FPN_s1x.yaml")
    cfg.MODEL.WEIGHTS = "/content/VITON-HD/models/densepose_rcnn_R_50_FPN_s1x.pkl"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
    return DefaultPredictor(cfg)

densepose_predictor = load_densepose()


def generate_full_viton_preprocess(root, pil_img, save_name):

    os.makedirs(os.path.join(root, "image"), exist_ok=True)
    os.makedirs(os.path.join(root, "image-parse-v3"), exist_ok=True)
    os.makedirs(os.path.join(root, "image-parse-agnostic-v3.2"), exist_ok=True)
    os.makedirs(os.path.join(root, "agnostic-v3.2"), exist_ok=True)
    os.makedirs(os.path.join(root, "image-densepose"), exist_ok=True)

    image_path = os.path.join(root, "image", f"{save_name}.jpg")
    pil_img.save(image_path)

    trans = transforms.Compose([
        transforms.Resize((1024, 768)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    img_tensor = trans(pil_img).unsqueeze(0)

    with torch.no_grad():
        out = schp(img_tensor)[0]
        parsing = out.argmax(0).cpu().numpy().astype(np.uint8)

    parse_path = os.path.join(root, "image-parse-v3", f"{save_name}.png")
    Image.fromarray(parsing).save(parse_path)

    parse_agnostic = generate_parse_agnostic(parsing)
    parse_agnostic_path = os.path.join(root, "image-parse-agnostic-v3.2", f"{save_name}.png")
    Image.fromarray(parse_agnostic).save(parse_agnostic_path)

    agnostic_img = generate_agnostic(
        np.array(pil_img),
        parsing,
        densepose=None,      # generated later
        pose_json=None       # only needed for virtual try-on, not agnostic gen
    )
    agnostic_path = os.path.join(root, "agnostic-v3.2", f"{save_name}.png")
    Image.fromarray(agnostic_img).save(agnostic_path)

    dense_out = densepose_predictor(np.array(pil_img))
    dp = dense_out["densepose"]
    dp_map = dp.labels.cpu().numpy().astype(np.uint8)

    dense_path = os.path.join(root, "image-densepose", f"{save_name}.png")
    Image.fromarray(dp_map).save(dense_path)

    return {
        "image": image_path,
        "parse": parse_path,
        "parse_agnostic": parse_agnostic_path,
        "agnostic": agnostic_path,
        "densepose": dense_path
    }

import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

# ------------------------------
# U2NET Model (same as VITON-HD)
# ------------------------------
from models.u2net import U2NETP

# Preprocessing for U2Net
u2net_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_u2net(model_path="u2netp.pth", device="cuda"):
    model = U2NETP(3, 1)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device).eval()
    return model


def generate_cloth_and_mask(cloth_img_pil, cloth_name, root, model_u2net):
    """
    Generates cloth mask using the official VITON-HD U2Net method and
    stores:
        cloth       → root/cloth/{cloth_name}.jpg
        cloth-mask  → root/cloth-mask/{cloth_name}.png
    """
    cloth_path = os.path.join(root, "cloth", f"{cloth_name}.jpg")
    cloth_img_pil.save(cloth_path)

    # --------------------------
    # Prepare input for U2Net
    # --------------------------
    device = next(model_u2net.parameters()).device

    img_tensor = u2net_transform(cloth_img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        d1, _, _, _, _, _, _ = model_u2net(img_tensor)
        pred_mask = d1[:, 0, :, :]  # first output
        pred_mask = F.interpolate(pred_mask.unsqueeze(0),
                                  size=cloth_img_pil.size[::-1],  # (H, W)
                                  mode="bilinear",
                                  align_corners=False)
        pred_mask = pred_mask.squeeze().cpu().numpy()

    # Normalize mask to 0–255
    mask_np = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min() + 1e-8)
    mask_np = (mask_np * 255).astype(np.uint8)

    # Apply thresholding (official repo)
    mask_np = cv2.GaussianBlur(mask_np, (3, 3), 0)
    _, mask_np = cv2.threshold(mask_np, 10, 255, cv2.THRESH_BINARY)

    # Save mask
    mask_pil = Image.fromarray(mask_np)
    mask_path = os.path.join(root, "cloth-mask", f"{cloth_name}.png")
    mask_pil.save(mask_path)

    print(f"[✔] Cloth saved: {cloth_path}")
    print(f"[✔] Cloth mask saved: {mask_path}")

    return cloth_path, mask_path

def preprocess_single_sample(
    img_path,
    cloth_path,
    root,
    model_openpose,
    model_parse,
    model_parse_agnostic,
    model_agnostic,
    model_densepose,
    model_u2net,
    person_name="0001",
    cloth_name="0001_00",
):

    # --------------------------------------------------------
    # Step 1 — Load images
    # --------------------------------------------------------
    person_img = Image.open(img_path).convert("RGB")
    cloth_img = Image.open(cloth_path).convert("RGB")

    # --------------------------------------------------------
    # Step 2 — Save Raw Person Image
    # --------------------------------------------------------
    person_out_path = os.path.join(root, "image", f"{person_name}.jpg")
    person_img.save(person_out_path)
    print(f"[✔] Saved raw person image → {person_out_path}")

    # --------------------------------------------------------
    # Step 3 — OpenPose (pose keypoints, hand, body)
    # --------------------------------------------------------
    pose_img_path, pose_json_path = generate_openpose(
        person_img, person_name, root, model_openpose
    )

    # --------------------------------------------------------
    # Step 4 — Human Parsing (SCHP)
    # --------------------------------------------------------
    parse_path = generate_parse(
        person_img, person_name, root, model_parse
    )

    # --------------------------------------------------------
    # Step 5 — Parse-Agnostic (VITON-HD)
    # --------------------------------------------------------
    parse_agnostic_path = generate_parse_agnostic(
        person_img, person_name, root, model_parse_agnostic
    )

    # --------------------------------------------------------
    # Step 6 — Agnostic Person Image
    # --------------------------------------------------------
    agnostic_path = generate_agnostic(
        person_img, person_name, root, model_agnostic
    )

    # --------------------------------------------------------
    # Step 7 — DensePose
    # --------------------------------------------------------
    densepose_path = generate_densepose(
        person_img, person_name, root, model_densepose
    )

    # --------------------------------------------------------
    # Step 8 — Cloth + Cloth Mask
    # --------------------------------------------------------
    cloth_img_path, cloth_mask_path = generate_cloth_and_mask(
        cloth_img, cloth_name, root, model_u2net
    )

    print("\n[🎉] Full VITON-HD preprocessing complete.")
    return {
        "person": person_out_path,
        "pose_img": pose_img_path,
        "pose_json": pose_json_path,
        "parse": parse_path,
        "parse_agnostic": parse_agnostic_path,
        "agnostic": agnostic_path,
        "densepose": densepose_path,
        "cloth": cloth_img_path,
        "cloth_mask": cloth_mask_path,
    }
