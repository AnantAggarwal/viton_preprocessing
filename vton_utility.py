# viton_kaggle_pipeline.py
# Kaggle-ready preprocessing module for VITON-HD style pipeline
# Depends on: openpifpaf, detectron2 + DensePose (optional), SCHP repo (optional),
# U-2-Net repo (for cloth masks).
#
# Run this AFTER you run setup_viton_hd_kaggle.sh and restarted the kernel.

import os
import sys
from PIL import Image
import numpy as np
import json
from pathlib import Path

# Add common cloned repo roots to sys.path so imports from those repos work.
# Adjust if you cloned repos to a different location.
REPO_ROOT = os.getcwd()
possible_paths = [
    os.path.join(REPO_ROOT, "DensePose"),
    os.path.join(REPO_ROOT, "SCHP"),
    os.path.join(REPO_ROOT, "U-2-Net"),
    os.path.join(REPO_ROOT, ""),  # repo root
]
for p in possible_paths:
    if p not in sys.path:
        sys.path.insert(0, p)


# -------------------------
# Utilities & Dataset dirs
# -------------------------
def create_dataset_structure(root):
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
    return root


# -------------------------
# OpenPifPaf (pose) wrapper
# -------------------------
def generate_openpose_with_openpifpaf(root, pil_image, image_name, checkpoint="resnet50"):
    """
    Runs OpenPifPaf on a PIL image and saves:
      - rendered skeleton image -> root/openpose-img/{image_name}.png
      - keypoints JSON (COCO-like single-person if available) -> root/openpose-json/{image_name}.json

    NOTE: openpifpaf must be installed (setup script installs it). API may change across versions.
    """
    try:
        import openpifpaf
    except Exception as e:
        raise RuntimeError("openpifpaf not installed. Run setup script and restart kernel.") from e

    # Prepare predictor
    predictor = openpifpaf.Predictor(checkpoint=checkpoint)

    # Convert PIL -> numpy RGB
    np_img = np.asarray(pil_image.convert("RGB"))

    # Predict
    annotations, _, _ = predictor.numpy_image(np_img)

    # Render skeleton onto image
    # Canvas drawing helper from openpifpaf
    try:
        from openpifpaf.show import Canvas
        canvas = Canvas(np_img)
        for ann in annotations:
            canvas.annotations(ann)
        rendered = canvas.image
    except Exception:
        # Fallback: no rendering, use original image
        rendered = np_img

    # Save rendered image
    out_img_path = os.path.join(root, "openpose-img", f"{image_name}.png")
    Image.fromarray(rendered).save(out_img_path)

    # Convert annotations to JSON structure similar to OpenPose COCO style:
    people = []
    for ann in annotations:
        # ann.data is an (nkp, 3) array: x, y, confidence
        try:
            keypoints = []
            kps = ann.data
            for (x, y, c) in kps:
                keypoints.extend([float(x), float(y), float(c)])
            person = {
                "person_id": [-1],
                "pose_keypoints_2d": keypoints,
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
                "pose_keypoints_3d": [],
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": []
            }
            people.append(person)
        except Exception:
            # skip malformed annotation
            continue

    out_json = {
        "version": 1.3,
        "people": people
    }

    out_json_path = os.path.join(root, "openpose-json", f"{image_name}.json")
    with open(out_json_path, "w") as f:
        json.dump(out_json, f, indent=2)

    return out_img_path, out_json_path


# -------------------------
# DensePose wrapper (Detectron2 + DensePose)
# -------------------------
def generate_densepose(root, pil_image, image_name, cfg_file=None, weights_path=None):
    """
    Runs DensePose inference and saves a visualized densepose map to root/image-densepose/{image_name}.png

    cfg_file: path to DensePose config YAML (relative to DensePose repo or absolute).
    weights_path: path to DensePose .pkl / .pth weights.

    If DensePose or Detectron2 are missing or weights are not provided, function raises a helpful error.
    """
    try:
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from densepose import add_densepose_config
        from detectron2 import model_zoo
        import cv2
    except Exception as e:
        raise RuntimeError("Detectron2 / DensePose not available. Make sure you installed them and restarted kernel.") from e

    # Set default config paths if not provided
    if cfg_file is None:
        # Try common DensePose config inside DensePose repo
        default_cfg = os.path.join(REPO_ROOT, "DensePose", "configs", "densepose_rcnn_R_50_FPN_s1x.yaml")
        if os.path.exists(default_cfg):
            cfg_file = default_cfg
        else:
            raise FileNotFoundError("DensePose config YAML not found. Provide cfg_file argument pointing to DensePose config.")

    if weights_path is None:
        # User should place pretrained weights in pretrained_models/densepose/
        candidate = os.path.join(REPO_ROOT, "pretrained_models", "densepose", "DensePose_ResNet50_FPN_s1x.pkl")
        if os.path.exists(candidate):
            weights_path = candidate
        else:
            raise FileNotFoundError("DensePose weights not found. Place pretrained weights in pretrained_models/densepose/ "
                                    "or pass weights_path to this function.")

    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(cfg_file)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
    predictor = DefaultPredictor(cfg)

    # prepare input
    np_img = np.asarray(pil_image.convert("RGB"))[:, :, ::-1]  # BGR for cv2
    outputs = predictor(np_img)

    # DensePose has 'densepose' key in outputs' instances
    instances = outputs["instances"].to("cpu")
    if not instances.has("pred_densepose"):
        # Visualize segmentation only if densepose missing
        vis = np_img.copy()
    else:
        # Use DensePose visualizer (densepose.visualize has helper, but API differs)
        # We'll create a simple visualization: overlay segmentation mask of all instances
        from detectron2.utils.visualizer import Visualizer
        v = Visualizer(np_img[:, :, ::-1], scale=1.0)
        vis = v.draw_instance_predictions(instances).get_image()  # returns RGB
        vis = vis[:, :, ::-1]  # convert to BGR

    # Save image (convert BGR->RGB for PIL)
    vis_rgb = vis[:, :, ::-1]
    out_path = os.path.join(root, "image-densepose", f"{image_name}.png")
    Image.fromarray(vis_rgb).save(out_path)
    return out_path


# -------------------------
# SCHP Human Parsing wrapper
# -------------------------
def generate_parse_schp(root, pil_image, image_name, schp_checkpoint=None):
    """
    Run SCHP human parsing. Expects SCHP repo present and a pretrained checkpoint placed at
    pretrained_models/schp/schp.pth or provide schp_checkpoint path.

    Returns path to saved segmentation PNG in root/image-parse-v3/{image_name}.png

    NOTE: SCHP's internals vary; we attempt to use the common inference API if available.
    If this wrapper cannot find a working inference routine inside SCHP, it raises an informative error.
    """
    # Try to import SCHP modules
    try:
        # The SCHP repo typically exposes `lib` or `networks` modules under its repo root.
        # We'll attempt a few import names.
        import networks  # common in SCHP forks
    except Exception:
        # try adding the SCHP repo explicitly and import again
        schp_repo = os.path.join(REPO_ROOT, "SCHP")
        if schp_repo not in sys.path:
            sys.path.insert(0, schp_repo)
        try:
            import networks
        except Exception as e:
            raise RuntimeError("Could not import SCHP networks module. Ensure SCHP repo is cloned at ./SCHP and "
                               "that you restarted the kernel after running the setup script.") from e

    # Find checkpoint
    if schp_checkpoint is None:
        candidate = os.path.join(REPO_ROOT, "pretrained_models", "schp", "schp.pth")
        if os.path.exists(candidate):
            schp_checkpoint = candidate
        else:
            raise FileNotFoundError("SCHP checkpoint not found. Please upload schp.pth to pretrained_models/schp/")

    # The SCHP repo typically requires writing a dedicated inference wrapper.
    # Since SCHP code layout varies, we implement a minimal fallback:
    # - Save input to a temp location
    # - Call SCHP's test script if present (e.g., SCHP/inference.py or SCHP/test.py)
    # We'll try a couple of common script names.
    temp_in = os.path.join("/tmp", f"{image_name}_schp_input.jpg")
    pil_image.save(temp_in)

    # Attempt to find an inference script in repo
    schp_scripts = [
        os.path.join(REPO_ROOT, "SCHP", "inference.py"),
        os.path.join(REPO_ROOT, "SCHP", "test.py"),
        os.path.join(REPO_ROOT, "SCHP", "demo.py"),
    ]
    found_script = None
    for s in schp_scripts:
        if os.path.exists(s):
            found_script = s
            break

    if found_script:
        # Run the script as a subprocess and expect it to produce an output segmentation image.
        import subprocess, shlex, tempfile
        out_dir = tempfile.mkdtemp(prefix="schp_out_")
        cmd = f"python {shlex.quote(found_script)} --checkpoint {shlex.quote(schp_checkpoint)} --input {shlex.quote(temp_in)} --output {shlex.quote(out_dir)}"
        print("Running SCHP inference (fallback):", cmd)
        try:
            subprocess.check_call(cmd, shell=True)
            # Expect output as image in out_dir; take first png/jpg file
            files = [f for f in os.listdir(out_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not files:
                raise RuntimeError("SCHP script ran but produced no output images in " + out_dir)
            seg_path = os.path.join(out_dir, files[0])
            # Move to dataset folder
            final_seg = os.path.join(root, "image-parse-v3", f"{image_name}.png")
            Image.open(seg_path).convert("L").save(final_seg)
            return final_seg
        except subprocess.CalledProcessError as e:
            raise RuntimeError("SCHP inference script failed. See the SCHP script's expected args and adapt accordingly.") from e
    else:
        raise RuntimeError("No SCHP inference script found in the SCHP repo. You will need to implement a wrapper using SCHP's model to perform parsing. "
                           "Check the SCHP repo layout and add a small script that accepts an input image and outputs a parsing map.")


# -------------------------
# Parse-agnostic & Agnostic placeholders
# -------------------------
def generate_parse_agnostic(root, parse_path, image_name):
    """
    Generates parse-agnostic image (mask that removes clothing regions) from a parse segmentation map.

    This is a VITON-specific logic: parse-agnostic usually masks head/arms/legs vs clothing labels.
    Exact label mapping depends on the parser. Here we implement a simple rule-based mapping that
    works if the parser uses CIHP/ATR label scheme:

    - labels representing clothes (coat, dress, upper clothes) -> removed (set to 0)
    - keep skin, arms, legs, hair, left/right shoes, left/right pants

    If your SCHP outputs different label IDs, adjust the `clothing_labels` list accordingly.
    """
    import cv2

    seg = np.array(Image.open(parse_path).convert("L"))
    # Typical CIHP label ids for clothing: 5 (upper clothes), 6 (dress), 7 (coat), 11 (skin?), etc.
    # WARNING: This mapping is approximate — please adapt to your parser.
    clothing_labels = {5, 6, 7, 12, 13, 15}  # adjust as needed

    # produce parse-agnostic map: copy parse, set clothing label pixels to 0 (background)
    parse_agnostic = seg.copy()
    for lbl in clothing_labels:
        parse_agnostic[seg == lbl] = 0

    out_path = os.path.join(root, "image-parse-agnostic-v3.2", f"{image_name}.png")
    Image.fromarray(parse_agnostic).save(out_path)
    return out_path


def generate_agnostic_image(root, pil_image, parse_agnostic_path, image_name):
    """
    A simple agnostic generation operation: uses parse-agnostic mask to remove clothing pixels
    by filling them with the average background color or with black.

    This is NOT the official VITON-HD agnostic generator. Replace this with official
    VITON-HD agnostic generator to obtain production-quality agnostic images.
    """
    img_np = np.array(pil_image.convert("RGB"))
    mask = np.array(Image.open(parse_agnostic_path).convert("L"))
    # Consider 0 as background / removed; produce a 3-channel mask where 1 means keep, 0 means remove
    keep = (mask > 0).astype(np.uint8)
    keep3 = np.stack([keep]*3, axis=-1)

    # Fill removed pixels with mean color of background border
    bg_color = img_np.mean(axis=(0, 1)).astype(np.uint8)
    agnostic_np = img_np.copy()
    agnostic_np[keep3 == 0] = bg_color

    out_path = os.path.join(root, "agnostic-v3.2", f"{image_name}.png")
    Image.fromarray(agnostic_np).save(out_path)
    return out_path


# -------------------------
# U2Net cloth mask wrapper
# -------------------------
def generate_cloth_and_mask(root, cloth_pil, cloth_name, u2net_checkpoint=None, device="cuda"):
    """
    Uses U-2-Net from cloned U-2-Net repo to generate cloth mask.
    Expects the U-2-Net repo present at ./U-2-Net and checkpoint at pretrained_models/u2net/u2netp.pth
    """
    # Attempt to import model from repo
    # The U-2-Net repo contains model.py at its root. Ensure sys.path includes its root.
    u2net_repo = os.path.join(REPO_ROOT, "U-2-Net")
    if u2net_repo not in sys.path:
        sys.path.insert(0, u2net_repo)
    try:
        from model import U2NETP  # U2NETP usually found in U-2-Net/model.py
    except Exception as e:
        # Some forks name the class differently; try importing the generic U2NET
        try:
            from model import U2NET
            U2NETP = U2NET
        except Exception as e2:
            raise RuntimeError("Could not import U2NET model from U-2-Net repo. Ensure U-2-Net is cloned and you restarted kernel.") from e2

    import torch
    import torchvision.transforms as T
    import torch.nn.functional as F

    # Check checkpoint
    if u2net_checkpoint is None:
        candidate = os.path.join(REPO_ROOT, "pretrained_models", "u2net", "u2netp.pth")
        if os.path.exists(candidate):
            u2net_checkpoint = candidate
        else:
            raise FileNotFoundError("U2Net checkpoint not found. Upload u2netp.pth to pretrained_models/u2net/ or pass checkpoint path.")

    # Prepare model
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = U2NETP(3, 1)
    state = torch.load(u2net_checkpoint, map_location=device)
    # Some checkpoints are saved directly as state_dict, others as full model dict
    if "state_dict" in state and isinstance(state, dict):
        sd = state["state_dict"]
    else:
        sd = state
    # try loading
    try:
        model.load_state_dict(sd)
    except Exception:
        # some checkpoints have key prefixes
        new_sd = {}
        for k, v in sd.items():
            nk = k.replace("module.", "") if k.startswith("module.") else k
            new_sd[nk] = v
        model.load_state_dict(new_sd)
    model.to(device)
    model.eval()

    # Preprocess cloth image to 512x512
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(cloth_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        d1, d2, d3, d4, d5, d6, d7 = model(input_tensor)
        pred = d1[:, 0, :, :]

        pred = F.interpolate(pred.unsqueeze(0),
                             size=(cloth_pil.size[1], cloth_pil.size[0]),
                             mode='bilinear', align_corners=False).squeeze().cpu().numpy()

    # normalize
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    mask = (pred * 255).astype(np.uint8)
    # threshold
    import cv2
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    _, mask_bin = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    # Save cloth image and mask
    cloth_dir = os.path.join(root, "cloth")
    mask_dir = os.path.join(root, "cloth-mask")
    os.makedirs(cloth_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    cloth_path = os.path.join(cloth_dir, f"{cloth_name}.jpg")
    mask_path = os.path.join(mask_dir, f"{cloth_name}.png")
    cloth_pil.save(cloth_path)
    Image.fromarray(mask_bin).save(mask_path)

    return cloth_path, mask_path


# -------------------------
# Main unified driver
# -------------------------
def preprocess_single_sample_kaggle(
    root,
    person_pil,
    cloth_pil,
    person_name="0001",
    cloth_name="0001_00",
    openpifpaf_checkpoint="resnet50",
    schp_checkpoint=None,
    densepose_cfg=None,
    densepose_weights=None,
    u2net_checkpoint=None
):
    """
    Runs the full Kaggle-friendly VITON style preprocessing pipeline for one sample.
    Saves outputs in the dataset structure under 'root'.
    Returns dict of saved paths.
    """
    # Ensure folder structure
    create_dataset_structure(root)

    # Save original image
    image_path = os.path.join(root, "image", f"{person_name}.jpg")
    person_pil.save(image_path)

    # 1) Pose (OpenPifPaf)
    try:
        pose_img_path, pose_json_path = generate_openpose_with_openpifpaf(root, person_pil, person_name, checkpoint=openpifpaf_checkpoint)
    except Exception as e:
        # Provide a clear message but continue for other steps
        pose_img_path, pose_json_path = None, None
        print("[WARN] OpenPifPaf failed:", e)

    # 2) SCHP parsing (optional)
    parse_path = None
    try:
        parse_path = generate_parse_schp(root, person_pil, person_name, schp_checkpoint)
    except Exception as e:
        print("[WARN] SCHP parsing failed or SCHP not configured:", e)

    # 3) parse-agnostic (if parse exists)
    parse_agn_path = None
    if parse_path is not None:
        try:
            parse_agn_path = generate_parse_agnostic(root, parse_path, person_name)
        except Exception as e:
            print("[WARN] parse-agnostic generation failed:", e)

    # 4) agnostic image (from parse_agnostic)
    agnostic_path = None
    if parse_agn_path is not None:
        try:
            agnostic_path = generate_agnostic_image(root, person_pil, parse_agn_path, person_name)
        except Exception as e:
            print("[WARN] agnostic image generation failed:", e)

    # 5) DensePose (optional)
    densepose_path = None
    try:
        densepose_path = generate_densepose(root, person_pil, person_name, cfg_file=densepose_cfg, weights_path=densepose_weights)
    except Exception as e:
        print("[WARN] DensePose failed or not configured:", e)

    # 6) Cloth + mask via U2Net
    cloth_path, cloth_mask_path = None, None
    try:
        cloth_path, cloth_mask_path = generate_cloth_and_mask(root, cloth_pil, cloth_name, u2net_checkpoint)
    except Exception as e:
        print("[WARN] U2Net cloth/mask generation failed:", e)

    result = {
        "image": image_path,
        "openpose_img": pose_img_path,
        "openpose_json": pose_json_path,
        "parse": parse_path,
        "parse_agnostic": parse_agn_path,
        "agnostic": agnostic_path,
        "densepose": densepose_path,
        "cloth": cloth_path,
        "cloth_mask": cloth_mask_path
    }
    print("[DONE] Preprocessing result:", result)
    return result
