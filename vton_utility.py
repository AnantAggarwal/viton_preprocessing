#!/usr/bin/env python3
"""
main.py

Universal preprocessing module for VITON-HD style pipeline.
- OpenPose BODY-25 via ONNX (onnxruntime)
- DensePose via Detectron2+DensePose (if installed + weights provided)
- Human parsing (SCHP) wrapper (requires SCHP checkpoint and optional inference script)
- Parse-agnostic generation (simple rule-based)
- A basic placeholder agnostic image generator (not official VITON-HD)
- U2Net cloth mask generator (requires U2Net checkpoint)
- Single driver function: preprocess_single_sample(...)

Usage:
    1) source env.sh (from setup.sh) or ensure Python can import the cloned repos
    2) ensure model files are placed under ./models (openpose ONNX, u2net.pth, schp.pth)
    3) call preprocess_single_sample(...)
"""

import os
import sys
import json
import math
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Tuple, List

from PIL import Image, ImageDraw
import numpy as np
import cv2

# Try to import onnxruntime for ONNX OpenPose
try:
    import onnxruntime as ort
except Exception:
    ort = None

# Try detectron2 + densepose
try:
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2 import model_zoo
    from densepose import add_densepose_config
    detectron2_available = True
except Exception:
    detectron2_available = False

# Basic constants
ROOT = os.getcwd()
MODELS_DIR = os.path.join(ROOT, "models")
REPO_DIR = os.path.join(ROOT, "repositories")

###############################################################################
# Utility: dataset folders
###############################################################################
DATASET_SUBDIRS = [
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

def create_dataset_structure(root: str):
    os.makedirs(root, exist_ok=True)
    for d in DATASET_SUBDIRS:
        os.makedirs(os.path.join(root, d), exist_ok=True)

###############################################################################
# OpenPose ONNX (BODY-25) implementation (decoder + rendering + JSON save)
# Note: adapted to be robust for common ONNX conversions (requires output names)
###############################################################################

# BODY_25 keypoint names and limb sequence (common mapping)
BODY_25_KP_NAMES = [
 "nose","neck","r_sho","r_elb","r_wri","l_sho","l_elb","l_wri",
 "r_hip","r_knee","r_ankle","l_hip","l_knee","l_ankle","r_eye","l_eye",
 "r_ear","l_ear","l_bigtoe","l_smalltoe","l_heel","r_bigtoe","r_smalltoe","r_heel","background"
]

# Limb sequence indexes for assembly (commonly used). Validate with your ONNX model's PAF order.
BODY_25_LIMB_SEQ = [
    (1,2),(1,5),(2,3),(3,4),(5,6),(6,7),(1,8),(8,9),(9,10),
    (1,11),(11,12),(12,13),(1,0),(0,14),(14,16),(0,15),(15,17),
    (2,17),(5,16),(10,18),(18,19),(10,20),(11,21),(21,22),(11,24),(8,22)
]

class PoseONNX:
    def __init__(self, onnx_path: str, input_size: Tuple[int,int]=(656,368), heatmap_name: Optional[str]=None, paf_name: Optional[str]=None, device: str='CPU'):
        if ort is None:
            raise RuntimeError("onnxruntime is not installed. Install onnxruntime and retry.")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        providers = ['CPUExecutionProvider']
        if device.upper().startswith('CUDA'):
            providers = ['CUDAExecutionProvider','CPUExecutionProvider']
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_size = input_size  # (W,H)
        # infer output names mapping
        outputs = self.sess.get_outputs()
        out_map = {o.name: tuple(o.shape) for o in outputs}
        # heuristics to pick heatmap & paf
        heat_guess = None
        paf_guess = None
        for name, shape in out_map.items():
            if len(shape) == 4:
                c = shape[1]
                if c in (25, 26):
                    heat_guess = name if heat_guess is None else heat_guess
                if c in (len(BODY_25_LIMB_SEQ)*2, len(BODY_25_LIMB_SEQ)*2 + 2):
                    paf_guess = name if paf_guess is None else paf_guess
        self.heatmap_name = heatmap_name or heat_guess
        self.paf_name = paf_name or paf_guess
        if self.heatmap_name is None or self.paf_name is None:
            raise RuntimeError(f"Could not autodetect heatmap/paf outputs. Available outputs: {list(out_map.keys())}. Provide explicit names.")
        # store shapes
        self.heat_shape = {o.name: o.shape for o in outputs}[self.heatmap_name]
        self.paf_shape = {o.name: o.shape for o in outputs}[self.paf_name]

    # Helpers: resizing/padding
    def _resize_keep_aspect(self, img: np.ndarray, target_size: Tuple[int,int]):
        h,w = img.shape[:2]
        tw, th = target_size
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh))
        pad_w = tw - nw
        pad_h = th - nh
        left = pad_w // 2
        top = pad_h // 2
        resized_padded = cv2.copyMakeBorder(resized, top, pad_h-top, left, pad_w-left, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
        return resized_padded, scale, (left, top)

    def _extract_peaks(self, heatmap: np.ndarray, thresh: float=0.1):
        h,w = heatmap.shape
        hm = cv2.GaussianBlur(heatmap, (3,3), 0)
        peaks = []
        for y in range(1, h-1):
            for x in range(1, w-1):
                v = hm[y,x]
                if v < thresh:
                    continue
                if v >= hm[y-1,x] and v >= hm[y+1,x] and v >= hm[y,x-1] and v >= hm[y,x+1]:
                    peaks.append((x,y,float(v)))
        return peaks

    def _get_peaks_from_heatmaps(self, heatmaps: np.ndarray, thresh: float=0.1):
        n_kpts = heatmaps.shape[0]
        peaks_per_kpt = []
        for k in range(n_kpts):
            peaks = self._extract_peaks(heatmaps[k], thresh=thresh)
            peaks_per_kpt.append(peaks)
        return peaks_per_kpt

    def _score_pair(self, paf_x, paf_y, a, b, num_inter=10, paf_th=0.05):
        ax,ay,_ = a
        bx,by,_ = b
        dx = bx-ax; dy = by-ay
        norm = math.hypot(dx,dy)
        if norm < 1e-6:
            return 0.0, 0
        vx = dx / norm; vy = dy / norm
        xs = np.linspace(ax,bx,num=num_inter).astype(np.int32)
        ys = np.linspace(ay,by,num=num_inter).astype(np.int32)
        h,w = paf_x.shape
        score_sum = 0.0; count=0
        for xi, yi in zip(xs, ys):
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                continue
            val_x = paf_x[yi, xi]; val_y = paf_y[yi, xi]
            score = val_x * vx + val_y * vy
            if score > paf_th:
                score_sum += score
                count += 1
        if count == 0:
            return 0.0, 0
        return score_sum / count, count

    def _find_connections(self, peaks_per_kpt, pafs, limb_seq, min_consensus=3, inter_points=10, paf_th=0.05):
        h, w = pafs.shape[1], pafs.shape[2]
        connections_all = []
        for limb_idx, (a_idx, b_idx) in enumerate(limb_seq):
            paf_x = pafs[2*limb_idx]
            paf_y = pafs[2*limb_idx+1]
            candA = peaks_per_kpt[a_idx]; candB = peaks_per_kpt[b_idx]
            conn_candidates = []
            if len(candA)==0 or len(candB)==0:
                connections_all.append([])
                continue
            for i,a in enumerate(candA):
                for j,b in enumerate(candB):
                    score, count = self._score_pair(paf_x, paf_y, a, b, num_inter=inter_points, paf_th=paf_th)
                    if count >= min_consensus and score>0:
                        conn_candidates.append((i,j,score))
            conn_candidates.sort(key=lambda x: x[2], reverse=True)
            usedA, usedB=set(), set(); connections=[]
            for (i,j,s) in conn_candidates:
                if i in usedA or j in usedB:
                    continue
                usedA.add(i); usedB.add(j)
                connections.append((i,j,s))
            connections_all.append(connections)
        return connections_all

    def _assemble_people(self, peaks_per_kpt, connections_all, limb_seq):
        persons=[]
        for limb_idx, connections in enumerate(connections_all):
            a_idx, b_idx = limb_seq[limb_idx]
            for (i,j,score) in connections:
                assigned=False
                for person in persons:
                    if person.get(a_idx)==(i,):
                        person[b_idx] = (j,)
                        assigned=True
                        break
                    if person.get(b_idx)==(j,):
                        person[a_idx] = (i,)
                        assigned=True
                        break
                if not assigned:
                    person = {}
                    person[a_idx] = (i,); person[b_idx]=(j,)
                    persons.append(person)
        final=[]
        n_kpt = len(peaks_per_kpt)
        for person in persons:
            kp_coords={}
            for k in range(n_kpt):
                if k in person:
                    idx = person[k][0]
                    x,y,s = peaks_per_kpt[k][idx]
                    kp_coords[k] = (float(x), float(y), float(s))
                else:
                    kp_coords[k] = (0.0, 0.0, 0.0)
            final.append(kp_coords)
        if len(final)==0:
            single={}
            for k,peaks in enumerate(peaks_per_kpt):
                if len(peaks)>0:
                    x,y,s = peaks[0]; single[k]=(float(x),float(y),float(s))
                else:
                    single[k]=(0.0,0.0,0.0)
            final.append(single)
        return final

    def infer(self, pil_image: Image.Image, heatmap_thresh: float=0.1, paf_th: float=0.05):
        if ort is None:
            raise RuntimeError("onnxruntime required for PoseONNX.")
        np_img = np.array(pil_image.convert("RGB"))
        orig_h, orig_w = np_img.shape[:2]
        target_w, target_h = self.input_size
        resized, scale, (left, top) = self._resize_keep_aspect(np_img, (target_w, target_h))
        input_blob = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR).astype(np.float32) / 255.0
        input_blob = np.transpose(input_blob, (2,0,1))[None, ...].astype(np.float32)
        input_name = self.sess.get_inputs()[0].name
        outs = self.sess.run(None, {input_name: input_blob})
        out_names = [o.name for o in self.sess.get_outputs()]
        out_map = {n:a for n,a in zip(out_names, outs)}
        heatmaps = out_map[self.heatmap_name][0]  # C x h x w
        pafs = out_map[self.paf_name][0]         # Cp x h x w
        # get peaks
        peaks_per_kpt = self._get_peaks_from_heatmaps(heatmaps, thresh=heatmap_thresh)
        # find connections
        connections_all = self._find_connections(peaks_per_kpt, pafs, BODY_25_LIMB_SEQ, min_consensus=3, inter_points=10, paf_th=paf_th)
        persons = self._assemble_people(peaks_per_kpt, connections_all, BODY_25_LIMB_SEQ)
        # map to original coordinates
        hmap_h, hmap_w = heatmaps.shape[1], heatmaps.shape[2]
        mapped=[]
        for p in persons:
            mp={}
            for k,(x_hm,y_hm,conf) in p.items():
                x_resized = x_hm * (resized.shape[1] / hmap_w)
                y_resized = y_hm * (resized.shape[0] / hmap_h)
                x_orig = (x_resized - left) / scale
                y_orig = (y_resized - top) / scale
                x_orig = max(0.0, min(orig_w-1.0, x_orig))
                y_orig = max(0.0, min(orig_h-1.0, y_orig))
                mp[k] = (x_orig, y_orig, float(conf))
            mapped.append(mp)
        return mapped

    def render(self, pil_image: Image.Image, persons: List[Dict[int, Tuple[float,float,float]]], line_w: int=3, kp_r:int=3):
        img = pil_image.convert("RGB")
        draw = ImageDraw.Draw(img)
        for p in persons:
            for (a_idx,b_idx) in BODY_25_LIMB_SEQ:
                ax,ay,ac = p.get(a_idx,(0,0,0))
                bx,by,bc = p.get(b_idx,(0,0,0))
                if ac>0.01 and bc>0.01:
                    draw.line(((ax,ay),(bx,by)), width=line_w, fill=(0,255,0))
            for k_idx in range(len(BODY_25_KP_NAMES)):
                x,y,c = p.get(k_idx,(0,0,0))
                if c>0.01:
                    r=kp_r
                    draw.ellipse((x-r,y-r,x+r,y+r), fill=(255,0,0))
        return img

    def save_openpose_json(self, persons: List[Dict[int,Tuple[float,float,float]]], out_path: str):
        people=[]
        for p in persons:
            pose=[]
            for k in range(len(BODY_25_KP_NAMES)):
                x,y,c = p.get(k,(0.0,0.0,0.0))
                pose.extend([float(x),float(y),float(c)])
            people.append({
                "person_id": [-1],
                "pose_keypoints_2d": pose,
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
                "pose_keypoints_3d": [],
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": []
            })
        out={"version":1.3,"people":people}
        with open(out_path,"w") as f:
            json.dump(out,f,indent=2)

###############################################################################
# DensePose wrapper (Detectron2 + DensePose)
###############################################################################
def generate_densepose(root: str, pil_image: Image.Image, name: str, cfg_file: Optional[str]=None, weights: Optional[str]=None) -> str:
    """
    Runs DensePose (Detectron2) and saves visualized result to root/image-densepose/{name}.png
    Requires detectron2 + densepose installed and a weights file.
    """
    if not detectron2_available:
        raise RuntimeError("Detectron2/DensePose not available. Install them following README or setup script.")
    if cfg_file is None:
        # default candidate config inside cloned DensePose repo
        cfg_candidate = os.path.join(REPO_DIR, "DensePose", "configs", "densepose_rcnn_R_50_FPN_s1x.yaml")
        if os.path.exists(cfg_candidate):
            cfg_file = cfg_candidate
        else:
            raise FileNotFoundError("DensePose config not provided and not found in repo.")
    if weights is None:
        candidate = os.path.join(MODELS_DIR, "DensePose_ResNet50_FPN_s1x.pkl")
        if os.path.exists(candidate):
            weights = candidate
        else:
            raise FileNotFoundError("DensePose weights not provided; place .pkl in models/ or pass weights path.")

    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(cfg_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
    predictor = DefaultPredictor(cfg)
    np_img = np.array(pil_image.convert("RGB"))[:, :, ::-1]
    outputs = predictor(np_img)
    instances = outputs["instances"].to("cpu")
    from detectron2.utils.visualizer import Visualizer
    v = Visualizer(np_img[:, :, ::-1], scale=1.0)
    vis = v.draw_instance_predictions(instances).get_image()  # RGB
    out_path = os.path.join(root, "image-densepose", f"{name}.png")
    Image.fromarray(vis).save(out_path)
    return out_path

###############################################################################
# SCHP parsing wrapper (best-effort)
###############################################################################
def generate_parse_schp(root: str, pil_image: Image.Image, name: str, schp_checkpoint: Optional[str]=None) -> str:
    """
    Runs SCHP human parsing. This wrapper attempts to find a standard SCHP inference script in the cloned repo.
    If none exists, it raises instructive error.
    """
    schp_repo = os.path.join(REPO_DIR, "SCHP")
    if schp_repo not in sys.path:
        sys.path.insert(0, schp_repo)
    # find possible inference script
    possible = [
        os.path.join(schp_repo, "inference.py"),
        os.path.join(schp_repo, "test.py"),
        os.path.join(schp_repo, "demo.py"),
        os.path.join(schp_repo, "tools", "inference.py")
    ]
    found = None
    for p in possible:
        if os.path.exists(p):
            found = p
            break
    if found is None:
        raise RuntimeError("No SCHP inference script found in SCHP repo. Please implement or adapt SCHP inference script. "
                           f"Checked: {possible}")
    # prepare input tmp file and output dir
    tmp_in = os.path.join(tempfile.gettempdir(), f"{name}_schp_in.jpg")
    pil_image.save(tmp_in)
    out_dir = os.path.join(tempfile.gettempdir(), f"schp_out_{name}")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    # craft command - you may need to modify args to match the repo's interface
    cmd = f"python {found} --input {tmp_in} --output {out_dir}"
    if schp_checkpoint:
        cmd += f" --checkpoint {schp_checkpoint}"
    print("[SCHP] Running:", cmd)
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError("SCHP inference failed. Check SCHP script args and checkpoint.")
    # find first image in out_dir
    outs = [f for f in os.listdir(out_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    if not outs:
        raise RuntimeError("SCHP script produced no output images.")
    seg_path_src = os.path.join(out_dir, outs[0])
    seg_path_dst = os.path.join(root, "image-parse-v3", f"{name}.png")
    Image.open(seg_path_src).convert("L").save(seg_path_dst)
    return seg_path_dst

###############################################################################
# Parse-agnostic generator (simple rule-based)
###############################################################################
def generate_parse_agnostic_from_parse(root: str, parse_path: str, name: str) -> str:
    """
    Simple parse-agnostic: set pixels of clothing labels to zero (background).
    The mapping depends on parser label scheme; adjust clothing_labels for your parser.
    """
    seg = np.array(Image.open(parse_path).convert("L"))
    # Approximate clothing labels for CIHP/ATR: adjust as needed
    clothing_labels = {5,6,7,12,13,15}
    parse_agn = seg.copy()
    for lbl in clothing_labels:
        parse_agn[seg == lbl] = 0
    out = os.path.join(root, "image-parse-agnostic-v3.2", f"{name}.png")
    Image.fromarray(parse_agn).save(out)
    return out

###############################################################################
# Simple "agnostic" image generator (placeholder, NOT VITON-HD official)
###############################################################################
def generate_agnostic_placeholder(root: str, pil_image: Image.Image, parse_agn_path: str, name: str) -> str:
    img = np.array(pil_image.convert("RGB"))
    mask = np.array(Image.open(parse_agn_path).convert("L"))
    keep = (mask > 0).astype(np.uint8)
    keep3 = np.stack([keep]*3, axis=-1)
    bg_color = img.mean(axis=(0,1)).astype(np.uint8)
    out_img = img.copy()
    out_img[keep3==0] = bg_color
    out = os.path.join(root, "agnostic-v3.2", f"{name}.png")
    Image.fromarray(out_img).save(out)
    return out

###############################################################################
# U2Net cloth mask generator
###############################################################################
def generate_cloth_and_mask(root: str, cloth_pil: Image.Image, cloth_name: str, u2net_checkpoint: Optional[str]=None, device: str='cpu') -> Tuple[str,str]:
    # ensure repo import path
    u2_repo = os.path.join(REPO_DIR, "U-2-Net")
    if u2_repo not in sys.path:
        sys.path.insert(0, u2_repo)
    try:
        from model import U2NETP
    except Exception:
        try:
            from model import U2NET
            U2NETP = U2NET
        except Exception as e:
            raise RuntimeError("Could not import U2Net model from repo. Ensure U-2-Net cloned and env.sh sourced.") from e
    import torch
    import torchvision.transforms as T
    import torch.nn.functional as F
    ckpt = u2net_checkpoint or os.path.join(MODELS_DIR, "u2net.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"U2Net checkpoint not found: {ckpt}")
    device_t = torch.device(device if torch.cuda.is_available() and device.lower().startswith('cuda') else 'cpu')
    model = U2NETP(3,1)
    state = torch.load(ckpt, map_location=device_t)
    if isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    else:
        sd = state
    new_sd={}
    for k,v in sd.items():
        nk = k.replace("module.","") if k.startswith("module.") else k
        new_sd[nk]=v
    model.load_state_dict(new_sd)
    model.to(device_t).eval()
    # preprocess
    transform = T.Compose([T.Resize((512,512)), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    input_tensor = transform(cloth_pil).unsqueeze(0).to(device_t)
    with torch.no_grad():
        outs = model(input_tensor)
        if isinstance(outs, tuple):
            pred = outs[0][:,0,:,:]
        else:
            pred = outs[:,0,:,:]
        pred = F.interpolate(pred.unsqueeze(0), size=(cloth_pil.size[1], cloth_pil.size[0]), mode='bilinear', align_corners=False).squeeze().cpu().numpy()
    pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    mask = (pred_norm * 255).astype(np.uint8)
    mask = cv2.GaussianBlur(mask, (3,3), 0)
    _, mask_bin = cv2.threshold(mask, 12, 255, cv2.THRESH_BINARY)
    cloth_dir = os.path.join(root, "cloth"); mask_dir = os.path.join(root, "cloth-mask")
    os.makedirs(cloth_dir, exist_ok=True); os.makedirs(mask_dir, exist_ok=True)
    cloth_path = os.path.join(cloth_dir, f"{cloth_name}.jpg")
    mask_path = os.path.join(mask_dir, f"{cloth_name}.png")
    cloth_pil.save(cloth_path)
    Image.fromarray(mask_bin).save(mask_path)
    return cloth_path, mask_path

###############################################################################
# Convenience: generate_openpose_from_pil wrapper (uses PoseONNX)
###############################################################################
def generate_openpose_from_pil(root: str, pil_image: Image.Image, name: str, onnx_path: Optional[str]=None, input_size: Tuple[int,int]=(656,368), device: str='CPU', heatmap_name: Optional[str]=None, paf_name: Optional[str]=None) -> Tuple[str,str]:
    onnx_path = onnx_path or os.path.join(MODELS_DIR, "openpose_body_25.onnx")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"OpenPose ONNX model not found at {onnx_path}. Place it in models/ or pass onnx_path.")
    pose = PoseONNX(onnx_path, input_size=input_size, heatmap_name=heatmap_name, paf_name=paf_name, device=device)
    persons = pose.infer(pil_image)
    rendered = pose.render(pil_image, persons)
    os.makedirs(os.path.join(root, "openpose-img"), exist_ok=True)
    os.makedirs(os.path.join(root, "openpose-json"), exist_ok=True)
    img_out = os.path.join(root, "openpose-img", f"{name}.png")
    json_out = os.path.join(root, "openpose-json", f"{name}.json")
    rendered.save(img_out)
    pose.save_openpose_json(persons, json_out)
    return img_out, json_out

###############################################################################
# Main driver
###############################################################################
def preprocess_single_sample(
    image_path: str,
    cloth_path: str,
    root: str,
    name: str = "000001",
    onnx_path: Optional[str] = None,
    schp_checkpoint: Optional[str] = None,
    densepose_cfg: Optional[str] = None,
    densepose_weights: Optional[str] = None,
    u2net_checkpoint: Optional[str] = None
) -> Dict[str, Optional[str]]:
    """
    Run the entire preprocessing pipeline for one sample.
    Saves outputs to dataset structure under `root` and returns file paths dict.
    """
    create_dataset_structure(root)
    img = Image.open(image_path).convert("RGB")
    cloth = Image.open(cloth_path).convert("RGB")
    # save original
    img_save = os.path.join(root, "image", f"{name}.jpg")
    img.save(img_save)

    # 1) OpenPose ONNX
    try:
        openpose_img, openpose_json = generate_openpose_from_pil(root, img, name, onnx_path=onnx_path)
    except Exception as e:
        openpose_img, openpose_json = None, None
        print("[WARN] OpenPose (ONNX) failed:", e)

    # 2) SCHP parsing
    try:
        parse_path = generate_parse_schp(root, img, name, schp_checkpoint)
    except Exception as e:
        parse_path = None
        print("[WARN] SCHP parse failed or not configured:", e)

    # 3) parse-agnostic
    parse_agn = None
    if parse_path:
        try:
            parse_agn = generate_parse_agnostic_from_parse(root, parse_path, name)
        except Exception as e:
            print("[WARN] parse-agnostic generation failed:", e)

    # 4) agnostic placeholder
    agnostic = None
    if parse_agn:
        try:
            agnostic = generate_agnostic_placeholder(root, img, parse_agn, name)
        except Exception as e:
            print("[WARN] agnostic placeholder failed:", e)

    # 5) DensePose
    densepose_path = None
    try:
        densepose_path = generate_densepose(root, img, name, cfg_file=densepose_cfg, weights=densepose_weights) if detectron2_available else None
    except Exception as e:
        densepose_path = None
        print("[WARN] DensePose failed or not configured:", e)

    # 6) cloth & mask
    cloth_out, cloth_mask = None, None
    try:
        cloth_out, cloth_mask = generate_cloth_and_mask(root, cloth, f"{name}_cloth", u2net_checkpoint)
    except Exception as e:
        print("[WARN] U2Net cloth generation failed:", e)

    result = {
        "image": img_save,
        "openpose_img": openpose_img,
        "openpose_json": openpose_json,
        "parse": parse_path,
        "parse_agnostic": parse_agn,
        "agnostic": agnostic,
        "densepose": densepose_path,
        "cloth": cloth_out,
        "cloth_mask": cloth_mask
    }
    print("[DONE] Preprocessing result:", result)
    return result

###############################################################################
# If run as script: small CLI example
###############################################################################
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to person image")
    parser.add_argument("--cloth", required=True, help="Path to cloth image")
    parser.add_argument("--root", default="./dataset", help="Root output directory")
    parser.add_argument("--name", default="000001", help="Output name prefix")
    parser.add_argument("--onnx", default=os.path.join(MODELS_DIR, "openpose_body_25.onnx"), help="OpenPose BODY25 ONNX path")
    parser.add_argument("--schp", default=os.path.join(MODELS_DIR, "schp.pth"), help="SCHP checkpoint (optional)")
    parser.add_argument("--u2net", default=os.path.join(MODELS_DIR, "u2net.pth"), help="U2Net checkpoint")
    parser.add_argument("--densepose_cfg", default=None, help="DensePose config YAML (optional)")
    parser.add_argument("--densepose_weights", default=None, help="DensePose weights (optional)")
    args = parser.parse_args()

    preprocess_single_sample(
        image_path=args.image,
        cloth_path=args.cloth,
        root=args.root,
        name=args.name,
        onnx_path=args.onnx,
        schp_checkpoint=(args.schp if os.path.exists(args.schp) else None),
        densepose_cfg=args.densepose_cfg,
        densepose_weights=args.densepose_weights,
        u2net_checkpoint=(args.u2net if os.path.exists(args.u2net) else None)
    )
