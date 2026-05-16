import os
import zipfile
import tempfile
import torch
import numpy as np
import cv2
from PIL import Image
import nibabel as nib
import gdown

GDRIVE_FILE_ID = "1cGNcrxizRowJhnP7RE9GFWsVRAwLzV13"
MODEL_CACHE_PATH = os.path.join(tempfile.gettempdir(), "brainalyze_3d_seg.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_3d_model_cache = None

# BraTS-standard colours (RGB) for each tumour sub-region
CLASS_COLORS_RGB = {
    1: (200,  90,  10),  # NCR/NET  — dark orange  rgb(200,90,10)
    2: (220, 180,   0),  # Edema    — gold/yellow  rgb(220,180,0)
    3: (235,  90,  70),  # ET       — light coral  rgb(235,90,70)
}


# ─── Model download & build ──────────────────────────────────────────────────

def download_3d_model_if_needed():
    if os.path.exists(MODEL_CACHE_PATH) and os.path.getsize(MODEL_CACHE_PATH) > 1_000_000:
        print("[3D Model] Model already cached at", MODEL_CACHE_PATH)
        return
    # Remove a corrupted/partial file before retrying
    if os.path.exists(MODEL_CACHE_PATH):
        os.remove(MODEL_CACHE_PATH)
        print("[3D Model] Removed incomplete/corrupted cache, re-downloading...")
    print("[3D Model] Downloading 3D segmentation model from Google Drive...")
    try:
        gdown.download(id=GDRIVE_FILE_ID, output=MODEL_CACHE_PATH, quiet=False)
    except Exception as dl_err:
        if os.path.exists(MODEL_CACHE_PATH):
            os.remove(MODEL_CACHE_PATH)
        raise RuntimeError(
            f"Failed to download the 3D segmentation model from Google Drive. "
            f"Check your internet connection and try again. Details: {dl_err}"
        )
    if not os.path.exists(MODEL_CACHE_PATH) or os.path.getsize(MODEL_CACHE_PATH) < 1_000_000:
        if os.path.exists(MODEL_CACHE_PATH):
            os.remove(MODEL_CACHE_PATH)
        raise RuntimeError(
            "3D model download incomplete or file is too small. "
            "Please check your internet connection and try again."
        )
    print("[3D Model] Download complete.")


def build_3d_model():
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
    model = PlainConvUNet(
        input_channels=4,
        n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=torch.nn.Conv3d,
        kernel_sizes=[[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
        strides=[[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
        n_conv_per_stage=[2,2,2,2,2,2],
        n_conv_per_stage_decoder=[2,2,2,2,2],
        conv_bias=True,
        norm_op=torch.nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=True,
        num_classes=4,
    )
    return model


def load_3d_model():
    global _3d_model_cache
    if _3d_model_cache is not None:
        return _3d_model_cache
    download_3d_model_if_needed()
    model = build_3d_model()
    data = torch.load(MODEL_CACHE_PATH, map_location=device, weights_only=False)
    model.load_state_dict(data["network_weights"])
    model.to(device)
    model.eval()
    print("[3D Model] Loaded successfully —", sum(p.numel() for p in model.parameters()) / 1e6, "M params")
    _3d_model_cache = model
    return model


# ─── Preprocessing ───────────────────────────────────────────────────────────

def zscore_normalize(volume):
    fg = volume[volume > 0]
    if fg.size == 0:
        return volume
    mean = float(fg.mean())
    std  = float(fg.std()) or 1.0
    return (volume - mean) / std


def load_nifti_volume(nifti_path):
    img  = nib.load(nifti_path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim == 3:
        data = np.transpose(data, (2, 0, 1))
    return data


def load_zip_nifti(zip_path):
    """
    Extract a ZIP that contains exactly 4 .nii files (FLAIR, T1, T1CE, T2).

    Returns:
        channels        : list of 4 numpy arrays shaped (D, H, W)
        voxel_spacing   : tuple (dz, dy, dx) — voxel size in mm (from FLAIR header)
        affine          : 4×4 numpy array — NIfTI affine for the FLAIR image
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tmpdir)

        nii_files = []
        for root, _, files in os.walk(tmpdir):
            for f in sorted(files):
                if f.endswith('.nii') or f.endswith('.nii.gz'):
                    nii_files.append(os.path.join(root, f))

        if len(nii_files) == 0:
            raise ValueError("No .nii / .nii.gz files found inside the ZIP.")

        # ── Map filenames to modalities ────────────────────────────────────────
        modality_map = {}
        for f in nii_files:
            fname = os.path.basename(f).lower()
            if 'flair' in fname:
                modality_map['flair'] = f
            elif 't1ce' in fname or 't1c' in fname:
                modality_map['t1ce'] = f
            elif 't1' in fname:
                modality_map['t1'] = f
            elif 't2' in fname:
                modality_map['t2'] = f

        # If no modality keyword matched, treat the first file as FLAIR
        if not modality_map:
            modality_map['flair'] = nii_files[0]

        # ── Load the best available reference (prefer FLAIR) ──────────────────
        ref_mod  = 'flair' if 'flair' in modality_map else next(iter(modality_map))
        ref_path = modality_map[ref_mod]
        ref_img  = nib.load(ref_path)
        ref_data = ref_img.get_fdata(dtype=np.float32)
        if ref_data.ndim == 4:
            ref_data = ref_data[..., 0]
        ref_data = np.transpose(ref_data, (2, 0, 1))   # (X,Y,Z) → (D,H,W)

        sx, sy, sz = [float(v) for v in ref_img.header.get_zooms()[:3]]
        voxel_spacing = (sz, sx, sy)
        affine        = ref_img.affine.copy()

        # ── Build 4-channel list (fill missing with reference) ────────────────
        channels = []
        for mod in ['flair', 't1', 't1ce', 't2']:
            if mod in modality_map:
                img  = nib.load(modality_map[mod])
                data = img.get_fdata(dtype=np.float32)
                if data.ndim == 4:
                    data = data[..., 0]
                data = np.transpose(data, (2, 0, 1))
            else:
                # Missing modality — substitute with reference channel
                data = ref_data.copy()
            channels.append(data)

        print(f"[ZIP] Loaded modalities: {list(modality_map.keys())}  "
              f"(missing filled with '{ref_mod}')")
        return channels, voxel_spacing, affine


# ─── 3D tumour volume & anatomical location ───────────────────────────────────

def _to_ras(mm_vec, affine):
    """
    Normalize world-space coordinates to RAS using the affine axis codes.
    Handles RAS, LPS, LAS, and any NIfTI orientation.
    """
    import nibabel.orientations as nibo
    try:
        ax = nibo.aff2axcodes(affine)          # e.g. ('L','A','S') or ('R','A','S')
        signs = [
            1 if ax[0] == 'R' else -1,         # x: Right positive
            1 if ax[1] == 'A' else -1,         # y: Anterior positive
            1 if ax[2] == 'S' else -1,         # z: Superior positive
        ]
        return float(mm_vec[0]) * signs[0], float(mm_vec[1]) * signs[1], float(mm_vec[2]) * signs[2]
    except Exception:
        return float(mm_vec[0]), float(mm_vec[1]), float(mm_vec[2])


def _anatomical_location(cz, cy, cx, D, H, W, affine, pred_label=None):
    """
    Precise anatomical location using:
    - Orientation-corrected RAS coordinates (fixes LPS/RAS confusion)
    - Enhancing Tumor centroid when available (more clinically relevant)
    - MNI152-calibrated lobe/depth thresholds
    - Deep structure & infratentorial detection
    """
    # ── 1. Pick best centroid: ET > whole tumour ──────────────────────────────
    lz, ly, lx = cz, cy, cx
    if pred_label is not None:
        et = (pred_label == 3)
        if int(et.sum()) > 50:
            zz, yy, xx = np.where(et)
            lz, ly, lx = float(zz.mean()), float(yy.mean()), float(xx.mean())

    # ── 2. Voxel → world mm → RAS ─────────────────────────────────────────────
    # After load_zip_nifti transpose (2,0,1):
    #   D-axis (lz) = original nibabel k  (axial / Superior-Inferior)
    #   H-axis (ly) = original nibabel i  (Left-Right)
    #   W-axis (lx) = original nibabel j  (Anterior-Posterior)
    use_affine = False
    x_ras = y_ras = z_ras = 0.0
    try:
        vox       = np.array([ly, lx, lz, 1.0])   # [i, j, k, 1] in nibabel space
        mm        = affine @ vox
        x_ras, y_ras, z_ras = _to_ras(mm, affine)
        use_affine = True
    except Exception as e:
        print(f"[Location] Affine transform failed: {e}")

    if use_affine:
        # ── Hemisphere (RAS x: positive = Right) ─────────────────────────────
        if x_ras > 8:
            hemi = "Right hemisphere"
        elif x_ras < -8:
            hemi = "Left hemisphere"
        else:
            hemi = "Midline / bilateral"

        # ── Infratentorial first (below tentorium) ────────────────────────────
        if z_ras < -15:
            if y_ras < -40:
                lobe = "cerebellum"
            else:
                lobe = "brainstem / posterior fossa"
            level = "infratentorial"

        # ── Deep / subcortical structures ─────────────────────────────────────
        elif z_ras < 10 and abs(x_ras) < 22 and -15 < y_ras < 20:
            lobe  = "deep / subcortical (basal ganglia — thalamus region)"
            level = "subcortical"

        # ── Supratentorial lobes ──────────────────────────────────────────────
        else:
            # Frontal  (anterior to central sulcus: y_ras > 0 in MNI)
            if y_ras > 5:
                if z_ras > 45:
                    lobe  = "superior frontal lobe"
                    level = "high convexity"
                elif z_ras > 15:
                    lobe  = "frontal lobe"
                    level = "mid convexity"
                else:
                    lobe  = "inferior frontal / orbitofrontal region"
                    level = "low convexity"

            # Central / peri-rolandic
            elif y_ras > -15:
                lobe  = "central region (peri-rolandic)"
                level = "high convexity" if z_ras > 40 else "mid convexity"

            # Parietal vs Temporal (differentiated by depth/z)
            elif y_ras > -60:
                if z_ras >= 12 or abs(x_ras) < 35:
                    lobe  = "parietal lobe"
                    level = "high convexity" if z_ras > 40 else "mid convexity"
                else:
                    lobe  = "temporal lobe"
                    level = "perisylvian" if z_ras > 0 else "inferior temporal"

            # Occipital
            else:
                if z_ras < 10 and abs(x_ras) > 30:
                    lobe  = "occipital-temporal (fusiform) region"
                    level = "inferior"
                else:
                    lobe  = "occipital lobe"
                    level = "high convexity" if z_ras > 30 else "mid-level"

    else:
        # ── Pure relative-position fallback ───────────────────────────────────
        hemi  = "Right hemisphere" if lx / W > 0.52 else ("Left hemisphere" if lx / W < 0.48 else "Bilateral")
        rz    = lz / D
        level = "high convexity" if rz > 0.65 else ("mid convexity" if rz > 0.35 else "low / deep")
        ry    = ly / H
        lobe  = ("frontal lobe"        if ry < 0.33 else
                 "parietal lobe"       if ry < 0.50 else
                 "temporal lobe"       if ry < 0.70 else
                 "occipital lobe")

    return f"{hemi} — {lobe} ({level})"


def compute_3d_tumor_metrics(pred_label, voxel_spacing_mm, affine=None):
    """
    Calculate tumour volume (in cm³) and anatomical location from a 3D
    segmentation mask.

    Formula (Source: nibabel docs https://nipy.org/nibabel/reference/nibabel.imagestats.html
             + Neurostars https://neurostars.org/t/calculate-volumes-counting-voxels-is-a-correct-approach/1433):
        Volume (mm³) = n_voxels × voxel_x (mm) × voxel_y (mm) × voxel_z (mm)
        Volume (cm³) = Volume (mm³) / 1000
    Voxel dimensions are read from the NIfTI header pixdim[1:4].
    cm³ is the standard clinical unit for tumour volume reporting.

    Args:
        pred_label      : (D, H, W) uint8 array with class labels 0-3
        voxel_spacing_mm: tuple (dz, dy, dx) in mm  — from load_zip_nifti
        affine          : 4×4 NIfTI affine           — from load_zip_nifti

    Returns:
        dict:
            total_volume_cm3   – float, whole-tumour volume in cm³
            ncr_volume_cm3     – float, necrotic core in cm³
            ed_volume_cm3      – float, peritumoral edema in cm³
            et_volume_cm3      – float, enhancing tumour in cm³
            location_text      – str  human-readable anatomical location
            centroid_voxel     – (cx, cy, cz) tuple
    """
    from scipy.ndimage import label as scipy_label

    dz, dy, dx  = voxel_spacing_mm
    vox_vol_mm3 = dz * dy * dx

    D, H, W    = pred_label.shape
    tumor_mask = pred_label > 0
    n_total    = int(tumor_mask.sum())
    aff        = affine if affine is not None else np.eye(4)

    # ── Overall centroid (fallback reference) ────────────────────────────────
    if n_total > 0:
        zz, yy, xx = np.where(tumor_mask)
        cz = float(zz.mean())
        cy = float(yy.mean())
        cx = float(xx.mean())
    else:
        cz, cy, cx = D / 2, H / 2, W / 2

    def _vol(mask):
        # Volume (cm³) = n_voxels × voxel_x × voxel_y × voxel_z / 1000
        # Source: nibabel docs (https://nipy.org/nibabel/reference/nibabel.imagestats.html)
        # 1 cm³ = 1000 mm³ — cm³ is the standard clinical unit for tumour volume
        return round(int(mask.sum()) * vox_vol_mm3 / 1000.0, 2)

    # ── Connected-component location analysis ────────────────────────────────
    # Minimum component size to report (≈0.3 cm³ at 1 mm isotropic)
    MIN_VOXELS = max(300, int(300 / vox_vol_mm3))

    labeled_arr, n_comp = scipy_label(tumor_mask)

    # Collect (size, location_string) for every significant component
    component_locs = []
    for cid in range(1, n_comp + 1):
        comp = (labeled_arr == cid)
        n_vox = int(comp.sum())
        if n_vox < MIN_VOXELS:
            continue

        # Mask pred_label to this component so ET centroid is component-local
        comp_pred = pred_label.copy()
        comp_pred[~comp] = 0

        zz_c, yy_c, xx_c = np.where(comp)
        cz_c = float(zz_c.mean())
        cy_c = float(yy_c.mean())
        cx_c = float(xx_c.mean())

        loc_c = _anatomical_location(cz_c, cy_c, cx_c, D, H, W, aff,
                                     pred_label=comp_pred)
        component_locs.append((n_vox, loc_c))

    # Deduplicate while preserving size order (largest first)
    component_locs.sort(key=lambda t: t[0], reverse=True)
    seen, unique_locs = set(), []
    for _, loc_c in component_locs:
        if loc_c not in seen:
            seen.add(loc_c)
            unique_locs.append(loc_c)

    if not unique_locs:
        # Absolute fallback — use overall centroid
        unique_locs = [_anatomical_location(cz, cy, cx, D, H, W, aff,
                                            pred_label=pred_label)]

    if len(unique_locs) == 1:
        location_text = unique_locs[0]
    else:
        parts = [f"({i + 1}) {l}" for i, l in enumerate(unique_locs)]
        location_text = "Multifocal — " + "  |  ".join(parts)

    return {
        "total_volume_cm3": _vol(tumor_mask),
        "ncr_volume_cm3":   _vol(pred_label == 1),
        "ed_volume_cm3":    _vol(pred_label == 2),
        "et_volume_cm3":    _vol(pred_label == 3),
        "location_text":    location_text,
        "location_list":    unique_locs,
        "n_foci":           len(unique_locs),
        "centroid_voxel":   (round(cx, 1), round(cy, 1), round(cz, 1)),
    }


# ─── 3D inference (single-channel / pseudo-3D) ───────────────────────────────

def prepare_volume_for_3d_model(volume, patch_size=(128, 128, 128)):
    """
    Prepare a single-modality NIfTI volume for the 4-channel BraTS nnUNet model.

    Channel assignment rationale (BraTS2020 order: FLAIR, T1, T1CE, T2):
      ch 0  FLAIR  — use the uploaded volume (most likely FLAIR or T2-weighted)
      ch 1  T1     — zero-filled  (cannot be estimated from FLAIR; honest absence)
      ch 2  T1CE   — zero-filled  (gadolinium enhancement unknown from FLAIR)
      ch 3  T2     — copy of ch 0 (T2-weighted; BraTS FLAIR & T2 share bright-edema contrast)

    Replicating all 4 channels with the same data caused the model to produce
    NCR everywhere because T1 (which should be dark in NCR) looked identical to
    FLAIR, sending a contradictory signal.
    """
    from scipy.ndimage import zoom
    norm   = zscore_normalize(volume.copy())
    D,H,W  = norm.shape
    pd,ph,pw = patch_size
    resized  = zoom(norm, (pd/D, ph/H, pw/W), order=1).astype(np.float32)

    vol_4ch = np.zeros((4, pd, ph, pw), dtype=np.float32)
    vol_4ch[0] = resized   # FLAIR — primary modality
    # ch 1 (T1) intentionally zero
    # ch 2 (T1CE) intentionally zero
    vol_4ch[3] = resized   # T2 — same T2-weighted contrast as FLAIR

    return torch.from_numpy(vol_4ch).float().unsqueeze(0)


def postprocess_segmentation_probs(pred_prob, confidence_threshold=0.72,
                                   smooth_sigma=1.4, min_component_voxels=400):
    """
    Standard medical-segmentation post-processing applied to raw softmax
    probability maps (4, D, H, W) → cleaned label volume (D, H, W, uint8).

    Steps
    -----
    1. Gaussian-smooth each class probability map (reduces salt-and-pepper noise).
    2. Re-normalise so probabilities sum to 1.
    3. argmax → class label per voxel.
    4. Confidence gate: voxels whose winning-class probability < threshold
       are forced to background (0).  Removes low-confidence fringe artefacts.
    5. Small-component removal per class: blobs smaller than
       `min_component_voxels` are erased to background.
    6. Keep only the 3 largest connected components of the merged tumour mask
       to eliminate scattered far-field noise when using single-channel input.
    """
    from scipy.ndimage import gaussian_filter, label as scipy_label

    # ── 1. Smooth ─────────────────────────────────────────────────────────────
    smoothed = np.zeros_like(pred_prob, dtype=np.float32)
    for c in range(pred_prob.shape[0]):
        smoothed[c] = gaussian_filter(pred_prob[c].astype(np.float32),
                                      sigma=smooth_sigma)

    # ── 2. Re-normalise ───────────────────────────────────────────────────────
    total = smoothed.sum(axis=0, keepdims=True)
    total[total == 0] = 1.0
    smoothed /= total

    # ── 3. Argmax ─────────────────────────────────────────────────────────────
    pred_label = np.argmax(smoothed, axis=0).astype(np.uint8)

    # ── 4. Confidence gate ────────────────────────────────────────────────────
    max_prob = smoothed.max(axis=0)
    pred_label[max_prob < confidence_threshold] = 0

    # ── 5. Small-component removal per class ──────────────────────────────────
    for cls_id in (1, 2, 3):
        cls_mask = (pred_label == cls_id).astype(np.uint8)
        if not cls_mask.any():
            continue
        labeled, n_comp = scipy_label(cls_mask)
        for cid in range(1, n_comp + 1):
            if (labeled == cid).sum() < min_component_voxels:
                pred_label[labeled == cid] = 0

    # ── 6. Keep only the largest connected components of the whole tumour ─────
    # This removes scattered far-field blobs (common with single-channel input)
    tumor_binary = (pred_label > 0).astype(np.uint8)
    if tumor_binary.any():
        labeled_all, n_all = scipy_label(tumor_binary)
        if n_all > 3:
            # Sort components by size (descending), keep top-3
            comp_sizes = [(cid, int((labeled_all == cid).sum())) for cid in range(1, n_all + 1)]
            comp_sizes.sort(key=lambda x: -x[1])
            keep_ids = {cid for cid, _ in comp_sizes[:3]}
            remove_mask = tumor_binary.astype(bool) & ~np.isin(labeled_all, list(keep_ids))
            pred_label[remove_mask] = 0

    # ── Debug: log label distribution ─────────────────────────────────────────
    n_total = pred_label.size
    for cls_id, name in ((0, "BG"), (1, "NCR"), (2, "ED"), (3, "ET")):
        n = int((pred_label == cls_id).sum())
        print(f"[3D Post] {name}: {n:,} voxels  ({100*n/n_total:.1f}%)")

    return pred_label


def intensity_guided_label_correction(pred_label, volume, smooth_sigma=1.5):
    """
    Refine 3D segmentation labels using FLAIR/T2-weighted MRI intensity priors.

    BraTS FLAIR intensity priors
    ─────────────────────────────
      ED  (edema)          → BRIGHTEST signal in tumor region  (class 2)
      ET  (enhancing)      → intermediate-bright               (class 3)
      NCR (necrotic core)  → DARKEST signal (fluid, cyst-like) (class 1)

    The model's predicted *tumour boundary* is preserved; only the intra-tumour
    class labels are reassigned based on intensity percentiles.  This removes
    the "NCR giant ball" artefact caused by feeding a single channel to a
    4-modality model.
    """
    from scipy.ndimage import gaussian_filter

    tumor_mask = (pred_label > 0)
    n_tumor = int(tumor_mask.sum())
    if n_tumor < 500:
        return pred_label          # too small to correct meaningfully

    vol_sm = gaussian_filter(volume.astype(np.float32), sigma=smooth_sigma)
    intensities = vol_sm[tumor_mask]

    # Three equal-sized intensity bands within tumor
    p33 = float(np.percentile(intensities, 33))
    p67 = float(np.percentile(intensities, 67))

    corrected = pred_label.copy()
    corrected[tumor_mask & (vol_sm <= p33)]                          = 1  # NCR
    corrected[tumor_mask & (vol_sm > p33) & (vol_sm <= p67)]        = 3  # ET
    corrected[tumor_mask & (vol_sm > p67)]                           = 2  # ED

    # Debug
    n_total = corrected.size
    for cid, name in ((0,"BG"),(1,"NCR"),(2,"ED"),(3,"ET")):
        n = int((corrected == cid).sum())
        print(f"[3D IntCorr] {name}: {n:,}  ({100*n/n_total:.1f}%)")

    return corrected


def run_3d_segmentation(model, volume):
    from scipy.ndimage import zoom
    D, H, W  = volume.shape
    patch    = (128, 128, 128)
    tensor   = prepare_volume_for_3d_model(volume, patch).to(device)

    with torch.no_grad():
        out = model(tensor)
        if isinstance(out, (list, tuple)):
            out = out[0]
        pred = torch.softmax(out, dim=1)[0].cpu().numpy()   # (4,128,128,128)

    # Step 1 – confidence gate + smoothing at patch resolution
    pred_label_patch = postprocess_segmentation_probs(pred, confidence_threshold=0.60)

    # Step 2 – zoom to original volume size
    zoom_factors = (D/128, H/128, W/128)
    pred_resized = zoom(pred_label_patch.astype(float), zoom_factors,
                        order=0).astype(np.uint8)

    tumor_mask = (pred_resized > 0).astype(np.uint8)
    return pred_resized, tumor_mask


# ─── 3D inference (4-channel — nnUNet-faithful BraTS input) ─────────────────

def _brain_bbox(channels):
    """Return the bounding box of the union-of-non-zero brain mask."""
    mask = np.zeros_like(channels[0], dtype=bool)
    for ch in channels:
        mask |= (ch > 0)
    coords = np.argwhere(mask)
    if len(coords) == 0:
        D, H, W = channels[0].shape
        return 0, D, 0, H, 0, W
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0) + 1
    return int(zmin), int(zmax), int(ymin), int(ymax), int(xmin), int(xmax)


def nnunet_preprocess_multichannel(channels, patch_size=(128, 128, 128)):
    """
    Replicate nnUNet v2 preprocessing for BraTS-style MRI:
      1. Crop to the non-zero brain bounding box (removes black borders).
      2. Z-score normalise each modality independently using only brain voxels.
      3. Resize cropped volume to patch_size via trilinear zoom.
    Returns (tensor [1,4,D,H,W], bbox) where bbox lets us map predictions back.
    """
    from scipy.ndimage import zoom

    zmin, zmax, ymin, ymax, xmin, xmax = _brain_bbox(channels)

    cropped, normed = [], []
    for ch in channels:
        crop = ch[zmin:zmax, ymin:ymax, xmin:xmax]
        fg   = crop[crop > 0]
        mean = float(fg.mean()) if fg.size > 0 else 0.0
        std  = float(fg.std())  if fg.size > 0 and fg.std() > 0 else 1.0
        norm = (crop - mean) / std
        cropped.append(crop)
        normed.append(norm)

    cD, cH, cW = normed[0].shape
    pd, ph, pw  = patch_size
    resized = [zoom(ch, (pd/cD, ph/cH, pw/cW), order=1) for ch in normed]

    vol_4ch = np.stack(resized, axis=0)
    tensor  = torch.from_numpy(vol_4ch).float().unsqueeze(0)
    bbox    = (zmin, zmax, ymin, ymax, xmin, xmax,
               channels[0].shape[0], channels[0].shape[1], channels[0].shape[2])
    return tensor, bbox


def run_3d_segmentation_multichannel(model, channels):
    """
    Run 4-channel 3D glioma segmentation with nnUNet-faithful preprocessing.
    Returns (pred_label_fullsize, tumor_mask_fullsize) both shaped (D,H,W).
    """
    from scipy.ndimage import zoom

    tensor, bbox = nnunet_preprocess_multichannel(channels)
    tensor = tensor.to(device)

    with torch.no_grad():
        out = model(tensor)
        if isinstance(out, (list, tuple)):
            out = out[0]
        pred = torch.softmax(out, dim=1)[0].cpu().numpy()   # (4,128,128,128)

    # Step 1 – Post-process at patch resolution (lower threshold — proper 4-channel input)
    pred_label_patch = postprocess_segmentation_probs(
        pred, confidence_threshold=0.45, smooth_sigma=0.8, min_component_voxels=100
    )

    # Unpack bbox
    zmin, zmax, ymin, ymax, xmin, xmax, D, H, W = bbox
    cD = zmax - zmin
    cH = ymax - ymin
    cW = xmax - xmin

    # Step 2 – Zoom prediction back to full volume size
    pred_cropped = zoom(pred_label_patch.astype(float),
                        (cD/128, cH/128, cW/128),
                        order=0).astype(np.uint8)

    pred_full = np.zeros((D, H, W), dtype=np.uint8)
    pred_full[zmin:zmax, ymin:ymax, xmin:xmax] = pred_cropped

    tumor_mask = (pred_full > 0).astype(np.uint8)
    return pred_full, tumor_mask


# ─── Overlay rendering ───────────────────────────────────────────────────────

def build_3d_plotly_figure(pred_full, flair_vol=None):
    """
    Build a 3D Slicer-style interactive figure:
      • Translucent brain surface (from flair_vol) as outer shell
      • Merged tumour mask rendered as a single solid RED 3D mesh inside
      • Clean blue-grey background (like 3D Slicer)
      • Orientation labels: S / I / R / L / A / P
      • Pink bounding-box frame for spatial reference

    pred_full : (D, H, W) uint8   — class label (0=bg, 1=NCR, 2=ED, 3=ET)
    flair_vol : (D, H, W) float   — raw MRI intensity volume (for brain surface)
    Returns   : JSON string for Plotly.react(), or None on failure.
    """
    try:
        from skimage.measure import marching_cubes
        from scipy.ndimage import zoom, gaussian_filter
        from scipy.sparse import csr_matrix
        import plotly.graph_objects as go
        import plotly.io as pio

        D, H, W = pred_full.shape
        TARGET  = 64
        sz      = (TARGET / D, TARGET / H, TARGET / W)

        # ── keep only the largest connected component (avoids multiple pieces) ──
        from scipy.ndimage import label as _label
        tumor_bin_full = (pred_full > 0).astype(np.uint8)
        labeled, n_comps = _label(tumor_bin_full)
        if n_comps > 1:
            comp_sizes = [(labeled == i).sum() for i in range(1, n_comps + 1)]
            largest = int(np.argmax(comp_sizes)) + 1
            tumor_bin_full = (labeled == largest).astype(np.uint8)

        # ── downsample ────────────────────────────────────────────────────────
        tumor_bin = tumor_bin_full.astype(np.float32)
        small     = zoom(tumor_bin, sz, order=1)

        if small.sum() < 5:
            return None

        # ── Laplacian mesh smoothing ──────────────────────────────────────────
        def _smooth(verts, faces, n=6, lam=0.5):
            nv = len(verts)
            r  = np.concatenate([faces[:,0], faces[:,1], faces[:,2]])
            c  = np.concatenate([faces[:,1], faces[:,2], faces[:,0]])
            adj = csr_matrix((np.ones(len(r)), (r, c)), shape=(nv, nv))
            deg = np.array(adj.sum(axis=1)).flatten(); deg[deg==0]=1
            v   = verts.astype(np.float64)
            for _ in range(n):
                v = (1-lam)*v + lam*(adj.dot(v)/deg[:,None])
            return v.astype(np.float32)

        # ── morphological closing to bridge tiny gaps before marching cubes ──
        from scipy.ndimage import binary_closing
        small_closed = binary_closing(small > 0.5,
                                      structure=np.ones((3,3,3))).astype(np.float32)

        # ── marching cubes on smoothed tumour mask ────────────────────────────
        blurred = gaussian_filter(small_closed, sigma=1.8)
        verts, faces, _, _ = marching_cubes(blurred, level=0.5,
                                             step_size=1,
                                             allow_degenerate=False)
        verts = _smooth(verts, faces, n=8, lam=0.5)

        # ── keep only the largest connected mesh component ────────────────────
        # (marching cubes can still produce tiny floating islands)
        if len(faces) > 0:
            from scipy.sparse.csgraph import connected_components as _cc
            nv = len(verts)
            ri = np.concatenate([faces[:,0], faces[:,1], faces[:,2]])
            ci = np.concatenate([faces[:,1], faces[:,2], faces[:,0]])
            adj2 = csr_matrix((np.ones(len(ri)), (ri, ci)), shape=(nv, nv))
            n_cc, labels_cc = _cc(adj2, directed=False)
            if n_cc > 1:
                comp_counts = np.bincount(labels_cc)
                keep_label  = int(np.argmax(comp_counts))
                keep_verts  = np.where(labels_cc == keep_label)[0]
                old2new     = np.full(nv, -1, dtype=int)
                old2new[keep_verts] = np.arange(len(keep_verts))
                mask_f = (old2new[faces[:,0]] >= 0) & \
                         (old2new[faces[:,1]] >= 0) & \
                         (old2new[faces[:,2]] >= 0)
                faces = old2new[faces[mask_f]]
                verts = verts[keep_verts]

        # Tumour verts are in TARGET (64) voxel space — do NOT centre yet;
        # we need to preserve spatial position relative to the brain.
        xi, yi, zi = faces[:,0], faces[:,1], faces[:,2]

        # ── brain surface (translucent shell from flair_vol) ──────────────────
        brain_mesh = None
        bv_raw     = None          # brain verts in 64-space (uncentred)
        if flair_vol is not None:
            try:
                BRAIN_TARGET = 48
                bsz = (BRAIN_TARGET / D, BRAIN_TARGET / H, BRAIN_TARGET / W)
                flair_small = zoom(flair_vol.astype(np.float32), bsz, order=1)
                # normalise to 0-1
                fmin, fmax = flair_small.min(), flair_small.max()
                if fmax > fmin:
                    flair_norm = (flair_small - fmin) / (fmax - fmin)
                else:
                    flair_norm = flair_small
                # brain mask: voxels > 15% of max intensity
                brain_mask = (flair_norm > 0.15).astype(np.float32)
                brain_blurred = gaussian_filter(brain_mask, sigma=2.0)
                bv, bf, _, _ = marching_cubes(brain_blurred, level=0.5,
                                               step_size=2,
                                               allow_degenerate=False)
                bv = _smooth(bv, bf, n=5, lam=0.6)
                # scale brain verts to same coordinate space as tumour (64-space)
                scale = TARGET / BRAIN_TARGET
                bv = bv * scale
                bv_raw = bv          # keep uncentred copy
                bxi, byi, bzi = bf[:,0], bf[:,1], bf[:,2]
            except Exception as be:
                print(f"[3D Plot] Brain surface skipped: {be}")
                bv_raw = None

        # ── Use a single common reference so tumour keeps its real position ──
        # Prefer brain centroid; fall back to volume centre.
        if bv_raw is not None:
            common_center = bv_raw.mean(axis=0)
        else:
            common_center = np.array([TARGET / 2, TARGET / 2, TARGET / 2],
                                     dtype=np.float32)

        # Shift tumour verts by the same reference
        verts_c = verts - common_center
        xv, yv, zv = verts_c[:,0], verts_c[:,1], verts_c[:,2]

        # ── tumour mesh (solid red, like 3D Slicer) ───────────────────────────
        tumor_mesh = go.Mesh3d(
            x=xv.tolist(), y=yv.tolist(), z=zv.tolist(),
            i=xi.tolist(), j=yi.tolist(), k=zi.tolist(),
            name="Tumour",
            color="rgb(210, 40, 40)",
            opacity=1.0,
            lighting=dict(diffuse=0.7, specular=0.2, ambient=0.5,
                          roughness=0.9, fresnel=0.1),
            lightposition=dict(x=100, y=200, z=300),
            flatshading=True,
            showlegend=True,
        )

        if bv_raw is not None:
            try:
                bv_c = bv_raw - common_center
                bxv, byv, bzv = bv_c[:,0], bv_c[:,1], bv_c[:,2]
                brain_mesh = go.Mesh3d(
                    x=bxv.tolist(), y=byv.tolist(), z=bzv.tolist(),
                    i=bxi.tolist(), j=byi.tolist(), k=bzi.tolist(),
                    name="Brain",
                    color="rgb(200, 215, 245)",
                    opacity=0.40,
                    lighting=dict(diffuse=0.7, specular=0.15, ambient=0.6,
                                  roughness=0.9, fresnel=0.08),
                    lightposition=dict(x=100, y=200, z=300),
                    flatshading=True,
                    showlegend=True,
                )
            except Exception as be:
                print(f"[3D Plot] Brain surface skipped: {be}")
                brain_mesh = None

        # ── bounding-box frame (pink/magenta, like 3D Slicer) ─────────────────
        pad   = max(xv.max()-xv.min(), yv.max()-yv.min(), zv.max()-zv.min()) * 0.18
        x0,x1 = xv.min()-pad, xv.max()+pad
        y0,y1 = yv.min()-pad, yv.max()+pad
        z0,z1 = zv.min()-pad, zv.max()+pad

        corners = np.array([
            [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],[x0,y0,z0],  # bottom
            [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],[x0,y0,z1],  # top
        ])
        # verticals
        segs = [(0,5),(1,6),(2,7),(3,8)]
        bx,by,bz = [],[],[]
        for p1,p2 in segs:
            bx += [corners[p1,0], corners[p2,0], None]
            by += [corners[p1,1], corners[p2,1], None]
            bz += [corners[p1,2], corners[p2,2], None]
        bx += list(corners[:,0]) + [None]
        by += list(corners[:,1]) + [None]
        bz += list(corners[:,2]) + [None]

        box_frame = go.Scatter3d(
            x=bx, y=by, z=bz,
            mode="lines",
            line=dict(color="rgb(220,80,180)", width=2),
            name="",
            showlegend=False,
            hoverinfo="skip",
        )

        # ── orientation label annotations ─────────────────────────────────────
        cx = float((x0+x1)/2); cy = float((y0+y1)/2); cz = float((z0+z1)/2)
        reach = max(abs(x1-x0), abs(y1-y0), abs(z1-z0)) * 0.68
        lbl_style = dict(showarrow=False, font=dict(color="white", size=15,
                                                     family="Arial Black"),
                         bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)")
        annotations = [
            dict(x=cx,     y=cy,     z=z1+reach*0.6, text="<b>S</b>", **lbl_style),
            dict(x=cx,     y=cy,     z=z0-reach*0.6, text="<b>I</b>", **lbl_style),
            dict(x=x0-reach*0.6, y=cy, z=cz,         text="<b>R</b>", **lbl_style),
            dict(x=x1+reach*0.6, y=cy, z=cz,         text="<b>L</b>", **lbl_style),
            dict(x=cx,     y=y0-reach*0.6, z=cz,     text="<b>A</b>", **lbl_style),
            dict(x=cx,     y=y1+reach*0.6, z=cz,     text="<b>P</b>", **lbl_style),
        ]

        traces = []
        if brain_mesh is not None:
            traces.append(brain_mesh)
        traces.append(tumor_mesh)
        traces.append(box_frame)
        fig = go.Figure(data=traces)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            scene=dict(
                bgcolor="rgb(100,120,160)",   # 3D-Slicer blue-grey
                xaxis=dict(visible=False, showgrid=False),
                yaxis=dict(visible=False, showgrid=False),
                zaxis=dict(visible=False, showgrid=False),
                aspectmode="cube",
                camera=dict(eye=dict(x=1.5, y=1.3, z=1.1),
                            up=dict(x=0, y=0, z=1)),
                annotations=annotations,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        return pio.to_json(fig)

    except Exception as e:
        print(f"[3D Plot] Could not build Plotly figure: {e}")
        import traceback; traceback.print_exc()
        return None


def slice_to_gray_uint8(slice_2d):
    s = slice_2d.copy().astype(float)
    s_min, s_max = s.min(), s.max()
    if s_max > s_min:
        s = (s - s_min) / (s_max - s_min) * 255.0
    return np.clip(s, 0, 255).astype(np.uint8)


def save_colored_overlay(slice_2d, pred_slice, out_path, alpha=0.42):
    """
    Render the original grayscale slice with per-class coloured mask blended on top.
    pred_slice contains class IDs 0-3; class 0 is transparent (background).
    Improvements:
      - Lower alpha so brain anatomy stays visible beneath colours
      - CLAHE contrast enhancement on the background slice
      - White outer + class-colour inner contour outlines for each region
      - Small burned-in legend at bottom-right corner
    """
    # ── Background: CLAHE-enhanced grayscale ──────────────────────────────────
    gray_raw = slice_to_gray_uint8(slice_2d)
    clahe    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray     = clahe.apply(gray_raw)
    rgb      = np.stack([gray, gray, gray], axis=-1).astype(float)
    result   = rgb.copy()

    # ── Colour blending per class ──────────────────────────────────────────────
    for cls_id, (r, g, b) in CLASS_COLORS_RGB.items():
        mask = (pred_slice == cls_id)
        if mask.any():
            result[mask, 0] = (1 - alpha) * rgb[mask, 0] + alpha * r
            result[mask, 1] = (1 - alpha) * rgb[mask, 1] + alpha * g
            result[mask, 2] = (1 - alpha) * rgb[mask, 2] + alpha * b

    result = np.clip(result, 0, 255).astype(np.uint8)

    # ── Class boundary outlines ────────────────────────────────────────────────
    for cls_id, (r, g, b) in CLASS_COLORS_RGB.items():
        binary = (pred_slice == cls_id).astype(np.uint8)
        if binary.any():
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            # white outer outline (1 px)
            cv2.drawContours(result, contours, -1, (255, 255, 255), 2)
            # class-colour inner outline (1 px) — sharpens region edge
            cv2.drawContours(result, contours, -1, (r, g, b), 1)

    # ── Burned-in legend (bottom-right corner) ────────────────────────────────
    CLASS_NAMES = {1: "NCR", 2: "Edema", 3: "ET"}
    classes_present = [cid for cid in CLASS_COLORS_RGB if (pred_slice == cid).any()]
    if classes_present:
        H, W = result.shape[:2]
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.28, min(0.38, W / 700))
        thickness  = 1
        pad        = 5
        box_h      = 12
        item_h     = box_h + pad + 2
        legend_h   = item_h * len(classes_present) + pad
        legend_w   = 80
        lx = W - legend_w - 6
        ly = H - legend_h - 6
        overlay_leg = result.copy()
        cv2.rectangle(overlay_leg, (lx - 2, ly - 2),
                      (W - 4, H - 4), (20, 20, 20), -1)
        result = cv2.addWeighted(overlay_leg, 0.55, result, 0.45, 0)
        for i, cls_id in enumerate(classes_present):
            r, g, b = CLASS_COLORS_RGB[cls_id]
            by = ly + i * item_h + pad
            cv2.rectangle(result, (lx, by), (lx + box_h, by + box_h),
                          (r, g, b), -1)          # result is RGB-ordered
            cv2.rectangle(result, (lx, by), (lx + box_h, by + box_h),
                          (255, 255, 255), 1)
            cv2.putText(result, CLASS_NAMES[cls_id],
                        (lx + box_h + 4, by + box_h - 2),
                        font, font_scale, (240, 240, 240), thickness,
                        cv2.LINE_AA)

    cv2.imwrite(out_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))


def save_slice_as_image(slice_2d, out_path):
    cv2.imwrite(out_path, slice_to_gray_uint8(slice_2d))


# ─── Save results ─────────────────────────────────────────────────────────────

def save_3d_results(volume_flair, pred_label, scan_id, n_representative=None,
                    out_dir="static/uploads"):
    """
    Save coloured overlay images (original FLAIR slice + class-coloured mask).

    Args:
        volume_flair    : (D,H,W) numpy array  — FLAIR channel used as background.
        pred_label      : (D,H,W) uint8        — per-voxel class label (0-3).
        scan_id         : str                  — used to name output files.
        n_representative: maximum slices to save (None = all tumor slices).

    Returns:
        list of {"overlay": url, "slice_idx": int}
    """
    overlays_dir = os.path.join(out_dir, "overlays_3d")
    os.makedirs(overlays_dir, exist_ok=True)

    D     = volume_flair.shape[0]
    start = max(0, int(D * 0.10))
    end   = min(D, int(D * 0.90))

    tumor_slices = [i for i in range(start, end) if (pred_label[i] > 0).any()]

    if tumor_slices:
        if n_representative is not None:
            step     = max(1, len(tumor_slices) // n_representative)
            selected = tumor_slices[::step][:n_representative]
            while len(selected) < n_representative:
                selected.append(tumor_slices[-1])
            indices = sorted(set(selected))
        else:
            indices = sorted(tumor_slices)
    else:
        n_rep   = n_representative if n_representative else 10
        indices = list(np.linspace(start, end - 1, n_rep, dtype=int))

    slices_info = []
    for i, idx in enumerate(indices):
        idx          = int(idx)
        overlay_path = os.path.join(overlays_dir, f"overlay3d_{scan_id}_{i}.png")
        save_colored_overlay(volume_flair[idx], pred_label[idx], overlay_path)
        slices_info.append({
            "overlay":   "/" + overlay_path.replace("\\", "/"),
            "slice_idx": idx
        })

    return slices_info


# ─── Helpers for 2D→3D fallback ──────────────────────────────────────────────

def get_best_slice_for_2d_seg(volume, tumor_mask):
    D     = volume.shape[0]
    start = max(0, D // 4)
    end   = min(D, 3 * D // 4)
    n     = min(30, max(1, end - start))
    candidates = np.linspace(start, end - 1, n, dtype=int)

    best_idx  = int(candidates[len(candidates) // 2])
    best_area = 0
    for idx in candidates:
        area = int(tumor_mask[int(idx)].sum())
        if area > best_area:
            best_area = area
            best_idx  = int(idx)
    return best_idx


def extract_slice_as_image_file(volume, slice_idx, out_path):
    save_slice_as_image(volume[slice_idx], out_path)
    return out_path


def load_image_as_pseudo_3d(image_path, depth=64):
    from PIL import Image as PILImage
    img = PILImage.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float32)
    return np.stack([arr] * depth, axis=0)


def render_mpr_slice_b64(flair_vol, pred_label, plane, idx, alpha=0.35):
    """
    Render a single MPR slice (axial / coronal / sagittal).
    Brain MRI shown in greyscale; all tumour labels merged into a single
    solid RED overlay — matching the 3D Slicer look.
    Returns a data-URI  "data:image/png;base64,..."  for <img src>.
    """
    import base64

    D, H, W = flair_vol.shape

    if plane == "axial":
        sl = flair_vol[idx, :, :]
        pr = pred_label[idx, :, :]
    elif plane == "coronal":
        sl = np.ascontiguousarray(flair_vol[:, idx, :][::-1])
        pr = np.ascontiguousarray(pred_label[:, idx, :][::-1])
    else:                                               # sagittal
        sl = np.ascontiguousarray(flair_vol[:, :, idx][::-1, ::-1])
        pr = np.ascontiguousarray(pred_label[:, :, idx][::-1, ::-1])

    # ── Greyscale + CLAHE (boost brain contrast) ───────────────────────────
    gray_raw = slice_to_gray_uint8(sl)
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray     = clahe.apply(gray_raw)
    rgb      = np.stack([gray, gray, gray], axis=-1).astype(float)
    result   = rgb.copy()

    # ── Binary tumour mask (all classes → 1) ───────────────────────────────
    raw_mask = (pr > 0).astype(np.uint8)

    # ── Intensity-guided 2-D cleanup ───────────────────────────────────────
    # FLAIR hyperintensity is the ground truth for tumor location.
    # Only keep model predictions that overlap with genuinely bright voxels,
    # then keep only the single largest connected component.
    tumor_mask_clean = np.zeros_like(raw_mask)
    if raw_mask.any():
        # 1. Build a brain mask (non-black background pixels)
        brain_px = gray_raw[gray_raw > 5]
        if brain_px.size > 0:
            # Bright-region gate: top 30 % of brain voxels
            bright_thresh = float(np.percentile(brain_px, 70))
            bright_gate   = (gray_raw >= bright_thresh).astype(np.uint8)

            # Dilate the bright gate slightly so we don't cut edges
            dil_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            bright_gate = cv2.dilate(bright_gate, dil_k, iterations=1)

            # AND: model prediction ∩ bright region
            gated = cv2.bitwise_and(raw_mask, bright_gate)
        else:
            gated = raw_mask

        # 2. Morphological closing to fill small holes
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed  = cv2.morphologyEx(gated, cv2.MORPH_CLOSE, close_k)

        # 3. Keep only the single largest connected component
        n_lbls, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        if n_lbls > 1:                             # 0 is background
            # stats[:,4] = area; skip component 0 (bg)
            largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
            tumor_mask_clean[labels == largest] = 1

    tumor_mask = tumor_mask_clean.astype(bool)

    # ── Solid red blend over tumour voxels ─────────────────────────────────
    TUMOR_R, TUMOR_G, TUMOR_B = 220, 50, 50
    if tumor_mask.any():
        result[tumor_mask, 0] = (1 - alpha) * rgb[tumor_mask, 0] + alpha * TUMOR_R
        result[tumor_mask, 1] = (1 - alpha) * rgb[tumor_mask, 1] + alpha * TUMOR_G
        result[tumor_mask, 2] = (1 - alpha) * rgb[tumor_mask, 2] + alpha * TUMOR_B

    result = np.clip(result, 0, 255).astype(np.uint8)

    # ── Single contour outline around the whole tumour ─────────────────────
    if tumor_mask.any():
        binary   = tumor_mask.astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (255, 210, 210), 2)  # soft pink ring
        cv2.drawContours(result, contours, -1, (220,  50,  50), 1)  # red inner line

    # ── Encode PNG → base64 ─────────────────────────────────────────────────
    bgr  = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"
