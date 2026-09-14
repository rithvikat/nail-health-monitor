"""
Nail Region Detector and Cropper Module.

Prepares uploaded nail images before disease-classification inference:
- Accepts both single-nail close-up images and whole-hand images.
- Validates input image resolution (width >= 300, height >= 300).
- Locates visible nail regions on whole-hand images and extracts high-quality individual crops.
- Validates crop quality (dimensions, blur, background ratio).
- Preserves untouched single-nail close-ups for direct inference.
"""

import os
import uuid
import cv2
import numpy as np


CROPS_DIR = os.path.join("uploads", "crops")
os.makedirs(CROPS_DIR, exist_ok=True)


def check_crop_quality(crop_bgr: np.ndarray) -> bool:
    """
    Verifies that a detected nail crop is usable for AI classification:
    - Minimum dimension >= 80x80
    - Contains sufficient skin/tissue (not mostly background)
    - Not severely blurred
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return False

    ch, cw = crop_bgr.shape[:2]
    if cw < 80 or ch < 80:
        return False

    # 1. Skin/tissue presence check in YCrCb
    ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCR_CB)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    skin_mask = (cr >= 125) & (cr <= 185) & (cb >= 65) & (cb <= 145)
    skin_ratio = float(np.mean(skin_mask))
    if skin_ratio < 0.15:
        return False

    # 2. Blur / sharpness check
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if laplacian_var < 6.0:
        return False

    return True


def detect_and_crop_nails(image_path_or_array) -> dict:
    """
    Analyzes an input photo, detects whether it is a close-up single nail
    or a whole-hand multi-nail image, and crops all visible nail regions.

    Returns:
        {
            "status": "close_up" | "multi_nail" | "none",
            "message": str,
            "original_dimensions": (width, height),
            "crops": [
                {
                    "crop_image": np.ndarray (BGR),
                    "crop_path": str (relative path in uploads/crops/),
                    "bbox": (x, y, w, h),
                    "dimensions": (w, h)
                }, ...
            ]
        }
    """
    # 1. Load image
    if isinstance(image_path_or_array, str):
        image_bgr = cv2.imread(image_path_or_array)
    elif isinstance(image_path_or_array, np.ndarray):
        image_bgr = image_path_or_array.copy()
    else:
        return {
            "status": "none",
            "message": "Invalid image input. Could not read image file.",
            "original_dimensions": (0, 0),
            "crops": []
        }

    if image_bgr is None or image_bgr.size == 0:
        return {
            "status": "none",
            "message": "Could not decode image. Please ensure the file is an uncorrupted image.",
            "original_dimensions": (0, 0),
            "crops": []
        }

    height, width = image_bgr.shape[:2]
    total_area = height * width

    # 2. Strict resolution check (width >= 300 and height >= 300)
    if width < 300 or height < 300:
        return {
            "status": "none",
            "message": f"Image resolution is too low ({width}x{height}px). Both width and height must be at least 300 pixels.",
            "original_dimensions": (width, height),
            "crops": []
        }

    # 3. Detect skin/tissue regions using YCrCb & HSV across ethnicities
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCR_CB)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    skin_ycrcb = (cr >= 128) & (cr <= 180) & (cb >= 70) & (cb <= 140)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    skin_hsv = ((H <= 25) | (H >= 165)) & (S >= 15) & (S <= 210) & (V >= 35)

    skin_mask = (skin_ycrcb & skin_hsv).astype(np.uint8) * 255
    skin_ratio = float(np.mean(skin_mask > 0))

    if skin_ratio < 0.04:
        return {
            "status": "none",
            "message": "No suitable nail region could be detected in this image. Please upload a clearer image with the nail(s) fully visible.",
            "original_dimensions": (width, height),
            "crops": []
        }

    # Clean mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_clean = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_clean = cv2.morphologyEx(skin_clean, cv2.MORPH_OPEN, kernel)

    # 4. Find external skin contours
    contours, _ = cv2.findContours(skin_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "status": "none",
            "message": "No suitable nail region could be detected in this image. Please upload a clearer image with the nail(s) fully visible.",
            "original_dimensions": (width, height),
            "crops": []
        }

    # Filter significant skin contours (at least 3% of total area)
    hand_contours = [c for c in contours if cv2.contourArea(c) > 0.03 * total_area]
    if not hand_contours:
        return {
            "status": "none",
            "message": "No suitable nail region could be detected in this image. Please upload a clearer image with the nail(s) fully visible.",
            "original_dimensions": (width, height),
            "crops": []
        }

    # 5. Analyze fingertips across significant hand contours
    raw_tips = []
    min_dim = min(width, height)
    for c in hand_contours:
        hull = cv2.convexHull(c, returnPoints=False)
        if hull is not None and len(hull) > 3:
            defects = cv2.convexityDefects(c, hull)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    depth_px = d / 256.0
                    # A true valley between fingers in a hand has significant depth (> 18% of min dimension)
                    if depth_px > 0.18 * min_dim:
                        start = tuple(c[s][0])
                        end = tuple(c[e][0])
                        far = tuple(c[f][0])
                        # Angle check: finger valleys are acute/V-shaped (< 105 degrees)
                        v1 = np.array(start) - np.array(far)
                        v2 = np.array(end) - np.array(far)
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                        if angle_deg < 105:
                            raw_tips.append(start)
                            raw_tips.append(end)

    # Cluster/deduplicate fingertips close to each other
    unique_tips = []
    min_dist = max(width, height) * 0.055
    for pt in raw_tips:
        if not any(np.hypot(pt[0] - u[0], pt[1] - u[1]) < min_dist for u in unique_tips):
            unique_tips.append(pt)

    # ========================================================
    # MODE 1: SINGLE-NAIL CLOSE-UP MODE
    # ========================================================
    # In a whole hand image, multiple fingers (>= 3) create distinct valleys.
    # If fewer than 3 fingertips are detected, the image is a single-nail close-up.
    if len(unique_tips) < 3:
        crop_filename = f"crop_single_{uuid.uuid4().hex[:10]}.jpg"
        crop_path = os.path.join(CROPS_DIR, crop_filename)
        cv2.imwrite(crop_path, image_bgr)
        rel_crop_path = os.path.join("uploads", "crops", crop_filename).replace("\\", "/")

        print("\n--- NAIL DETECTION DEBUG ---")
        print(f"Original dimensions: {width}x{height}")
        print("Mode: Single-nail close-up")
        print("Number of detected nails: 1")
        print(f"Bounding box: (0, 0, {width}, {height})")
        print(f"Crop dimensions: {width}x{height}\n")

        return {
            "status": "close_up",
            "message": "Single nail region detected and analyzed.",
            "original_dimensions": (width, height),
            "crops": [
                {
                    "crop_image": image_bgr,
                    "crop_path": rel_crop_path,
                    "bbox": (0, 0, width, height),
                    "dimensions": (width, height)
                }
            ]
        }

    # ========================================================
    # MODE 2: WHOLE-HAND MULTI-NAIL MODE
    # ========================================================
    # Multiple fingertips were detected. Extract individual nail crops.
    valid_crops = []
    box_size = int(max(width, height) * 0.22)  # Proportional nail crop window

    # Sort tips spatially (left-to-right) for consistent nail numbering
    unique_tips.sort(key=lambda pt: pt[0])

    for idx, (tx, ty) in enumerate(unique_tips):
        x1 = max(0, int(tx - box_size // 2))
        y1 = max(0, int(ty - box_size // 2))
        x2 = min(width, x1 + box_size)
        y2 = min(height, y1 + box_size)

        # Keep box square if clipped at borders
        if (x2 - x1) < box_size and x1 > 0:
            x1 = max(0, x2 - box_size)
        if (y2 - y1) < box_size and y1 > 0:
            y1 = max(0, y2 - box_size)

        crop = image_bgr[y1:y2, x1:x2]

        if check_crop_quality(crop):
            crop_filename = f"crop_nail_{idx+1}_{uuid.uuid4().hex[:8]}.jpg"
            crop_path = os.path.join(CROPS_DIR, crop_filename)
            cv2.imwrite(crop_path, crop)
            rel_crop_path = os.path.join("uploads", "crops", crop_filename).replace("\\", "/")
            cw, ch = crop.shape[1], crop.shape[0]

            valid_crops.append({
                "crop_image": crop,
                "crop_path": rel_crop_path,
                "bbox": (x1, y1, cw, ch),
                "dimensions": (cw, ch)
            })

    # Fallback if crops failed quality checks
    if not valid_crops:
        crop_filename = f"crop_fallback_{uuid.uuid4().hex[:10]}.jpg"
        crop_path = os.path.join(CROPS_DIR, crop_filename)
        cv2.imwrite(crop_path, image_bgr)
        rel_crop_path = os.path.join("uploads", "crops", crop_filename).replace("\\", "/")

        return {
            "status": "close_up",
            "message": "Single nail region analyzed.",
            "original_dimensions": (width, height),
            "crops": [
                {
                    "crop_image": image_bgr,
                    "crop_path": rel_crop_path,
                    "bbox": (0, 0, width, height),
                    "dimensions": (width, height)
                }
            ]
        }

    # Debug logging as required
    print("\n--- NAIL DETECTION DEBUG ---")
    print(f"Original dimensions: {width}x{height}")
    print("Mode: Whole-hand multi-nail")
    print(f"Number of detected nails: {len(valid_crops)}")
    for i, c in enumerate(valid_crops):
        print(f"  Nail {i+1}: Bounding Box={c['bbox']}, Crop Dimensions={c['dimensions']}")
    print("----------------------------\n")

    return {
        "status": "multi_nail",
        "message": f"Whole-hand image analyzed: {len(valid_crops)} nail regions detected and analyzed.",
        "original_dimensions": (width, height),
        "crops": valid_crops
    }
