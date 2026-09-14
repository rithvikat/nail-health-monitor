"""
Nail Region Detector and Cropper Module.

Prepares uploaded nail images before disease-classification inference:
- Accepts both single-nail close-up images and whole-hand images.
- Validates input image resolution.
- Locates visible nail regions on whole-hand images.
- Extracts individual nail crops.
- Validates crop quality.
- Preserves single-nail close-ups for direct inference.
"""

import os
import uuid

import cv2
import numpy as np


# Store generated nail crops
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CROPS_DIR = os.path.join(BASE_DIR, "uploads", "crops")

os.makedirs(CROPS_DIR, exist_ok=True)


def check_crop_quality(crop_bgr: np.ndarray) -> bool:
    """
    Check whether a detected nail crop is suitable for classification.

    Conditions:
    - Minimum size: 80x80
    - Contains sufficient skin/tissue
    - Not severely blurred
    """

    if crop_bgr is None or crop_bgr.size == 0:
        return False

    ch, cw = crop_bgr.shape[:2]

    # Minimum crop size
    if cw < 80 or ch < 80:
        return False

    # ---------------------------------------------------------
    # 1. Skin/tissue presence check using YCrCb
    # ---------------------------------------------------------
    ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCR_CB)

    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]

    skin_mask = (
        (cr >= 125)
        & (cr <= 185)
        & (cb >= 65)
        & (cb <= 145)
    )

    skin_ratio = float(np.mean(skin_mask))

    if skin_ratio < 0.15:
        return False

    # ---------------------------------------------------------
    # 2. Blur / sharpness check
    # ---------------------------------------------------------
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    laplacian_var = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )

    if laplacian_var < 6.0:
        return False

    return True


def _save_crop(crop, prefix):
    """
    Save a crop and return its relative path.
    """

    filename = f"{prefix}_{uuid.uuid4().hex[:10]}.jpg"

    crop_path = os.path.join(
        CROPS_DIR,
        filename
    )

    cv2.imwrite(crop_path, crop)

    relative_path = os.path.join(
        "uploads",
        "crops",
        filename
    ).replace("\\", "/")

    return crop_path, relative_path


def detect_and_crop_nails(image_path_or_array) -> dict:
    """
    Analyze an uploaded image.

    Supports:
    - Single-nail close-up images
    - Whole-hand images

    Returns:
        {
            "status": "close_up" | "multi_nail" | "none",
            "message": str,
            "original_dimensions": (width, height),
            "crops": [...]
        }
    """

    # =========================================================
    # 1. LOAD IMAGE
    # =========================================================

    if isinstance(image_path_or_array, str):

        image_bgr = cv2.imread(image_path_or_array)

    elif isinstance(image_path_or_array, np.ndarray):

        image_bgr = image_path_or_array.copy()

    else:

        return {
            "status": "none",
            "message": (
                "Invalid image input. "
                "Could not read image file."
            ),
            "original_dimensions": (0, 0),
            "crops": []
        }

    # Check image decoding
    if image_bgr is None or image_bgr.size == 0:

        return {
            "status": "none",
            "message": (
                "Could not decode image. "
                "Please ensure the file is an uncorrupted image."
            ),
            "original_dimensions": (0, 0),
            "crops": []
        }

    # =========================================================
    # 2. IMAGE DIMENSIONS
    # =========================================================

    height, width = image_bgr.shape[:2]

    total_area = height * width

    # Minimum resolution
    if width < 300 or height < 300:

        return {
            "status": "none",
            "message": (
                f"Image resolution is too low "
                f"({width}x{height}px). "
                "Both width and height must be at least "
                "300 pixels."
            ),
            "original_dimensions": (width, height),
            "crops": []
        }

    # =========================================================
    # 3. SKIN / TISSUE DETECTION
    # =========================================================

    # YCrCb
    ycrcb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2YCR_CB
    )

    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]

    skin_ycrcb = (
        (cr >= 128)
        & (cr <= 180)
        & (cb >= 70)
        & (cb <= 140)
    )

    # HSV
    hsv = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2HSV
    )

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    skin_hsv = (
        ((H <= 25) | (H >= 165))
        & (S >= 15)
        & (S <= 210)
        & (V >= 35)
    )

    # Combine masks
    skin_mask = (
        skin_ycrcb & skin_hsv
    ).astype(np.uint8) * 255

    skin_ratio = float(
        np.mean(skin_mask > 0)
    )

    # No sufficient skin/tissue
    if skin_ratio < 0.04:

        return {
            "status": "none",
            "message": (
                "No suitable nail region could be detected "
                "in this image. Please upload a clearer image "
                "with the nail(s) fully visible."
            ),
            "original_dimensions": (width, height),
            "crops": []
        }

    # =========================================================
    # 4. CLEAN SKIN MASK
    # =========================================================

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    skin_clean = cv2.morphologyEx(
        skin_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    skin_clean = cv2.morphologyEx(
        skin_clean,
        cv2.MORPH_OPEN,
        kernel
    )

    # =========================================================
    # 5. FIND HAND CONTOURS
    # =========================================================

    contours, _ = cv2.findContours(
        skin_clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:

        return {
            "status": "none",
            "message": (
                "No suitable nail region could be detected "
                "in this image. Please upload a clearer image "
                "with the nail(s) fully visible."
            ),
            "original_dimensions": (width, height),
            "crops": []
        }

    # Keep significant contours
    hand_contours = [
        c
        for c in contours
        if cv2.contourArea(c) > 0.03 * total_area
    ]

    if not hand_contours:

        return {
            "status": "none",
            "message": (
                "No suitable nail region could be detected "
                "in this image. Please upload a clearer image "
                "with the nail(s) fully visible."
            ),
            "original_dimensions": (width, height),
            "crops": []
        }

    # =========================================================
    # 6. DETECT FINGERTIPS
    # =========================================================

    raw_tips = []

    for contour in hand_contours:

        hull = cv2.convexHull(
            contour,
            returnPoints=False
        )

        if hull is None:
            continue

        if len(hull) <= 3:
            continue

        defects = cv2.convexityDefects(
            contour,
            hull
        )

        if defects is None:
            continue

        # -----------------------------------------------------
        # Handle both OpenCV array formats:
        #
        # (N, 1, 4)
        # (N, 4)
        # -----------------------------------------------------

        for i in range(defects.shape[0]):

            try:

                if defects.ndim == 3:

                    s, e, f, d = defects[i][0]

                elif defects.ndim == 2:

                    s, e, f, d = defects[i]

                else:

                    continue

            except (ValueError, TypeError, IndexError):

                continue

            # Make sure indexes are valid
            if (
                s < 0
                or e < 0
                or f < 0
                or s >= len(contour)
                or e >= len(contour)
                or f >= len(contour)
            ):
                continue

            # Get points from CURRENT contour
            start = tuple(
                contour[int(s)][0]
            )

            end = tuple(
                contour[int(e)][0]
            )

            far = tuple(
                contour[int(f)][0]
            )

            # -------------------------------------------------
            # Angle calculation
            # -------------------------------------------------

            v1 = (
                np.array(start, dtype=float)
                - np.array(far, dtype=float)
            )

            v2 = (
                np.array(end, dtype=float)
                - np.array(far, dtype=float)
            )

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 < 1e-6 or norm2 < 1e-6:
                continue

            cos_angle = (
                np.dot(v1, v2)
                / (norm1 * norm2)
            )

            cos_angle = np.clip(
                cos_angle,
                -1.0,
                1.0
            )

            angle_deg = np.degrees(
                np.arccos(cos_angle)
            )

            # Finger valleys are generally V-shaped
            if angle_deg < 105:

                raw_tips.append(start)
                raw_tips.append(end)

    # =========================================================
    # 7. REMOVE DUPLICATE FINGERTIPS
    # =========================================================

    unique_tips = []

    min_dist = (
        max(width, height) * 0.055
    )

    for pt in raw_tips:

        too_close = any(
            np.hypot(
                pt[0] - u[0],
                pt[1] - u[1]
            ) < min_dist
            for u in unique_tips
        )

        if not too_close:
            unique_tips.append(pt)

    # =========================================================
    # DEBUG
    # =========================================================

    print("\n--- NAIL DETECTION DEBUG ---")
    print(
        f"Original dimensions: {width}x{height}"
    )
    print(
        f"Raw fingertip points: {len(raw_tips)}"
    )
    print(
        f"Unique fingertip points: {len(unique_tips)}"
    )

    # =========================================================
    # MODE 1: SINGLE NAIL CLOSE-UP
    # =========================================================

    if len(unique_tips) < 3:

        crop_path, rel_crop_path = _save_crop(
            image_bgr,
            "crop_single"
        )

        print(
            "Mode: Single-nail close-up"
        )

        print(
            "Number of detected nails: 1"
        )

        print(
            f"Bounding box: "
            f"(0, 0, {width}, {height})"
        )

        print(
            f"Crop dimensions: "
            f"{width}x{height}"
        )

        print("----------------------------\n")

        return {
            "status": "close_up",
            "message": (
                "Single nail region detected and analyzed."
            ),
            "original_dimensions": (
                width,
                height
            ),
            "crops": [
                {
                    "crop_image": image_bgr,
                    "crop_path": rel_crop_path,
                    "bbox": (
                        0,
                        0,
                        width,
                        height
                    ),
                    "dimensions": (
                        width,
                        height
                    )
                }
            ]
        }

    # =========================================================
    # MODE 2: WHOLE HAND
    # =========================================================

    valid_crops = []

    # Crop window size
    box_size = int(
        max(width, height) * 0.22
    )

    # Sort left-to-right
    unique_tips.sort(
        key=lambda pt: pt[0]
    )

    for idx, (tx, ty) in enumerate(
        unique_tips
    ):

        # Initial bounding box
        x1 = max(
            0,
            int(tx - box_size // 2)
        )

        y1 = max(
            0,
            int(ty - box_size // 2)
        )

        x2 = min(
            width,
            x1 + box_size
        )

        y2 = min(
            height,
            y1 + box_size
        )

        # -----------------------------------------------------
        # Keep crop approximately square
        # -----------------------------------------------------

        if (
            x2 - x1 < box_size
            and x1 > 0
        ):

            x1 = max(
                0,
                x2 - box_size
            )

        if (
            y2 - y1 < box_size
            and y1 > 0
        ):

            y1 = max(
                0,
                y2 - box_size
            )

        # Extract crop
        crop = image_bgr[
            y1:y2,
            x1:x2
        ]

        # Check crop quality
        if not check_crop_quality(crop):
            continue

        # Save crop
        crop_path, rel_crop_path = _save_crop(
            crop,
            f"crop_nail_{idx + 1}"
        )

        cw = crop.shape[1]
        ch = crop.shape[0]

        valid_crops.append(
            {
                "crop_image": crop,
                "crop_path": rel_crop_path,
                "bbox": (
                    x1,
                    y1,
                    cw,
                    ch
                ),
                "dimensions": (
                    cw,
                    ch
                )
            }
        )

    # =========================================================
    # FALLBACK
    # =========================================================

    if not valid_crops:

        crop_path, rel_crop_path = _save_crop(
            image_bgr,
            "crop_fallback"
        )

        print(
            "Mode: Fallback single-nail mode"
        )

        print(
            "No individual crops passed quality checks."
        )

        print("----------------------------\n")

        return {
            "status": "close_up",
            "message": (
                "Single nail region analyzed."
            ),
            "original_dimensions": (
                width,
                height
            ),
            "crops": [
                {
                    "crop_image": image_bgr,
                    "crop_path": rel_crop_path,
                    "bbox": (
                        0,
                        0,
                        width,
                        height
                    ),
                    "dimensions": (
                        width,
                        height
                    )
                }
            ]
        }

    # =========================================================
    # FINAL DEBUG
    # =========================================================

    print(
        "Mode: Whole-hand multi-nail"
    )

    print(
        f"Number of detected nails: "
        f"{len(valid_crops)}"
    )

    for i, crop_info in enumerate(
        valid_crops
    ):

        print(
            f"  Nail {i + 1}: "
            f"Bounding Box={crop_info['bbox']}, "
            f"Crop Dimensions="
            f"{crop_info['dimensions']}"
        )

    print(
        "----------------------------\n"
    )

    # =========================================================
    # RETURN RESULT
    # =========================================================

    return {
        "status": "multi_nail",
        "message": (
            f"Whole-hand image analyzed: "
            f"{len(valid_crops)} nail regions "
            f"detected and analyzed."
        ),
        "original_dimensions": (
            width,
            height
        ),
        "crops": valid_crops
    }
