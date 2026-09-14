import os
import cv2
import numpy as np
from PIL import Image


def check_nail_image(image_path: str) -> dict:
    """
    Multi-factor quality gate for nail image suitability.
    Runs BEFORE ML inference. Rejects unsuitable images with distinct,
    actionable, non-confusing reasons.

    Returns:
        {
            "suitable": bool,
            "message": str,
            "reasons": list[str]
        }
    """
    reasons = []

    # 1. File existence check
    if not os.path.exists(image_path):
        return {
            "suitable": False,
            "message": "Image Not Suitable. Please retake the image.",
            "reasons": ["Image file could not be found."]
        }

    # 2. PIL integrity check
    try:
        with Image.open(image_path) as pil_img:
            pil_img.verify()
    except Exception:
        return {
            "suitable": False,
            "message": "Image Not Suitable. Please retake the image.",
            "reasons": ["Image file is invalid or corrupted. Please upload a valid JPEG or PNG image."]
        }

    # 3. OpenCV reading
    image = cv2.imread(image_path)
    if image is None:
        return {
            "suitable": False,
            "message": "Image Not Suitable. Please retake the image.",
            "reasons": ["Could not decode image. Please ensure the file is an uncorrupted image."]
        }

    height, width = image.shape[:2]

    # 4. Minimum resolution check (ensure adequate spatial resolution)
    if width < 300 or height < 300:
        return {
            "suitable": False,
            "message": "Image Not Suitable. Please retake the image.",
            "reasons": ["Image resolution is too low. Please upload a photo with at least 300x300 pixels."]
        }

    # 5. Extract Central Region of Interest (ROI) - central 70%
    x1 = int(width * 0.15)
    x2 = int(width * 0.85)
    y1 = int(height * 0.15)
    y2 = int(height * 0.85)
    color_roi = image[y1:y2, x1:x2] if image.ndim == 3 else image
    gray_roi = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)

    # ========================================================
    # CHECK A: NON-NAIL DETECTION (Is there a finger/nail here?)
    # ========================================================
    # Checks for natural human skin/nail tissue across all human ethnicities.
    ycrcb = cv2.cvtColor(color_roi, cv2.COLOR_BGR2YCR_CB)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    skin_ycrcb = (cr >= 128) & (cr <= 175) & (cb >= 75) & (cb <= 135)

    hsv_roi = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
    H, S, V = hsv_roi[:, :, 0], hsv_roi[:, :, 1], hsv_roi[:, :, 2]
    skin_hsv = ((H <= 25) | (H >= 165)) & (S >= 20) & (S <= 200) & (V >= 35)

    skin_ratio = float(np.mean(skin_ycrcb & skin_hsv))

    # If virtually no human skin/flesh tone is detected, the image is not a nail
    if skin_ratio < 0.05:
        return {
            "suitable": False,
            "message": "Image Not Suitable. Please retake the image.",
            "reasons": ["No suitable nail region could be detected in this image. Please upload a clearer image with the nail(s) fully visible."]
        }

    # ========================================================
    # CHECK B: EXPOSURE & LIGHTING
    # ========================================================
    brightness = float(np.mean(gray_roi))
    underexposed_ratio = float(np.mean(gray_roi < 15))
    overexposed_ratio = float(np.mean(gray_roi > 245))
    contrast = float(np.std(gray_roi))

    if brightness < 35.0 or underexposed_ratio > 0.35:
        reasons.append("Image is too dark. Please take the photo in better lighting.")
    elif brightness > 225.0 or overexposed_ratio > 0.35:
        reasons.append("Image is overexposed. Please avoid direct light.")

    if contrast < 12.0:
        reasons.append("Image has very low contrast. Please ensure clear lighting.")

    # ========================================================
    # CHECK C: BLUR & SHARPNESS
    # ========================================================
    laplacian_var = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())
    gx = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = float(np.mean(np.sqrt(gx**2 + gy**2)))

    if laplacian_var < 9.0 or (laplacian_var < 10.0 and sobel_mag < 9.5):
        reasons.append("Image is too blurry. Please hold your hand steady and retake the photo.")

    # Deduplicate reasons while preserving order
    unique_reasons = []
    for r in reasons:
        if r not in unique_reasons:
            unique_reasons.append(r)

    # Decision
    if len(unique_reasons) == 0:
        return {
            "suitable": True,
            "message": "Image quality is suitable for screening.",
            "reasons": []
        }
    else:
        return {
            "suitable": False,
            "message": "Image Not Suitable. Please retake the image.",
            "reasons": unique_reasons
        }
