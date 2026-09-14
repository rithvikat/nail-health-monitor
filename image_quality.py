import cv2
import numpy as np
import os


# ============================================================
# NAIL IMAGE QUALITY + VISIBILITY CHECKER
# ============================================================

def check_nail_image(image_path):

    # --------------------------------------------------------
    # 1. Load image
    # --------------------------------------------------------
    image = cv2.imread(image_path)

    if image is None:
        return {
            "suitable": False,
            "status": "REJECT",
            "reasons": ["Image could not be read"],
            "warnings": []
        }

    height, width = image.shape[:2]

    reasons = []
    warnings = []

    # ========================================================
    # A. BASIC IMAGE CHECK
    # ========================================================

    # --------------------------------------------------------
    # 2. Resolution
    # --------------------------------------------------------

    if width < 400 or height < 400:
        reasons.append(
            "Image resolution is too low"
        )

    # --------------------------------------------------------
    # 3. Convert to grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------------
    # 4. Central region
    #
    # Avoid letting the background dominate the quality score.
    # --------------------------------------------------------

    x1 = int(width * 0.15)
    x2 = int(width * 0.85)

    y1 = int(height * 0.15)
    y2 = int(height * 0.85)

    roi = gray[y1:y2, x1:x2]

    # ========================================================
    # B. BLUR CHECK
    # ========================================================

    blur_score = cv2.Laplacian(
        roi,
        cv2.CV_64F
    ).var()

    print("Blur score:", round(blur_score, 2))

    # Very low = genuinely difficult to analyze
    if blur_score < 4:

        reasons.append(
            "Image is too blurry to see nail details"
        )

    # Slight softness = warning only
    elif blur_score < 8:

        warnings.append(
            "Image is slightly soft"
        )

    # ========================================================
    # C. BRIGHTNESS CHECK
    # ========================================================

    brightness = np.mean(roi)

    print("Brightness:", round(brightness, 2))

    if brightness < 35:

        reasons.append(
            "Image is too dark"
        )

    elif brightness > 225:

        reasons.append(
            "Image is too bright"
        )

    elif brightness < 50:

        warnings.append(
            "Lighting is somewhat low"
        )

    # ========================================================
    # D. CONTRAST CHECK
    # ========================================================

    contrast = np.std(roi)

    print("Contrast:", round(contrast, 2))

    if contrast < 12:

        reasons.append(
            "Image has very low contrast"
        )

    elif contrast < 20:

        warnings.append(
            "Image has low contrast"
        )

    # ========================================================
    # E. OVEREXPOSURE CHECK
    # ========================================================

    overexposed_ratio = np.mean(roi > 245)

    print(
        "Overexposed:",
        round(overexposed_ratio * 100, 2),
        "%"
    )

    if overexposed_ratio > 0.35:

        reasons.append(
            "Large part of the image is overexposed"
        )

    # ========================================================
    # F. UNDEREXPOSURE CHECK
    # ========================================================

    underexposed_ratio = np.mean(roi < 15)

    print(
        "Underexposed:",
        round(underexposed_ratio * 100, 2),
        "%"
    )

    if underexposed_ratio > 0.35:

        reasons.append(
            "Large part of the image is too dark"
        )

    # ========================================================
    # G. EDGE / DETAIL CHECK
    # ========================================================
    #
    # A nail normally contains some boundaries and surface
    # information. This is only a rough visibility check.
    #
    # It is NOT a medical nail detector.
    # ========================================================

    edges = cv2.Canny(
        roi,
        threshold1=50,
        threshold2=150
    )

    edge_ratio = np.mean(edges > 0)

    print(
        "Edge/detail ratio:",
        round(edge_ratio * 100, 2),
        "%"
    )

    if edge_ratio < 0.002:

        warnings.append(
            "Very little visible detail detected"
        )

    # ========================================================
    # H. POSSIBLE COVERING / POLISH CHECK
    # ========================================================
    #
    # IMPORTANT:
    # This is a HEURISTIC.
    #
    # It does NOT reliably identify nail polish.
    # It only looks for unusually uniform regions.
    #
    # Later, this should be replaced by a trained classifier.
    # ========================================================

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Saturation indicates how strongly colored an area is.
    saturation = hsv[:, :, 1]

    central_sat = saturation[y1:y2, x1:x2]

    saturation_mean = np.mean(central_sat)

    print(
        "Average saturation:",
        round(saturation_mean, 2)
    )

    # Very high saturation can indicate strong artificial
    # coloring, but can also simply be skin/background.
    if saturation_mean > 170:

        warnings.append(
            "Strong surface coloring detected; possible covering or polish"
        )

    # ========================================================
    # I. IMAGE COMPOSITION CHECK
    # ========================================================

    # If the image contains almost no edges/details,
    # the nail may be too small or not properly visible.

    if edge_ratio < 0.005:

        warnings.append(
            "Nail may not be clearly positioned in the image"
        )

    # ========================================================
    # J. FINAL DECISION
    # ========================================================

    if len(reasons) == 0:

        suitable = True
        status = "ACCEPT"

    else:

        suitable = False
        status = "REJECT"

    # ========================================================
    # K. RETURN RESULT
    # ========================================================

    return {
        "suitable": suitable,
        "status": status,
        "reasons": reasons,
        "warnings": warnings,

        "metrics": {
            "width": width,
            "height": height,
            "blur_score": round(blur_score, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "edge_ratio": round(edge_ratio, 4),
            "saturation": round(saturation_mean, 2),
            "overexposed_percent":
                round(overexposed_ratio * 100, 2),
            "underexposed_percent":
                round(underexposed_ratio * 100, 2)
        }
    }


# ============================================================
# DISPLAY RESULT
# ============================================================

def print_result(result):

    print("\n")
    print("=" * 55)
    print("        NAIL IMAGE QUALITY CHECK")
    print("=" * 55)

    if result["status"] == "ACCEPT":

        print("STATUS : ✓ ACCEPTED")

    else:

        print("STATUS : ✗ REJECTED")

    # --------------------------------------------------------
    # Reasons
    # --------------------------------------------------------

    print("\nProblems:")

    if len(result["reasons"]) == 0:

        print("  None")

    else:

        for reason in result["reasons"]:

            print("  ✗", reason)

    # --------------------------------------------------------
    # Warnings
    # --------------------------------------------------------

    print("\nWarnings:")

    if len(result["warnings"]) == 0:

        print("  None")

    else:

        for warning in result["warnings"]:

            print("  ⚠", warning)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    print("\nImage measurements:")

    for key, value in result["metrics"].items():

        print(f"  {key}: {value}")

    print("=" * 55)


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # Change this to your image
    image_path = "test2.jpeg"

    if not os.path.exists(image_path):

        print(
            f"Image not found: {image_path}"
        )

    else:

        result = check_nail_image(image_path)

        print_result(result)