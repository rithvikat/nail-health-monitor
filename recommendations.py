"""
Nail Health Monitor - Recommendations & Screening Guidance Engine
Provides safe, non-diagnostic screening interpretations, condition-specific
recommendations, and structured longitudinal monitoring trend analysis.
"""

from typing import Optional, Dict, Any, List


# Exact Medical Disclaimer
MEDICAL_DISCLAIMER_TEXT = (
    "This AI screening report is intended strictly for informational, educational, and "
    "preliminary self-monitoring purposes. It does not constitute a medical diagnosis, "
    "clinical opinion, or dermatological assessment. AI predictions may not detect all "
    "medical conditions and may sometimes produce incorrect or uncertain results. Any "
    "health concerns or persistent nail changes should be evaluated by a licensed medical "
    "professional or qualified dermatologist."
)

# AI Model Information
MODEL_INFO = {
    "screening_type": "Image-based AI screening",
    "ai_model": "MobileNetV2 Feature Extractor + PCA-SVM Pipeline",
    "input_type": "Nail Image"
}


# ======================================================================
# CONDITION-SPECIFIC SCREENING RECOMMENDATIONS & CARE GUIDANCE (8 CLASSES)
# ======================================================================
CONDITION_RECOMMENDATIONS = {
    # 0 = Acral_Lentiginous_Melanoma
    0: {
        "condition_name": "Acral Lentiginous Melanoma",
        "priority_status": "Prompt Dermatological Evaluation Recommended",
        "recommended_action": (
            "The AI screening model identified image features associated with a pigmented nail pattern "
            "that requires professional assessment. Prompt evaluation by a qualified dermatologist is recommended."
        ),
        "clinical_guidance": (
            "A dermatologist can examine the nail and determine whether additional clinical evaluation is necessary."
        ),
        "daily_care": (
            "Maintain gentle nail hygiene and avoid unnecessary trauma to the affected nail."
        ),
        "what_to_avoid": (
            "Do not attempt to remove, scrape, or alter a pigmented nail area yourself."
        ),
        "suggested_monitoring": (
            "Record changes in pigmentation, shape, or appearance and discuss them with a qualified healthcare professional."
        )
    },
    # 1 = Blue_Finger
    1: {
        "condition_name": "Blue Finger",
        "priority_status": "Medical Evaluation Recommended",
        "recommended_action": (
            "The AI screening model identified image features associated with a bluish nail or finger appearance. "
            "Medical evaluation is recommended, particularly if the change is new or persistent."
        ),
        "clinical_guidance": (
            "A healthcare professional can evaluate the appearance and determine whether further assessment is appropriate."
        ),
        "daily_care": (
            "Keep the hands warm and maintain normal nail hygiene while monitoring any changes."
        ),
        "what_to_avoid": (
            "Avoid ignoring persistent or unexplained color changes."
        ),
        "suggested_monitoring": (
            "Record changes in nail or finger coloration and seek professional evaluation when appropriate."
        )
    },
    # 2 = Clubbing
    2: {
        "condition_name": "Clubbing",
        "priority_status": "Professional Evaluation Recommended",
        "recommended_action": (
            "The AI screening model identified image features associated with nail clubbing. "
            "Professional medical evaluation is recommended for further assessment."
        ),
        "clinical_guidance": (
            "A qualified healthcare professional can evaluate the nail and determine whether further examination is appropriate."
        ),
        "daily_care": (
            "Maintain gentle nail hygiene and avoid unnecessary pressure or trauma to the nails."
        ),
        "what_to_avoid": (
            "Avoid aggressive nail manipulation, picking, or attempting to correct the nail shape yourself."
        ),
        "suggested_monitoring": (
            "Continue periodic screening and record changes in nail appearance for discussion with a healthcare professional."
        )
    },
    # 3 = Healthy
    3: {
        "condition_name": "Healthy Nail",
        "priority_status": "Routine Nail Care",
        "recommended_action": (
            "No concerning nail pattern was identified by the AI screening model. "
            "Continue routine nail care and monitor for any persistent or noticeable changes."
        ),
        "clinical_guidance": (
            "Professional evaluation may be considered if new, persistent, or unusual nail changes develop."
        ),
        "daily_care": (
            "Keep nails clean and dry, maintain regular nail hygiene, and avoid excessive nail trauma."
        ),
        "what_to_avoid": (
            "Avoid picking, biting, or aggressively trimming the nails and cuticles."
        ),
        "suggested_monitoring": (
            "Continue periodic screening to monitor nail appearance over time."
        )
    },
    # 4 = Onychogryphosis
    4: {
        "condition_name": "Onychogryphosis",
        "priority_status": "Professional Evaluation Recommended",
        "recommended_action": (
            "The AI screening model identified image features associated with an abnormal thickening or curvature pattern. "
            "Professional evaluation is recommended."
        ),
        "clinical_guidance": (
            "A qualified healthcare professional can assess the nail and determine appropriate next steps."
        ),
        "daily_care": (
            "Maintain gentle nail hygiene and keep the surrounding skin clean and dry."
        ),
        "what_to_avoid": (
            "Avoid forcefully cutting, pulling, or attempting to reshape a severely thickened nail."
        ),
        "suggested_monitoring": (
            "Continue periodic screening to monitor changes in nail thickness and appearance."
        )
    },
    # 5 = Onychomycosis
    5: {
        "condition_name": "Onychomycosis",
        "priority_status": "Professional Evaluation Recommended",
        "recommended_action": (
            "The AI screening model identified image features associated with a possible fungal nail pattern. "
            "Professional evaluation is recommended for confirmation."
        ),
        "clinical_guidance": (
            "A healthcare professional can examine the nail and determine whether further assessment is needed."
        ),
        "daily_care": (
            "Keep nails clean and dry and maintain regular nail hygiene."
        ),
        "what_to_avoid": (
            "Avoid sharing nail-care tools and avoid damaging or aggressively trimming affected nails."
        ),
        "suggested_monitoring": (
            "Continue periodic screening to document changes in nail appearance."
        )
    },
    # 6 = Pitting
    6: {
        "condition_name": "Pitting",
        "priority_status": "Professional Evaluation Recommended",
        "recommended_action": (
            "The AI screening model identified image features associated with nail pitting. "
            "Professional evaluation is recommended, particularly if the changes persist."
        ),
        "clinical_guidance": (
            "A dermatologist or qualified healthcare professional can assess the nail appearance and determine whether further evaluation is appropriate."
        ),
        "daily_care": (
            "Maintain gentle nail hygiene and moisturize the surrounding skin when needed."
        ),
        "what_to_avoid": (
            "Avoid picking at the nail surface or causing repeated nail trauma."
        ),
        "suggested_monitoring": (
            "Continue periodic screening to monitor changes in nail surface appearance."
        )
    },
    # 7 = Psoriasis
    7: {
        "condition_name": "Psoriasis",
        "priority_status": "Professional Evaluation Recommended",
        "recommended_action": (
            "The AI screening model identified image features associated with psoriasis-related nail patterns. "
            "Professional dermatological evaluation is recommended for confirmation."
        ),
        "clinical_guidance": (
            "A qualified dermatologist can evaluate the nail appearance and determine whether further examination is appropriate."
        ),
        "daily_care": (
            "Keep nails clean and dry, maintain gentle nail hygiene, and avoid excessive nail trauma."
        ),
        "what_to_avoid": (
            "Avoid picking, aggressive cuticle trimming, or cleaning under the nail with sharp instruments."
        ),
        "suggested_monitoring": (
            "Continue periodic screening to monitor changes in nail appearance over time."
        )
    }
}


def resolve_class_index(predicted_class: Optional[Any] = None, predicted_label: Optional[str] = None) -> int:
    """
    Resolve predicted_class or predicted_label to a standardized 0..7 integer index.
    """
    if isinstance(predicted_class, int) and 0 <= predicted_class <= 7:
        return predicted_class

    if isinstance(predicted_class, str):
        if predicted_class.strip().isdigit():
            val = int(predicted_class.strip())
            if 0 <= val <= 7:
                return val
        norm_c = predicted_class.lower().replace(" ", "_").replace("-", "_")
        for idx, key in enumerate([
            "acral_lentiginous_melanoma", "blue_finger", "clubbing",
            "healthy", "onychogryphosis", "onychomycosis", "pitting", "psoriasis"
        ]):
            if key in norm_c:
                return idx

    if predicted_label:
        norm = predicted_label.lower().replace(" ", "_").replace("-", "_")
        for idx, key in enumerate([
            "acral_lentiginous_melanoma", "blue_finger", "clubbing",
            "healthy", "onychogryphosis", "onychomycosis", "pitting", "psoriasis"
        ]):
            if key in norm:
                return idx
        if "melanoma" in norm or "acral" in norm:
            return 0
        if "blue" in norm:
            return 1
        if "clubbing" in norm:
            return 2
        if "healthy" in norm:
            return 3
        if "onychogryphosis" in norm:
            return 4
        if "onychomycosis" in norm or "fungal" in norm:
            return 5
        if "pitting" in norm:
            return 6
        if "psoriasis" in norm:
            return 7

    return 3  # Default to Healthy Nail


def get_screening_interpretation(predicted_class: Optional[Any] = None, predicted_label: Optional[str] = None) -> str:
    """
    Generate dynamic safe, non-diagnostic AI screening interpretation.
    """
    idx = resolve_class_index(predicted_class, predicted_label)
    is_healthy = (idx == 3)
    if is_healthy:
        return (
            "The AI model did not detect features associated with the monitored nail conditions in this image. "
            "This is a preliminary AI screening result and should not be considered a medical diagnosis."
        )
    else:
        condition_name = CONDITION_RECOMMENDATIONS[idx]["condition_name"]
        return (
            f"The AI model identified image features associated with {condition_name}. "
            "This is a preliminary AI screening result and should not be considered a medical diagnosis."
        )


def get_safe_recommendations(predicted_class: Optional[Any] = None, predicted_label: Optional[str] = None) -> Dict[str, str]:
    """
    Return condition-specific screening-oriented recommendations using safe, non-prescriptive language.
    """
    idx = resolve_class_index(predicted_class, predicted_label)
    return CONDITION_RECOMMENDATIONS.get(idx, CONDITION_RECOMMENDATIONS[3])


# Backwards-compatible alias
get_condition_recommendations = get_safe_recommendations


def get_monitoring_progression(current_screening, previous_screening=None) -> Dict[str, Any]:
    """
    Calculate monitoring trend and change detection statement from actual stored history.
    """
    curr_label = getattr(current_screening, "predicted_label", "Unknown Finding")
    curr_class = getattr(current_screening, "predicted_class", None)

    if not previous_screening:
        return {
            "current_finding": curr_label,
            "previous_finding": "None (Initial Baseline)",
            "trend": "No Previous Record",
            "trend_class": "baseline",
            "change_detected": "No previous screening is available for comparison.",
            "has_previous": False
        }

    prev_label = getattr(previous_screening, "predicted_label", "Unknown Finding")
    prev_class = getattr(previous_screening, "predicted_class", None)

    # Calculate trend based on stored classification
    if curr_class == prev_class:
        trend = "Stable"
        trend_class = "stable"
        change_detected = "No significant change detected compared with the previous screening."
    else:
        trend = "Change Detected"
        trend_class = "changed"
        change_detected = "The current AI screening result differs from the previous screening result."

    return {
        "current_finding": curr_label,
        "previous_finding": prev_label,
        "trend": trend,
        "trend_class": trend_class,
        "change_detected": change_detected,
        "has_previous": True
    }


def get_full_recommendation(screening, previous_screening=None, top_2: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Compile complete bundle for web page display and report generation.
    """
    pred_class = getattr(screening, "predicted_class", 3)
    pred_label = getattr(screening, "predicted_label", "Healthy Nail")
    confidence = getattr(screening, "confidence", 0.0)

    # Use provided top_2, or screening.top_2_predictions, or fallback
    if top_2 is None:
        if hasattr(screening, "top_2_predictions"):
            top_2 = screening.top_2_predictions
        else:
            top_2 = [
                {
                    "class_id": pred_class,
                    "label": pred_label,
                    "confidence": confidence,
                    "percentage": round(confidence * 100, 1)
                }
            ]

    # Confidence rule:
    # If TOP-1 confidence < 80%: display low-confidence warning
    # If TOP-1 confidence >= 80%: do not display low-confidence warning
    show_low_confidence_warning = (confidence < 0.80)
    low_confidence_warning_text = (
        "IMPORTANT: The AI prediction has relatively low confidence. Professional evaluation is recommended."
    )

    interpretation = get_screening_interpretation(pred_class, pred_label)
    recommendations = get_safe_recommendations(pred_class, pred_label)
    monitoring = get_monitoring_progression(screening, previous_screening)

    priority_status = recommendations.get("priority_status", "Professional Evaluation Recommended")
    priority_norm = priority_status.lower()
    if "prompt" in priority_norm:
        p_type = "danger"
    elif "routine" in priority_norm:
        p_type = "routine"
    else:
        p_type = "warning"

    condition_data = {
        "summary": recommendations["recommended_action"],
        "priority_level": priority_status,
        "priority_type": p_type,
        "clinical_action": recommendations["clinical_guidance"],
        "daily_care": recommendations["daily_care"],
        "what_to_avoid": recommendations["what_to_avoid"],
        "suggested_monitoring": recommendations["suggested_monitoring"]
    }

    return {
        "interpretation": interpretation,
        "recommendations": recommendations,
        "monitoring": monitoring,
        "top_2": top_2,
        "show_low_confidence_warning": show_low_confidence_warning,
        "low_confidence_warning_text": low_confidence_warning_text,
        "disclaimer": MEDICAL_DISCLAIMER_TEXT,
        "model_info": MODEL_INFO,
        "condition": condition_data
    }
