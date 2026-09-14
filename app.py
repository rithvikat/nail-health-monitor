import os
import uuid
import hmac
import hashlib
import secrets
import time
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Form, UploadFile, File, Depends, status
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import (
    init_db,
    get_db,
    User,
    Screening,
    Report,
    Consultation,
    hash_password,
    verify_password
)
from image_checker import check_nail_image
from nail_detector import detect_and_crop_nails
from ml_model import load_ml_models, predict_nail
from reports import generate_pdf_report
from recommendations import get_full_recommendation, get_condition_recommendations


# ============================================================
# APP CONFIGURATION & LIFESPAN
# ============================================================

SESSION_SECRET = "nail_monitor_secret_key_2026_secure"
SESSION_COOKIE_NAME = "nail_session"

# Ensure critical directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("uploads/reports", exist_ok=True)
os.makedirs("uploads/crops", exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize SQLite database and load ML models once at startup."""
    print("Initializing SQLite database...")
    init_db()
    print("Database tables initialized.")

    print("Loading existing ML models...")
    try:
        load_ml_models()
        print("MobileNetV2 feature extractor and PCA+SVM classifier loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load ML models at startup: {e}")

    yield
    print("Shutting down Nail Health Monitor.")


app = FastAPI(title="Nail Health Monitor", lifespan=lifespan)

# Mount static and media directories
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

templates = Jinja2Templates(directory="templates")
templates.env.globals["get_recommendation"] = get_full_recommendation


# ============================================================
# SESSION & AUTHENTICATION HELPERS
# ============================================================

def make_session_token(user_id: int) -> str:
    """Create a signed, tamper-proof session token."""
    payload = f"{user_id}:{int(time.time())}"
    signature = hmac.new(
        SESSION_SECRET.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    return f"{payload}:{signature}"


def verify_session_token(token: str) -> Optional[int]:
    """Verify session token signature and extract user_id."""
    if not token or ":" not in token:
        return None
    try:
        parts = token.split(":")
        if len(parts) != 3:
            return None
        user_id_str, timestamp_str, sig = parts
        payload = f"{user_id_str}:{timestamp_str}"
        expected_sig = hmac.new(
            SESSION_SECRET.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        if hmac.compare_digest(sig, expected_sig):
            return int(user_id_str)
    except Exception:
        return None
    return None


def get_current_user(request: Request, db: Session = Depends(get_db)) -> Optional[User]:
    """Retrieve logged-in user from signed cookie."""
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return None
    user_id = verify_session_token(token)
    if not user_id:
        return None
    return db.query(User).filter(User.id == user_id).first()


def require_auth(request: Request, db: Session = Depends(get_db)) -> User:
    """Dependency that enforces active authentication."""
    user = get_current_user(request, db)
    if not user:
        raise PermissionError("Authentication required")
    return user


class DoctorAccessRequired(Exception):
    """Raised when non-doctor user attempts to access doctor portal."""
    pass


@app.exception_handler(DoctorAccessRequired)
async def doctor_access_handler(request: Request, exc: DoctorAccessRequired):
    return RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)


def require_doctor(request: Request, db: Session = Depends(get_db)) -> User:
    """Dependency that enforces doctor authentication."""
    user = require_auth(request, db)
    if not user.is_doctor:
        raise DoctorAccessRequired()
    return user


# Exception handler for unauthenticated access
@app.exception_handler(PermissionError)
async def auth_exception_handler(request: Request, exc: PermissionError):
    return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)


# ============================================================
# AUTHENTICATION ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
def root(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if user:
        if user.is_doctor:
            return RedirectResponse(url="/doctor/dashboard", status_code=status.HTTP_302_FOUND)
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if user:
        if user.is_doctor:
            return RedirectResponse(url="/doctor/dashboard", status_code=status.HTTP_302_FOUND)
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"error": None}
    )


@app.post("/login", response_class=HTMLResponse)
def login_post(
    request: Request,
    name: Optional[str] = Form(None),
    password: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    name_clean = (name or "").strip()
    password_clean = (password or "").strip()

    # 1. Validate presence of name and password
    if not name_clean:
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={"error": "Please enter your name."},
            status_code=status.HTTP_400_BAD_REQUEST
        )

    if not password_clean:
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={"error": "Please enter your password.", "name": name_clean},
            status_code=status.HTTP_400_BAD_REQUEST
        )

    # 2. Reliable case-insensitive and trimmed name lookup
    users = db.query(User).filter(func.lower(func.trim(User.name)) == func.lower(name_clean)).all()

    if not users:
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={"error": "User not found.", "name": name_clean},
            status_code=status.HTTP_404_NOT_FOUND
        )

    # 3. Compare entered password with stored password hash using existing verify_password
    matched_user = None
    for u in users:
        if verify_password(password_clean, u.password_hash):
            matched_user = u
            break

    if not matched_user:
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={"error": "Incorrect password.", "name": name_clean},
            status_code=status.HTTP_401_UNAUTHORIZED
        )

    user = matched_user
    redirect_url = "/doctor/dashboard" if user.is_doctor else "/dashboard"
    response = RedirectResponse(url=redirect_url, status_code=status.HTTP_303_SEE_OTHER)
    token = make_session_token(user.id)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        max_age=86400 * 7, # 7 days
        samesite="lax"
    )
    return response


@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if user:
        if user.is_doctor:
            return RedirectResponse(url="/doctor/dashboard", status_code=status.HTTP_302_FOUND)
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse(
        request=request,
        name="register.html",
        context={"error": None}
    )


@app.post("/register", response_class=HTMLResponse)
def register_post(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form("patient"),
    db: Session = Depends(get_db)
):
    name_clean = name.strip()
    email_clean = email.strip().lower()

    if not name_clean or not email_clean or not password:
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={"error": "All fields are required."},
            status_code=status.HTTP_400_BAD_REQUEST
        )

    existing_user = db.query(User).filter(User.email == email_clean).first()
    if existing_user:
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={"error": "An account with this email already exists. Please log in."},
            status_code=status.HTTP_400_BAD_REQUEST
        )

    is_doc = (role.strip().lower() == "doctor")
    pwd_hash = hash_password(password)
    new_user = User(
        name=name_clean,
        email=email_clean,
        password_hash=pwd_hash,
        is_doctor=is_doc,
        doctor_license=f"#MED-{secrets.token_hex(3).upper()}" if is_doc else None,
        doctor_specialty="Board-Certified Dermatologist & Nail Biology Specialist" if is_doc else None
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Automatically log the user in after registration
    redirect_url = "/doctor/dashboard" if is_doc else "/dashboard"
    response = RedirectResponse(url=redirect_url, status_code=status.HTTP_303_SEE_OTHER)
    token = make_session_token(new_user.id)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        max_age=86400 * 7,
        samesite="lax"
    )
    return response


@app.get("/logout")
def logout():
    response = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(key=SESSION_COOKIE_NAME)
    return response


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(
    request: Request,
    user: User = Depends(require_auth),
    db: Session = Depends(get_db)
):
    if user.is_doctor:
        return RedirectResponse(url="/doctor/dashboard", status_code=status.HTTP_303_SEE_OTHER)

    # Fetch user-specific records
    screenings = db.query(Screening).filter(
        Screening.user_id == user.id
    ).order_by(Screening.created_at.desc()).all()

    total_screenings = len(screenings)
    latest_screening = screenings[0] if total_screenings > 0 else None
    previous_screening = screenings[1] if total_screenings > 1 else None
    latest_recommendation = get_full_recommendation(latest_screening, previous_screening) if latest_screening else None

    reports = db.query(Report).join(Screening).filter(
        Screening.user_id == user.id
    ).all()
    total_reports = len(reports)
    latest_report = latest_screening.report if latest_screening else None

    total_consultations = db.query(Consultation).filter(
        Consultation.user_id == user.id
    ).count()

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "user": user,
            "total_screenings": total_screenings,
            "latest_screening": latest_screening,
            "latest_recommendation": latest_recommendation,
            "latest_report": latest_report,
            "total_reports": total_reports,
            "total_consultations": total_consultations
        }
    )


# ============================================================
# SCREENING (UPLOAD, QUALITY GATE, ML, STORAGE, REPORT)
# ============================================================

@app.get("/screening", response_class=HTMLResponse)
def screening_page(
    request: Request,
    user: User = Depends(require_auth)
):
    return templates.TemplateResponse(
        request=request,
        name="screening.html",
        context={
            "user": user,
            "rejected": False,
            "reasons": [],
            "success": False,
            "screening": None,
            "report": None
        }
    )


@app.post("/screening", response_class=HTMLResponse)
async def screening_post(
    request: Request,
    image: UploadFile = File(...),
    user: User = Depends(require_auth),
    db: Session = Depends(get_db)
):
    # 1. Validate file format
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    orig_filename = image.filename or "upload.jpg"
    ext = os.path.splitext(orig_filename)[1].lower()

    if ext not in allowed_extensions:
        return templates.TemplateResponse(
            request=request,
            name="screening.html",
            context={
                "user": user,
                "rejected": True,
                "reasons": ["Invalid file format. Please upload a standard JPEG or PNG image."],
                "success": False,
                "screening": None,
                "report": None
            }
        )

    # 2. Save to a TEMPORARY file for quality inspection
    temp_filename = f"temp_{uuid.uuid4().hex}{ext}"
    temp_path = os.path.join("uploads", temp_filename)

    try:
        contents = await image.read()
        with open(temp_path, "wb") as f:
            f.write(contents)
    except Exception:
        return templates.TemplateResponse(
            request=request,
            name="screening.html",
            context={
                "user": user,
                "rejected": True,
                "reasons": ["Could not read uploaded file. Please try again."],
                "success": False,
                "screening": None,
                "report": None
            }
        )

    # 3. HARD QUALITY GATE - image_checker.py MUST run BEFORE ML model
    quality_result = check_nail_image(temp_path)

    if not quality_result["suitable"]:
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return templates.TemplateResponse(
            request=request,
            name="screening.html",
            context={
                "user": user,
                "rejected": True,
                "reasons": quality_result["reasons"],
                "success": False,
                "screening": None,
                "report": None
            }
        )

    # 4. NAIL REGION DETECTION & CROPPING PREPARATION STAGE
    detection_result = detect_and_crop_nails(temp_path)

    if detection_result["status"] == "none":
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return templates.TemplateResponse(
            request=request,
            name="screening.html",
            context={
                "user": user,
                "rejected": True,
                "reasons": [detection_result["message"]],
                "success": False,
                "screening": None,
                "report": None
            }
        )

    # 5. IF GOOD: Move original image to permanent storage with safe filename
    safe_filename = f"nail_{user.id}_{uuid.uuid4().hex[:8]}_{int(time.time())}{ext}"
    perm_path = os.path.join("uploads", safe_filename)
    os.rename(temp_path, perm_path)

    # Convert path for web display with forward slashes: uploads/...
    web_image_path = f"uploads/{safe_filename}"

    # 6. RUN ML INFERENCE (MobileNetV2 feature extractor -> PCA + SVM) ON NAIL CROP(S)
    crops_info = detection_result.get("crops", [])
    detected_nails = []

    try:
        if detection_result["status"] == "multi_nail" and len(crops_info) > 1:
            # Multi-nail whole-hand mode: analyze each visible nail individually
            for idx, c in enumerate(crops_info):
                crop_res = predict_nail(c["crop_path"])
                conf = crop_res["confidence"]
                detected_nails.append({
                    "nail_number": idx + 1,
                    "crop_path": c["crop_path"],
                    "bbox": c["bbox"],
                    "dimensions": c["dimensions"],
                    "predicted_class": crop_res["predicted_class"],
                    "predicted_label": crop_res["predicted_label"],
                    "confidence": conf,
                    "percentage": round(conf * 100, 1),
                    "show_low_confidence_warning": conf < 0.80,
                    "low_confidence_warning_text": "IMPORTANT: The AI prediction has relatively low confidence. Professional evaluation is recommended."
                })

            # Primary screening record set to the most confident detected nail
            primary_nail = max(detected_nails, key=lambda n: n["confidence"])
            prediction = {
                "predicted_class": primary_nail["predicted_class"],
                "predicted_label": primary_nail["predicted_label"],
                "confidence": primary_nail["confidence"],
                "top_2": primary_nail["top_2"]
            }
        else:
            # Single-nail close-up mode
            primary_crop_path = crops_info[0]["crop_path"] if crops_info else perm_path
            prediction = predict_nail(primary_crop_path)
            detected_nails = [
                {
                    "nail_number": 1,
                    "crop_path": primary_crop_path,
                    "bbox": crops_info[0]["bbox"] if crops_info else (0, 0, 0, 0),
                    "dimensions": crops_info[0]["dimensions"] if crops_info else (0, 0),
                    "predicted_class": prediction["predicted_class"],
                    "predicted_label": prediction["predicted_label"],
                    "confidence": prediction["confidence"],
                    "percentage": round(prediction["confidence"] * 100, 1),
                    "top_2": prediction["top_2"],
                    "show_low_confidence_warning": prediction["confidence"] < 0.80,
                    "low_confidence_warning_text": "IMPORTANT: The AI prediction has relatively low confidence. Professional evaluation is recommended."
                }
            ]

    except Exception as e:
        print(f"ML Inference Error: {e}")
        if os.path.exists(perm_path):
            os.remove(perm_path)
        return templates.TemplateResponse(
            request=request,
            name="screening.html",
            context={
                "user": user,
                "rejected": True,
                "reasons": ["An unexpected error occurred during AI analysis. Please try again."],
                "success": False,
                "screening": None,
                "report": None
            }
        )

    # 7. SAVE SUCCESSFUL SCREENING TO DATABASE
    import json
    top2_str = json.dumps(prediction.get("top_2", []))
    multi_nail_str = json.dumps(detected_nails) if len(detected_nails) > 1 else None

    new_screening = Screening(
        user_id=user.id,
        image_path=web_image_path,
        predicted_class=prediction["predicted_class"],
        predicted_label=prediction["predicted_label"],
        confidence=prediction["confidence"],
        top2_json=top2_str,
        multi_nail_json=multi_nail_str
    )
    db.add(new_screening)
    db.commit()
    db.refresh(new_screening)

    # 8. FETCH PREVIOUS SCREENING (FOR MONITORING COMPARISON)
    previous_screening = db.query(Screening).filter(
        Screening.user_id == user.id,
        Screening.id != new_screening.id
    ).order_by(Screening.created_at.desc()).first()

    # 9. GENERATE PDF REPORT
    try:
        report_pdf_path = generate_pdf_report(
            new_screening,
            user,
            previous_screening,
            top_2=prediction.get("top_2")
        )
        new_report = Report(
            screening_id=new_screening.id,
            report_path=report_pdf_path
        )
        db.add(new_report)
        db.commit()
        db.refresh(new_report)
    except Exception as e:
        print(f"Report Generation Error: {e}")
        new_report = None

    # 10. COMPILE RECOMMENDATIONS, INTERPRETATION & MONITORING BUNDLE
    rec_bundle = get_full_recommendation(
        new_screening,
        previous_screening,
        top_2=prediction.get("top_2")
    )

    # 11. RETURN SUCCESSFUL RESULT
    return templates.TemplateResponse(
        request=request,
        name="screening.html",
        context={
            "user": user,
            "rejected": False,
            "reasons": [],
            "success": True,
            "screening": new_screening,
            "report": new_report,
            "top_2": rec_bundle["top_2"],
            "show_low_confidence_warning": rec_bundle["show_low_confidence_warning"],
            "low_confidence_warning_text": rec_bundle["low_confidence_warning_text"],
            "interpretation": rec_bundle["interpretation"],
            "recommendations": rec_bundle["recommendations"],
            "monitoring": rec_bundle["monitoring"],
            "disclaimer": rec_bundle["disclaimer"],
            "recommendation": rec_bundle,
            "detected_nails": detected_nails,
            "detection_mode": detection_result["status"],
            "detection_message": detection_result["message"]
        }
    )


# ============================================================
# HISTORY & MONITORING TIMELINE
# ============================================================

@app.get("/history", response_class=HTMLResponse)
def history_page(
    request: Request,
    user: User = Depends(require_auth),
    db: Session = Depends(get_db)
):
    # Fetch only the logged-in user's successful screenings
    screenings = db.query(Screening).filter(
        Screening.user_id == user.id
    ).order_by(Screening.created_at.desc()).all()

    # Build chronological timeline with comparison pairs and recommendations
    timeline_items = []
    for i, curr in enumerate(screenings):
        prev = screenings[i + 1] if i + 1 < len(screenings) else None
        rec = get_full_recommendation(curr, prev)
        timeline_items.append({
            "current": curr,
            "previous": prev,
            "recommendation": rec
        })

    # Attach recommendation to each screening for table view
    for s in screenings:
        s.recommendation = get_full_recommendation(s, None)

    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={
            "user": user,
            "screenings": screenings,
            "timeline_items": timeline_items
        }
    )


# ============================================================
# REPORTS
# ============================================================

@app.get("/reports", response_class=HTMLResponse)
def reports_page(
    request: Request,
    user: User = Depends(require_auth),
    db: Session = Depends(get_db)
):
    reports = db.query(Report).join(Screening).filter(
        Screening.user_id == user.id
    ).order_by(Report.created_at.desc()).all()

    # Attach recommendation to each screening report
    for r in reports:
        if r.screening:
            r.screening.recommendation = get_full_recommendation(r.screening, None)

    return templates.TemplateResponse(
        request=request,
        name="reports.html",
        context={
            "user": user,
            "reports": reports
        }
    )


@app.get("/reports/{report_id}")
def view_report_by_id(
    report_id: int,
    user: User = Depends(require_auth),
    db: Session = Depends(get_db)
):
    if user.is_doctor:
        report = db.query(Report).filter(Report.id == report_id).first()
    else:
        report = db.query(Report).join(Screening).filter(
            Report.id == report_id,
            Screening.user_id == user.id
        ).first()

    if not report or not os.path.exists(report.report_path):
        return RedirectResponse(
            url="/doctor/dashboard" if user.is_doctor else "/reports",
            status_code=status.HTTP_303_SEE_OTHER
        )

    return FileResponse(report.report_path, media_type="application/pdf")


# ============================================================
# DOCTOR CONSULTATION
# ============================================================

@app.get("/consultation", response_class=HTMLResponse)
def consultation_page(
    request: Request,
    screening_id: Optional[int] = None,
    user: User = Depends(require_auth),
    db: Session = Depends(get_db)
):
    screenings = db.query(Screening).filter(
        Screening.user_id == user.id
    ).order_by(Screening.created_at.desc()).all()

    consultations = db.query(Consultation).filter(
        Consultation.user_id == user.id
    ).order_by(Consultation.created_at.desc()).all()

    return templates.TemplateResponse(
        request=request,
        name="consultation.html",
        context={
            "user": user,
            "screenings": screenings,
            "selected_id": screening_id,
            "consultations": consultations,
            "success_msg": None,
            "error_msg": None
        }
    )


@app.post("/consultation", response_class=HTMLResponse)
def consultation_post(
    request: Request,
    screening_id: int = Form(...),
    message: str = Form(...),
    user: User = Depends(require_auth),
    db: Session = Depends(get_db)
):
    message_clean = message.strip()
    screenings = db.query(Screening).filter(
        Screening.user_id == user.id
    ).order_by(Screening.created_at.desc()).all()

    # Validate that screening belongs to user
    target_screening = db.query(Screening).filter(
        Screening.id == screening_id,
        Screening.user_id == user.id
    ).first()

    if not target_screening or not message_clean:
        consultations = db.query(Consultation).filter(
            Consultation.user_id == user.id
        ).order_by(Consultation.created_at.desc()).all()

        return templates.TemplateResponse(
            request=request,
            name="consultation.html",
            context={
                "user": user,
                "screenings": screenings,
                "selected_id": screening_id,
                "consultations": consultations,
                "success_msg": None,
                "error_msg": "Please select a valid screening and include a message for the doctor."
            },
            status_code=status.HTTP_400_BAD_REQUEST
        )

    # Create consultation record with initial pending status
    new_consultation = Consultation(
        user_id=user.id,
        screening_id=screening_id,
        message=message_clean,
        doctor_response=None,
        status="Pending"
    )
    db.add(new_consultation)
    db.commit()

    consultations = db.query(Consultation).filter(
        Consultation.user_id == user.id
    ).order_by(Consultation.created_at.desc()).all()

    return templates.TemplateResponse(
        request=request,
        name="consultation.html",
        context={
            "user": user,
            "screenings": screenings,
            "selected_id": screening_id,
            "consultations": consultations,
            "success_msg": f"Consultation requested successfully for Screening #{screening_id}. A healthcare provider will review your inquiry.",
            "error_msg": None
        }
    )


# ============================================================
# DOCTOR PORTAL
# ============================================================

@app.get("/doctor/dashboard", response_class=HTMLResponse)
def doctor_dashboard(
    request: Request,
    doctor: User = Depends(require_doctor),
    db: Session = Depends(get_db)
):
    consultations = db.query(Consultation).order_by(Consultation.created_at.desc()).all()
    pending_count = db.query(Consultation).filter(Consultation.status == "Pending").count()
    reviewed_count = db.query(Consultation).filter(Consultation.status == "Reviewed").count()
    total_consultations = len(consultations)

    return templates.TemplateResponse(
        request=request,
        name="doctor_dashboard.html",
        context={
            "doctor": doctor,
            "consultations": consultations,
            "pending_count": pending_count,
            "reviewed_count": reviewed_count,
            "total_consultations": total_consultations
        }
    )


@app.post("/doctor/respond/{consultation_id}")
def doctor_respond_post(
    consultation_id: int,
    doctor_response: str = Form(...),
    doctor: User = Depends(require_doctor),
    db: Session = Depends(get_db)
):
    consultation = db.query(Consultation).filter(Consultation.id == consultation_id).first()
    if not consultation:
        return RedirectResponse(url="/doctor/dashboard", status_code=status.HTTP_303_SEE_OTHER)

    response_text = doctor_response.strip()
    if response_text:
        consultation.doctor_response = response_text
        consultation.status = "Reviewed"
        db.commit()

    return RedirectResponse(url="/doctor/dashboard", status_code=status.HTTP_303_SEE_OTHER)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

