import json
import hashlib
import secrets
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = "sqlite:///./database.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ============================================================
# PASSWORD HASHING UTILITIES
# ============================================================

def hash_password(password: str) -> str:
    """Hash a password using PBKDF2 with SHA-256 and a random salt."""
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000
    )
    return f"{salt}${key.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a plain password against the stored salt$hash string."""
    try:
        salt, key_hex = stored_hash.split("$", 1)
        expected_key = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            100000
        )
        return secrets.compare_digest(key_hex, expected_key.hex())
    except Exception:
        return False


# ============================================================
# DATABASE MODELS
# ============================================================

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_doctor = Column(Boolean, default=False)
    doctor_license = Column(String(100), nullable=True, default=None)
    doctor_specialty = Column(String(150), nullable=True, default=None)
    created_at = Column(DateTime, default=datetime.utcnow)

    screenings = relationship("Screening", back_populates="user", cascade="all, delete-orphan")
    consultations = relationship("Consultation", back_populates="user", cascade="all, delete-orphan")


class Screening(Base):
    __tablename__ = "screenings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    image_path = Column(String(255), nullable=False)
    predicted_class = Column(Integer, nullable=False)
    predicted_label = Column(String(150), nullable=False)
    confidence = Column(Float, nullable=False)
    top2_json = Column(Text, nullable=True, default=None)
    multi_nail_json = Column(Text, nullable=True, default=None)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="screenings")
    report = relationship("Report", back_populates="screening", uselist=False, cascade="all, delete-orphan")
    consultation = relationship("Consultation", back_populates="screening", uselist=False)

    @property
    def top_2_predictions(self):
        if self.top2_json:
            try:
                return json.loads(self.top2_json)
            except Exception:
                pass
        return [
            {
                "class_id": self.predicted_class,
                "label": self.predicted_label,
                "confidence": self.confidence,
                "percentage": round(self.confidence * 100, 1)
            }
        ]

    @property
    def multi_nail_results(self):
        if self.multi_nail_json:
            try:
                return json.loads(self.multi_nail_json)
            except Exception:
                pass
        return None


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    screening_id = Column(Integer, ForeignKey("screenings.id"), nullable=False)
    report_path = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    screening = relationship("Screening", back_populates="report")


class Consultation(Base):
    __tablename__ = "consultations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    screening_id = Column(Integer, ForeignKey("screenings.id"), nullable=False)
    message = Column(Text, nullable=False)
    doctor_response = Column(Text, nullable=True, default=None)
    status = Column(String(50), default="Pending")
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="consultations")
    screening = relationship("Screening", back_populates="consultation")


# ============================================================
# DATABASE INITIALIZATION & SESSION DEPENDENCY
# ============================================================

def init_db():
    """Create database tables and seed default doctor account if needed."""
    Base.metadata.create_all(bind=engine)

    # Safely migrate existing SQLite database without dropping tables
    with engine.connect() as conn:
        user_cols = [row[1] for row in conn.exec_driver_sql("PRAGMA table_info(users)").fetchall()]
        if "is_doctor" not in user_cols:
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN is_doctor BOOLEAN DEFAULT 0")
        if "doctor_license" not in user_cols:
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN doctor_license VARCHAR(100) DEFAULT NULL")
        if "doctor_specialty" not in user_cols:
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN doctor_specialty VARCHAR(150) DEFAULT NULL")

        screening_cols = [row[1] for row in conn.exec_driver_sql("PRAGMA table_info(screenings)").fetchall()]
        if "top2_json" not in screening_cols:
            conn.exec_driver_sql("ALTER TABLE screenings ADD COLUMN top2_json TEXT DEFAULT NULL")
        if "multi_nail_json" not in screening_cols:
            conn.exec_driver_sql("ALTER TABLE screenings ADD COLUMN multi_nail_json TEXT DEFAULT NULL")

        conn.commit()

    # Seed default doctor account if missing
    db = SessionLocal()
    try:
        doctor_email = "doctor@nailhealth.com"
        doc = db.query(User).filter(User.email == doctor_email).first()
        if not doc:
            doc = User(
                name="Dr. Evelyn Vance, MD, FAAD",
                email=doctor_email,
                password_hash=hash_password("Doctor123!"),
                is_doctor=True,
                doctor_license="#MED-482910-D",
                doctor_specialty="Board Certified Dermatologist & Nail Biology Specialist"
            )
            db.add(doc)
            db.commit()
            print("Default doctor account seeded: doctor@nailhealth.com / Doctor123!")
    except Exception as e:
        print(f"Doctor seeding note: {e}")
    finally:
        db.close()


def get_db():
    """FastAPI database session generator."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
