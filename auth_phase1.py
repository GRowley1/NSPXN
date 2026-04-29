import csv
import io
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List

import jwt
from jwt import ExpiredSignatureError, InvalidTokenError
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, create_engine, func, inspect, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# NSPXN AUTH PHASE 3C/3D
# - Database-backed users
# - Company/account controls
# - Monthly upload limits
# - Trial expiration
# - Usage logging
# - Usage dashboard
# - Admin activity audit log
# - JWT bearer token login
# - Helper for ASGI router auth checks
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nspxn_auth.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "CHANGE_ME_IN_RENDER")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "480"))

DEFAULT_MONTHLY_UPLOAD_LIMIT = int(os.getenv("NSPXN_DEFAULT_MONTHLY_UPLOAD_LIMIT", "100"))
DEFAULT_TRIAL_DAYS = int(os.getenv("NSPXN_DEFAULT_TRIAL_DAYS", "15"))

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

auth_app = FastAPI(title="NSPXN Auth Phase 2")
auth_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nspxn.com",
        "https://www.nspxn.com",
        "http://nspxn.com",
        "http://www.nspxn.com",
        "http://localhost",
        "http://localhost:3000",
        "http://127.0.0.1:5500",
    ],
    allow_origin_regex=r"https://.*\.nspxn\.com",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Company(Base):
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True)
    company_name = Column(String, unique=True, index=True, nullable=False)
    plan_name = Column(String, default="AI-4-IA", nullable=False)
    monthly_upload_limit = Column(Integer, default=DEFAULT_MONTHLY_UPLOAD_LIMIT, nullable=False)
    trial_start = Column(DateTime, nullable=True)
    trial_end = Column(DateTime, nullable=True)
    billing_status = Column(String, default="trial", nullable=False)  # trial, active, past_due, suspended
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class AuthUser(Base):
    __tablename__ = "auth_users"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=True)
    nspxn_id = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=True)
    password_hash = Column(String, nullable=False)
    # Kept for backward compatibility with Phase 1 records and token display.
    company_name = Column(String, nullable=True)
    role = Column(String, default="user", nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)


class UsageLog(Base):
    __tablename__ = "usage_log"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True)
    company_id = Column(Integer, nullable=True)
    nspxn_id = Column(String, nullable=True)
    company_name = Column(String, nullable=True)
    ai_intent = Column(String, nullable=True)
    file_number = Column(String, nullable=True)
    status = Column(String, default="completed", nullable=False)
    compliance_score = Column(Float, nullable=True)
    score_source = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class AdminActivityLog(Base):
    __tablename__ = "admin_activity_log"

    id = Column(Integer, primary_key=True, index=True)
    admin_user_id = Column(Integer, nullable=True)
    admin_nspxn_id = Column(String, nullable=True)
    action = Column(String, nullable=False)
    target_type = Column(String, nullable=True)
    target_id = Column(Integer, nullable=True)
    target_label = Column(String, nullable=True)
    details_json = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class LoginRequest(BaseModel):
    nspxn_id: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    nspxn_id: str
    email: Optional[str] = None
    company_id: Optional[int] = None
    company_name: Optional[str] = None
    role: str
    plan_name: Optional[str] = None
    billing_status: Optional[str] = None
    monthly_upload_limit: Optional[int] = None
    uploads_used_this_month: int = 0
    uploads_remaining_this_month: Optional[int] = None
    trial_end: Optional[str] = None
    account_status: Optional[str] = None


class CurrentUser(BaseModel):
    id: int
    nspxn_id: str
    email: Optional[str] = None
    company_id: Optional[int] = None
    company_name: Optional[str] = None
    role: str
    is_active: bool


# ============================================================
# DB INIT + LIGHTWEIGHT MIGRATION
# ============================================================

def _column_names(table_name: str) -> set:
    try:
        return {c["name"] for c in inspect(engine).get_columns(table_name)}
    except Exception:
        return set()


def _add_column_if_missing(table_name: str, column_name: str, ddl_type: str) -> None:
    cols = _column_names(table_name)
    if column_name in cols:
        return
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {ddl_type}"))


def _ensure_phase2_schema() -> None:
    """Keep Phase 2 deploy safe over an existing Phase 1 database.

    SQLAlchemy create_all() creates new tables, but it will not add missing
    columns to an existing auth_users table. This narrow migration adds the
    Phase 2 columns without requiring Alembic for this first account-controls step.
    """
    Base.metadata.create_all(bind=engine)

    if "auth_users" in inspect(engine).get_table_names():
        _add_column_if_missing("auth_users", "company_id", "INTEGER")

    if "usage_log" in inspect(engine).get_table_names():
        _add_column_if_missing("usage_log", "compliance_score", "FLOAT")
        _add_column_if_missing("usage_log", "score_source", "VARCHAR")

    # Backfill existing Phase 1 users into companies so current logins survive.
    db = SessionLocal()
    try:
        users = db.query(AuthUser).all()
        for user in users:
            if user.company_id:
                continue
            name = (user.company_name or "NSPXN").strip() or "NSPXN"
            company = db.query(Company).filter(Company.company_name == name).first()
            if not company:
                now = datetime.utcnow()
                company = Company(
                    company_name=name,
                    plan_name="AI-4-IA",
                    monthly_upload_limit=DEFAULT_MONTHLY_UPLOAD_LIMIT,
                    trial_start=now,
                    trial_end=now + timedelta(days=DEFAULT_TRIAL_DAYS),
                    billing_status="trial",
                    is_active=True,
                )
                db.add(company)
                db.flush()
            user.company_id = company.id
            user.company_name = company.company_name
        db.commit()
    finally:
        db.close()


def init_auth_db() -> None:
    _ensure_phase2_schema()


def get_auth_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================
# PASSWORD + TOKEN HELPERS
# ============================================================

def hash_password(password: str) -> str:
    if password and len(password.encode("utf-8")) > 72:
        raise ValueError("password cannot be longer than 72 bytes")
    return pwd_context.hash(password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    return pwd_context.verify(plain_password, password_hash)


def create_access_token(user: AuthUser) -> str:
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {
        "sub": user.nspxn_id,
        "user_id": user.id,
        "company_id": user.company_id,
        "role": user.role,
        "company_name": user.company_name,
        "iat": now,
        "exp": expire,
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Login expired. Please log in again.")
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid login token.")


# ============================================================
# LOOKUPS + ACCOUNT STATUS
# ============================================================

def get_user_by_nspxn_id(db: Session, nspxn_id: str) -> Optional[AuthUser]:
    clean_id = (nspxn_id or "").strip()
    if not clean_id:
        return None
    return db.query(AuthUser).filter(AuthUser.nspxn_id == clean_id).first()


def get_company_for_user(db: Session, user: AuthUser) -> Optional[Company]:
    if user.company_id:
        company = db.query(Company).filter(Company.id == user.company_id).first()
        if company:
            return company
    if user.company_name:
        return db.query(Company).filter(Company.company_name == user.company_name).first()
    return None


def _month_start_utc() -> datetime:
    now = datetime.utcnow()
    return datetime(now.year, now.month, 1)


def count_company_uploads_this_month(db: Session, company_id: Optional[int]) -> int:
    if not company_id:
        return 0
    return int(
        db.query(func.count(UsageLog.id))
        .filter(UsageLog.company_id == company_id)
        .filter(UsageLog.status == "completed")
        .filter(UsageLog.created_at >= _month_start_utc())
        .scalar()
        or 0
    )


def count_user_uploads_this_month(db: Session, user_id: Optional[int]) -> int:
    if not user_id:
        return 0
    return int(
        db.query(func.count(UsageLog.id))
        .filter(UsageLog.user_id == user_id)
        .filter(UsageLog.status == "completed")
        .filter(UsageLog.created_at >= _month_start_utc())
        .scalar()
        or 0
    )


def _account_status_payload(db: Session, user: AuthUser) -> dict:
    company = get_company_for_user(db, user)
    used = count_company_uploads_this_month(db, company.id) if company else 0
    limit = company.monthly_upload_limit if company else None
    remaining = None if limit is None else max(int(limit) - used, 0)

    return {
        "company_id": company.id if company else None,
        "company_name": company.company_name if company else user.company_name,
        "plan_name": company.plan_name if company else None,
        "billing_status": company.billing_status if company else None,
        "monthly_upload_limit": limit,
        "uploads_used_this_month": used,
        "uploads_remaining_this_month": remaining,
        "trial_end": company.trial_end.isoformat() if company and company.trial_end else None,
        "account_status": "active" if company and company.is_active else "unknown",
    }


def enforce_account_for_upload(current_user: CurrentUser) -> dict:
    """Return account status or raise HTTPException with clean blocked details."""
    db = SessionLocal()
    try:
        user = db.query(AuthUser).filter(AuthUser.id == current_user.id).first()
        if not user or not user.is_active:
            raise HTTPException(status_code=403, detail={
                "status": "blocked",
                "error": "User account inactive.",
                "reasons": ["This NSPXN user account is inactive or no longer exists."],
            })

        company = get_company_for_user(db, user)
        if not company:
            raise HTTPException(status_code=403, detail={
                "status": "blocked",
                "error": "Company account missing.",
                "reasons": ["This NSPXN user is not assigned to an active company account."],
            })

        if not company.is_active:
            raise HTTPException(status_code=403, detail={
                "status": "blocked",
                "error": "Company account inactive.",
                "reasons": ["This NSPXN company account is inactive."],
            })

        now = datetime.utcnow()
        if company.billing_status == "trial" and company.trial_end and now > company.trial_end:
            raise HTTPException(status_code=403, detail={
                "status": "blocked",
                "error": "Account trial has expired.",
                "reasons": ["Your NSPXN trial period has ended. Please contact NSPXN to activate billing."],
            })

        if company.billing_status in {"past_due", "suspended", "cancelled", "canceled"}:
            raise HTTPException(status_code=403, detail={
                "status": "blocked",
                "error": "Company billing status is not active.",
                "reasons": [f"This company account billing status is {company.billing_status}."],
            })

        used = count_company_uploads_this_month(db, company.id)
        limit = int(company.monthly_upload_limit or 0)
        if limit >= 0 and used >= limit:
            raise HTTPException(status_code=403, detail={
                "status": "blocked",
                "error": "Monthly upload limit reached.",
                "reasons": [f"This account has used {used} of {limit} monthly uploads."],
            })

        payload = _account_status_payload(db, user)
        payload["account_status"] = "active"
        return payload
    finally:
        db.close()


def log_usage_event(
    current_user: CurrentUser,
    ai_intent: Optional[str],
    file_number: Optional[str],
    status: str = "completed",
    compliance_score: Optional[float] = None,
    score_source: Optional[str] = None,
) -> None:
    db = SessionLocal()
    try:
        user = db.query(AuthUser).filter(AuthUser.id == current_user.id).first()
        company = get_company_for_user(db, user) if user else None
        entry = UsageLog(
            user_id=current_user.id,
            company_id=company.id if company else current_user.company_id,
            nspxn_id=current_user.nspxn_id,
            company_name=company.company_name if company else current_user.company_name,
            ai_intent=ai_intent,
            file_number=file_number,
            status=status or "completed",
            compliance_score=compliance_score,
            score_source=score_source,
        )
        db.add(entry)
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"WARNING: usage logging failed: {exc}")
    finally:
        db.close()


def authenticate_user(db: Session, nspxn_id: str, password: str) -> Optional[AuthUser]:
    user = get_user_by_nspxn_id(db, nspxn_id)
    if not user:
        return None
    if not user.is_active:
        raise HTTPException(status_code=403, detail="This NSPXN account is inactive.")
    if not verify_password(password, user.password_hash):
        return None
    company = get_company_for_user(db, user)
    if company and not company.is_active:
        raise HTTPException(status_code=403, detail="This NSPXN company account is inactive.")
    user.last_login = datetime.utcnow()
    db.commit()
    return user


def _to_current_user(user: AuthUser) -> CurrentUser:
    return CurrentUser(
        id=user.id,
        nspxn_id=user.nspxn_id,
        email=user.email,
        company_id=user.company_id,
        company_name=user.company_name,
        role=user.role,
        is_active=user.is_active,
    )


def get_current_auth_user(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_auth_db),
) -> CurrentUser:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing login token.")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header.")

    token = authorization.split(" ", 1)[1].strip()
    payload = decode_access_token(token)
    nspxn_id = payload.get("sub")
    if not nspxn_id:
        raise HTTPException(status_code=401, detail="Invalid login token.")

    user = get_user_by_nspxn_id(db, nspxn_id)
    if not user:
        raise HTTPException(status_code=401, detail="User account no longer exists.")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="This NSPXN account is inactive.")

    return _to_current_user(user)


def require_auth_from_authorization_header(authorization: Optional[str]) -> CurrentUser:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing login token.")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header.")

    token = authorization.split(" ", 1)[1].strip()
    payload = decode_access_token(token)
    nspxn_id = payload.get("sub")
    if not nspxn_id:
        raise HTTPException(status_code=401, detail="Invalid login token.")

    db = SessionLocal()
    try:
        user = get_user_by_nspxn_id(db, nspxn_id)
        if not user:
            raise HTTPException(status_code=401, detail="User account no longer exists.")
        if not user.is_active:
            raise HTTPException(status_code=403, detail="This NSPXN account is inactive.")
        return _to_current_user(user)
    finally:
        db.close()


# ============================================================
# CREATION HELPERS USED BY create_nspxn_user.py
# ============================================================

def get_or_create_company(
    db: Session,
    company_name: str,
    plan_name: str = "AI-4-IA",
    monthly_upload_limit: int = DEFAULT_MONTHLY_UPLOAD_LIMIT,
    trial_days: int = DEFAULT_TRIAL_DAYS,
    billing_status: str = "trial",
    is_active: bool = True,
) -> Company:
    clean_name = (company_name or "NSPXN").strip() or "NSPXN"
    company = db.query(Company).filter(Company.company_name == clean_name).first()
    if company:
        return company

    now = datetime.utcnow()
    trial_start = now if billing_status == "trial" else None
    trial_end = (now + timedelta(days=int(trial_days or 0))) if billing_status == "trial" and int(trial_days or 0) > 0 else None
    company = Company(
        company_name=clean_name,
        plan_name=plan_name or "AI-4-IA",
        monthly_upload_limit=int(monthly_upload_limit or DEFAULT_MONTHLY_UPLOAD_LIMIT),
        trial_start=trial_start,
        trial_end=trial_end,
        billing_status=billing_status or "trial",
        is_active=is_active,
    )
    db.add(company)
    db.commit()
    db.refresh(company)
    return company


def create_auth_user(
    db: Session,
    nspxn_id: str,
    password: str,
    email: Optional[str] = None,
    company_name: Optional[str] = None,
    role: str = "user",
    is_active: bool = True,
    plan_name: str = "AI-4-IA",
    monthly_upload_limit: int = DEFAULT_MONTHLY_UPLOAD_LIMIT,
    trial_days: int = DEFAULT_TRIAL_DAYS,
    billing_status: str = "trial",
) -> AuthUser:
    clean_id = (nspxn_id or "").strip()
    if not clean_id:
        raise ValueError("nspxn_id is required")
    if not password:
        raise ValueError("password is required")
    if len(password.encode("utf-8")) > 72:
        raise ValueError("password cannot be longer than 72 bytes")

    existing = get_user_by_nspxn_id(db, clean_id)
    if existing:
        raise ValueError(f"NSPXN ID already exists: {clean_id}")

    company = get_or_create_company(
        db=db,
        company_name=company_name or "NSPXN",
        plan_name=plan_name,
        monthly_upload_limit=monthly_upload_limit,
        trial_days=trial_days,
        billing_status=billing_status,
        is_active=True,
    )

    user = AuthUser(
        company_id=company.id,
        nspxn_id=clean_id,
        email=email or None,
        password_hash=hash_password(password),
        company_name=company.company_name,
        role=role or "user",
        is_active=is_active,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# ============================================================
# ROUTES
# ============================================================

@auth_app.on_event("startup")
def _startup_auth_db() -> None:
    init_auth_db()


@auth_app.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, db: Session = Depends(get_auth_db)):
    user = authenticate_user(db=db, nspxn_id=payload.nspxn_id, password=payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid NSPXN ID # or Password.")

    token = create_access_token(user)
    account = _account_status_payload(db, user)
    return LoginResponse(
        access_token=token,
        nspxn_id=user.nspxn_id,
        email=user.email,
        company_id=account.get("company_id"),
        company_name=account.get("company_name"),
        role=user.role,
        plan_name=account.get("plan_name"),
        billing_status=account.get("billing_status"),
        monthly_upload_limit=account.get("monthly_upload_limit"),
        uploads_used_this_month=account.get("uploads_used_this_month") or 0,
        uploads_remaining_this_month=account.get("uploads_remaining_this_month"),
        trial_end=account.get("trial_end"),
        account_status=account.get("account_status"),
    )


@auth_app.get("/me")
def me(current_user: CurrentUser = Depends(get_current_auth_user)):
    db = SessionLocal()
    try:
        user = db.query(AuthUser).filter(AuthUser.id == current_user.id).first()
        account = _account_status_payload(db, user) if user else {}
    finally:
        db.close()

    return {
        "nspxn_id": current_user.nspxn_id,
        "email": current_user.email,
        "company_id": account.get("company_id"),
        "company_name": account.get("company_name"),
        "role": current_user.role,
        "is_active": current_user.is_active,
        "plan_name": account.get("plan_name"),
        "billing_status": account.get("billing_status"),
        "monthly_upload_limit": account.get("monthly_upload_limit"),
        "uploads_used_this_month": account.get("uploads_used_this_month") or 0,
        "uploads_remaining_this_month": account.get("uploads_remaining_this_month"),
        "trial_end": account.get("trial_end"),
        "account_status": account.get("account_status"),
    }


# ============================================================
# PHASE 3A - ADMIN BACKEND ENDPOINTS
# Admin-only account management. Requires role="admin".
# ============================================================

class AdminCompanyCreateRequest(BaseModel):
    company_name: str
    plan_name: str = "AI-4-IA"
    monthly_upload_limit: int = DEFAULT_MONTHLY_UPLOAD_LIMIT
    trial_days: int = DEFAULT_TRIAL_DAYS
    billing_status: str = "trial"
    is_active: bool = True


class AdminCompanyUpdateRequest(BaseModel):
    company_name: Optional[str] = None
    plan_name: Optional[str] = None
    monthly_upload_limit: Optional[int] = None
    billing_status: Optional[str] = None
    is_active: Optional[bool] = None
    trial_days_from_now: Optional[int] = None
    clear_trial_dates: Optional[bool] = None


class AdminUserCreateRequest(BaseModel):
    nspxn_id: str
    password: str
    email: Optional[str] = None
    company_id: Optional[int] = None
    company_name: Optional[str] = None
    role: str = "user"
    is_active: bool = True
    plan_name: str = "AI-4-IA"
    monthly_upload_limit: int = DEFAULT_MONTHLY_UPLOAD_LIMIT
    trial_days: int = DEFAULT_TRIAL_DAYS
    billing_status: str = "trial"


class AdminUserUpdateRequest(BaseModel):
    email: Optional[str] = None
    company_id: Optional[int] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


class AdminPasswordResetRequest(BaseModel):
    password: str


class AdminComplianceScoreUpdateRequest(BaseModel):
    compliance_score: Optional[float] = None
    clear_score: bool = False
    reason: Optional[str] = None


def require_admin_user(current_user: CurrentUser = Depends(get_current_auth_user)) -> CurrentUser:
    if (current_user.role or "").lower() != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return current_user


def _company_to_dict(db: Session, company: Company) -> dict:
    used = count_company_uploads_this_month(db, company.id)
    limit = int(company.monthly_upload_limit or 0)
    monthly_score_stats = _comprehensive_score_stats_for_company(db, company.id, month_only=True)
    lifetime_score_stats = _comprehensive_score_stats_for_company(db, company.id, month_only=False)
    return {
        "id": company.id,
        "company_name": company.company_name,
        "plan_name": company.plan_name,
        "monthly_upload_limit": company.monthly_upload_limit,
        "uploads_used_this_month": used,
        "uploads_remaining_this_month": max(limit - used, 0),
        "avg_comprehensive_compliance_score_this_month": monthly_score_stats["avg"],
        "comprehensive_score_count_this_month": monthly_score_stats["count"],
        "avg_comprehensive_compliance_score_lifetime": lifetime_score_stats["avg"],
        "comprehensive_score_count_lifetime": lifetime_score_stats["count"],
        "trial_start": company.trial_start.isoformat() if company.trial_start else None,
        "trial_end": company.trial_end.isoformat() if company.trial_end else None,
        "billing_status": company.billing_status,
        "is_active": company.is_active,
        "created_at": company.created_at.isoformat() if company.created_at else None,
    }


def _comprehensive_score_stats_for_user(db: Session, user_id: Optional[int], month_only: bool = False) -> dict:
    if not user_id:
        return {"count": 0, "avg": None, "low": None, "high": None}
    query = (
        db.query(UsageLog.compliance_score)
        .filter(UsageLog.user_id == int(user_id))
        .filter(UsageLog.status == "completed")
        .filter(UsageLog.ai_intent == "comprehensive")
        .filter(UsageLog.compliance_score.isnot(None))
    )
    if month_only:
        query = query.filter(UsageLog.created_at >= _month_start_utc())
    scores = [float(x[0]) for x in query.all() if x and x[0] is not None]
    if not scores:
        return {"count": 0, "avg": None, "low": None, "high": None}
    return {
        "count": len(scores),
        "avg": round(sum(scores) / len(scores), 1),
        "low": round(min(scores), 1),
        "high": round(max(scores), 1),
    }


def _comprehensive_score_stats_for_company(db: Session, company_id: Optional[int], month_only: bool = False) -> dict:
    if not company_id:
        return {"count": 0, "avg": None, "low": None, "high": None}
    query = (
        db.query(UsageLog.compliance_score)
        .filter(UsageLog.company_id == int(company_id))
        .filter(UsageLog.status == "completed")
        .filter(UsageLog.ai_intent == "comprehensive")
        .filter(UsageLog.compliance_score.isnot(None))
    )
    if month_only:
        query = query.filter(UsageLog.created_at >= _month_start_utc())
    scores = [float(x[0]) for x in query.all() if x and x[0] is not None]
    if not scores:
        return {"count": 0, "avg": None, "low": None, "high": None}
    return {
        "count": len(scores),
        "avg": round(sum(scores) / len(scores), 1),
        "low": round(min(scores), 1),
        "high": round(max(scores), 1),
    }


def _user_to_dict(db: Session, user: AuthUser) -> dict:
    company = get_company_for_user(db, user)
    user_used = count_user_uploads_this_month(db, user.id)
    company_used = count_company_uploads_this_month(db, company.id) if company else 0
    company_limit = int(company.monthly_upload_limit or 0) if company else 0
    monthly_score_stats = _comprehensive_score_stats_for_user(db, user.id, month_only=True)
    lifetime_score_stats = _comprehensive_score_stats_for_user(db, user.id, month_only=False)
    return {
        "id": user.id,
        "nspxn_id": user.nspxn_id,
        "email": user.email,
        "company_id": company.id if company else user.company_id,
        "company_name": company.company_name if company else user.company_name,
        "role": user.role,
        "is_active": user.is_active,
        "uploads_used_this_month": user_used,
        "company_uploads_used_this_month": company_used,
        "company_monthly_upload_limit": company_limit,
        "company_uploads_remaining_this_month": max(company_limit - company_used, 0) if company else 0,
        "avg_comprehensive_compliance_score_this_month": monthly_score_stats["avg"],
        "comprehensive_score_count_this_month": monthly_score_stats["count"],
        "lowest_comprehensive_compliance_score_this_month": monthly_score_stats["low"],
        "highest_comprehensive_compliance_score_this_month": monthly_score_stats["high"],
        "avg_comprehensive_compliance_score_lifetime": lifetime_score_stats["avg"],
        "comprehensive_score_count_lifetime": lifetime_score_stats["count"],
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "last_login": user.last_login.isoformat() if user.last_login else None,
    }


def _usage_to_dict(entry: UsageLog) -> dict:
    return {
        "id": entry.id,
        "user_id": entry.user_id,
        "company_id": entry.company_id,
        "nspxn_id": entry.nspxn_id,
        "company_name": entry.company_name,
        "ai_intent": entry.ai_intent,
        "file_number": entry.file_number,
        "status": entry.status,
        "compliance_score": entry.compliance_score,
        "score_source": entry.score_source,
        "created_at": entry.created_at.isoformat() if entry.created_at else None,
    }


def _activity_to_dict(entry: AdminActivityLog) -> dict:
    details = None
    if entry.details_json:
        try:
            details = json.loads(entry.details_json)
        except Exception:
            details = entry.details_json
    return {
        "id": entry.id,
        "admin_user_id": entry.admin_user_id,
        "admin_nspxn_id": entry.admin_nspxn_id,
        "action": entry.action,
        "target_type": entry.target_type,
        "target_id": entry.target_id,
        "target_label": entry.target_label,
        "details": details,
        "created_at": entry.created_at.isoformat() if entry.created_at else None,
    }


def _log_admin_activity(
    db: Session,
    admin_user: CurrentUser,
    action: str,
    target_type: Optional[str] = None,
    target_id: Optional[int] = None,
    target_label: Optional[str] = None,
    details: Optional[dict] = None,
) -> None:
    try:
        entry = AdminActivityLog(
            admin_user_id=admin_user.id if admin_user else None,
            admin_nspxn_id=admin_user.nspxn_id if admin_user else None,
            action=(action or "unknown"),
            target_type=target_type,
            target_id=target_id,
            target_label=target_label,
            details_json=json.dumps(details or {}, default=str),
        )
        db.add(entry)
    except Exception as exc:
        print(f"WARNING: admin activity logging failed: {exc}")


def _parse_admin_date(value: Optional[str], end_of_day: bool = False) -> Optional[datetime]:
    if not value:
        return None
    clean = str(value).strip()
    if not clean:
        return None
    try:
        if len(clean) == 10:
            dt = datetime.strptime(clean, "%Y-%m-%d")
            return dt + timedelta(days=1) if end_of_day else dt
        return datetime.fromisoformat(clean.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")


def _filtered_usage_query(
    db: Session,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    company_id: Optional[int] = None,
    user_id: Optional[int] = None,
    status: Optional[str] = None,
    ai_intent: Optional[str] = None,
):
    start = _parse_admin_date(start_date) or _month_start_utc()
    end = _parse_admin_date(end_date, end_of_day=True)

    query = db.query(UsageLog).filter(UsageLog.created_at >= start)
    if end:
        query = query.filter(UsageLog.created_at < end)
    if company_id:
        query = query.filter(UsageLog.company_id == int(company_id))
    if user_id:
        query = query.filter(UsageLog.user_id == int(user_id))
    if status and status != "all":
        query = query.filter(UsageLog.status == status)
    if ai_intent and ai_intent != "all":
        query = query.filter(UsageLog.ai_intent == ai_intent)
    return query, start, end


@auth_app.get("/admin/summary")
def admin_summary(
    admin_user: CurrentUser = Depends(require_admin_user),
    db: Session = Depends(get_auth_db),
):
    companies_count = int(db.query(func.count(Company.id)).scalar() or 0)
    users_count = int(db.query(func.count(AuthUser.id)).scalar() or 0)
    active_users_count = int(db.query(func.count(AuthUser.id)).filter(AuthUser.is_active == True).scalar() or 0)  # noqa: E712
    usage_this_month = int(
        db.query(func.count(UsageLog.id))
        .filter(UsageLog.status == "completed")
        .filter(UsageLog.created_at >= _month_start_utc())
        .scalar()
        or 0
    )
    return {
        "companies": companies_count,
        "users": users_count,
        "active_users": active_users_count,
        "uploads_this_month": usage_this_month,
    }


@auth_app.get("/admin/companies")
def admin_list_companies(
    q: Optional[str] = None,
    admin_user: CurrentUser = Depends(require_admin_user),
    db: Session = Depends(get_auth_db),
):
    query = db.query(Company)
    if q:
        query = query.filter(Company.company_name.ilike(f"%{q.strip()}%"))
    companies = query.order_by(Company.company_name.asc()).all()
    return {"companies": [_company_to_dict(db, c) for c in companies]}


@auth_app.post("/admin/companies")
def admin_create_company(
    payload: AdminCompanyCreateRequest,
    admin_user: CurrentUser = Depends(require_admin_user),
    db: Session = Depends(get_auth_db),
):
    existing = db.query(Company).filter(Company.company_name == payload.company_name.strip()).first()
    if existing:
        raise HTTPException(status_code=400, detail="Company already exists.")
    company = get_or_create_company(
        db=db,
        company_name=payload.company_name,
        plan_name=payload.plan_name,
        monthly_upload_limit=payload.monthly_upload_limit,
        trial_days=payload.trial_days,
        billing_status=payload.billing_status,
        is_active=payload.is_active,
    )
    company.is_active = payload.is_active
    _log_admin_activity(
        db, admin_user, "company_created", "company", company.id, company.company_name,
        {"plan_name": company.plan_name, "monthly_upload_limit": company.monthly_upload_limit, "billing_status": company.billing_status, "is_active": company.is_active},
    )
    db.commit()
    db.refresh(company)
    return {"company": _company_to_dict(db, company)}


@auth_app.patch("/admin/companies/{company_id}")
def admin_update_company(
    company_id: int,
    payload: AdminCompanyUpdateRequest,
    admin_user: CurrentUser = Depends(require_admin_user),
    db: Session = Depends(get_auth_db),
):
    company = db.query(Company).filter(Company.id == company_id).first()
    if not company:
        raise HTTPException(status_code=404, detail="Company not found.")

    if payload.company_name is not None:
        new_name = payload.company_name.strip()
        if not new_name:
            raise HTTPException(status_code=400, detail="Company name cannot be blank.")
        duplicate = db.query(Company).filter(Company.company_name == new_name, Company.id != company_id).first()
        if duplicate:
            raise HTTPException(status_code=400, detail="Another company already uses that name.")
        company.company_name = new_name

    if payload.plan_name is not None:
        company.plan_name = payload.plan_name.strip() or company.plan_name
    if payload.monthly_upload_limit is not None:
        company.monthly_upload_limit = int(payload.monthly_upload_limit)
    if payload.billing_status is not None:
        company.billing_status = payload.billing_status.strip() or company.billing_status
    if payload.is_active is not None:
        company.is_active = bool(payload.is_active)
    if payload.clear_trial_dates:
        company.trial_start = None
        company.trial_end = None
    elif payload.trial_days_from_now is not None:
        days = int(payload.trial_days_from_now)
        now = datetime.utcnow()
        company.trial_start = now
        company.trial_end = now + timedelta(days=days) if days > 0 else now

    db.query(AuthUser).filter(AuthUser.company_id == company.id).update({AuthUser.company_name: company.company_name})

    _log_admin_activity(
        db, admin_user, "company_updated", "company", company.id, company.company_name,
        {"company_name": company.company_name, "plan_name": company.plan_name, "monthly_upload_limit": company.monthly_upload_limit, "billing_status": company.billing_status, "is_active": company.is_active, "trial_end": company.trial_end.isoformat() if company.trial_end else None},
    )
    db.commit()
    db.refresh(company)
    return {"company": _company_to_dict(db, company)}


@auth_app.get("/admin/users")
def admin_list_users(
    q: Optional[str] = None,
    company_id: Optional[int] = None,
    admin_user: CurrentUser = Depends(require_admin_user),
    db: Session = Depends(get_auth_db),
):
    query = db.query(AuthUser)
    if q:
        like = f"%{q.strip()}%"
        query = query.filter((AuthUser.nspxn_id.ilike(like)) | (AuthUser.email.ilike(like)) | (AuthUser.company_name.ilike(like)))
    if company_id:
        query = query.filter(AuthUser.company_id == company_id)
    users = query.order_by(AuthUser.id.asc()).all()
    return {"users": [_user_to_dict(db, u) for u in users]}


@auth_app.post("/admin/users")
def admin_create_user(
    payload: AdminUserCreateRequest,
    admin_user: CurrentUser = Depends(require_admin_user),
    db: Session = Depends(get_auth_db),
):
    if not payload.password or len(payload.password.encode("utf-8")) > 72:
        raise HTTPException(status_code=400, detail="Password is required and must be 72 bytes or shorter.")

    company = None
    if payload.company_id:
        company = db.query(Company).filter(Company.id == payload.company_id).first()
        if not company:
            raise HTTPException(status_code=404, detail="Company not found.")

    user = create_auth_user(
        db=db,
        nspxn_id=payload.nspxn_id,
        password=payload.password,
        email=payload.email,
        company_name=(company.company_name if company else payload.company_name),
        role=payload.role,
        is_active=payload.is_active,
        plan_name=payload.plan_name,
        monthly_upload_limit=payload.monthly_upload_limit,
        trial_days=payload.trial_days,
        billing_status=payload.billing_status,
    )

    if company and user.company_id != company.id:
        user.company_id = company.id
        user.company_name = company.company_name
        db.commit()
        db.refresh(user)

    _log_admin_activity(
        db, admin_user, "user_created", "user", user.id, user.nspxn_id,
        {"email": user.email, "company_id": user.company_id, "company_name": user.company_name, "role": user.role, "is_active": user.is_active},
    )
    db.commit()
    return {"user": _user_to_dict(db, user)}


@auth_app.patch("/admin/users/{user_id}")
def admin_update_user(
    user_id: int,
    payload: AdminUserUpdateRequest,
    admin_user: CurrentUser = Depends(require_admin_user),
    db: Session = Depends(get_auth_db),
):
    user = db.query(AuthUser).filter(AuthUser.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    if payload.email is not None:
        user.email = payload.email.strip() or None
    if payload.company_id is not None:
        company = db.query(Company).filter(Company.id == int(payload.company_id)).first()
        if not company:
            raise HTTPException(status_code=404, detail="Company not found.")
        user.company_id = company.id
        user.company_name = company.company_name
    if payload.role is not None:
        role = payload.role.strip().lower() or "user"
        if role not in {"user", "admin"}:
            raise HTTPException(status_code=400, detail="Role must be user or admin.")
        user.role = role
    if payload.is_active is not None:
        user.is_active = bool(payload.is_active)

    _log_admin_activity(
        db, admin_user, "user_updated", "user", user.id, user.nspxn_id,
        {"email": user.email, "company_id": user.company_id, "company_name": user.company_name, "role": user.role, "is_active": user.is_active},
    )
    db.commit()
    db.refresh(user)
    return {"user": _user_to_dict(db, user)}


@auth_app.post("/admin/users/{user_id}/reset-password")
def admin_reset_user_password(
    user_id: int,
    payload: AdminPasswordResetRequest,
    admin_user: CurrentUser = Depends(require_admin_user),
    db: Session = Depends(get_auth_db),
):
    user = db.query(AuthUser).filter(AuthUser.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    if not payload.password or len(payload.password.encode("utf-8")) > 72:
        raise HTTPException(status_code=400, detail="Password is required and must be 72 bytes or shorter.")
    user.password_hash = hash_password(payload.password)
    _log_admin_activity(db, admin_user, "password_reset", "user", user.id, user.nspxn_id, {})
    db.commit()
    return {"ok": True, "message": f"Password reset for {user.nspxn_id}."}


@auth_app.get("/admin/usage")
def admin_usage(
    company_id: Optional[int] = None,
    user_id: Optional[int] = None,
    limit: int = 100,
    admin_user: CurrentUser = Depends(require_admin_user),
    db: Session = Depends(get_auth_db),
):
    query = db.query(UsageLog)
    if company_id:
        query = query.filter(UsageLog.company_id == company_id)
    if user_id:
        query = query.filter(UsageLog.user_id == user_id)
    rows = query.order_by(UsageLog.created_at.desc()).limit(max(1, min(int(limit or 100), 500))).all()
    return {"usage": [_usage_to_dict(r) for r in rows]}

@auth_app.post("/admin/usage/{usage_id}/score")
def admin_update_usage_compliance_score(
    usage_id: int,
    payload: AdminComplianceScoreUpdateRequest,
    admin_user: CurrentUser = Depends(require_admin_user),
    db: Session = Depends(get_auth_db),
):
    """Admin correction for a single report compliance score.

    This preserves the usage row and records old/new values in the
    admin activity log. Compliance scores should only be assigned to
    completed Comprehensive usage rows.
    """
    row = db.query(UsageLog).filter(UsageLog.id == int(usage_id)).first()
    if not row:
        raise HTTPException(status_code=404, detail="Usage row not found.")

    old_score = row.compliance_score
    old_source = row.score_source
    reason = (payload.reason or "").strip()

    if payload.clear_score:
        row.compliance_score = None
        row.score_source = "manual_admin_cleared"
        action_message = f"Compliance score cleared for usage row {row.id}."
    else:
        if payload.compliance_score is None:
            raise HTTPException(status_code=400, detail="Enter a compliance score or choose clear_score=true.")
        try:
            score = float(payload.compliance_score)
        except Exception:
            raise HTTPException(status_code=400, detail="Compliance score must be numeric.")
        if score < 0 or score > 100:
            raise HTTPException(status_code=400, detail="Compliance score must be between 0 and 100.")
        if (row.ai_intent or "").lower() != "comprehensive":
            raise HTTPException(status_code=400, detail="Compliance scores should only be set on Comprehensive usage rows.")
        if (row.status or "").lower() != "completed":
            raise HTTPException(status_code=400, detail="Compliance scores should only be set on completed usage rows.")
        row.compliance_score = round(score, 1)
        row.score_source = "manual_admin_correction"
        action_message = f"Compliance score updated for usage row {row.id}."

    _log_admin_activity(
        db,
        admin_user,
        "compliance_score_corrected",
        "usage",
        row.id,
        row.file_number or row.nspxn_id or f"usage #{row.id}",
        {
            "usage_id": row.id,
            "file_number": row.file_number,
            "nspxn_id": row.nspxn_id,
            "company_name": row.company_name,
            "ai_intent": row.ai_intent,
            "status": row.status,
            "old_score": old_score,
            "new_score": row.compliance_score,
            "old_score_source": old_source,
            "new_score_source": row.score_source,
            "reason": reason,
        },
    )
    db.commit()
    db.refresh(row)
    return {"ok": True, "message": action_message, "usage": _usage_to_dict(row)}


@auth_app.post("/admin/usage/reset")
def admin_reset_current_month_usage(
    company_id: Optional[int] = None,
    user_id: Optional[int] = None,
    reset_all: bool = False,
    admin_user: CurrentUser = Depends(require_admin_user),
    db: Session = Depends(get_auth_db),
):
    """Admin current-month usage reset.

    This does not delete historical rows. It marks matching current-month
    completed usage rows as reset_by_admin so monthly counters immediately
    drop while preserving an audit trail.
    """
    if not reset_all and not company_id and not user_id:
        raise HTTPException(status_code=400, detail="Provide company_id, user_id, or reset_all=true.")

    if reset_all and (company_id or user_id):
        raise HTTPException(status_code=400, detail="Use reset_all by itself, or reset one company/user.")

    month_start = _month_start_utc()
    query = (
        db.query(UsageLog)
        .filter(UsageLog.status == "completed")
        .filter(UsageLog.created_at >= month_start)
    )

    target_label = "all accounts"

    if user_id:
        user = db.query(AuthUser).filter(AuthUser.id == int(user_id)).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found.")
        query = query.filter(UsageLog.user_id == int(user_id))
        target_label = f"user {user.nspxn_id}"

    if company_id:
        company = db.query(Company).filter(Company.id == int(company_id)).first()
        if not company:
            raise HTTPException(status_code=404, detail="Company not found.")
        query = query.filter(UsageLog.company_id == int(company_id))
        target_label = f"company {company.company_name}"

    rows = query.all()
    reset_count = len(rows)

    for row in rows:
        row.status = "reset_by_admin"

    reset_target_type = "all" if reset_all else ("user" if user_id else "company")
    reset_target_id = int(user_id) if user_id else (int(company_id) if company_id else None)
    _log_admin_activity(
        db, admin_user, "usage_reset", reset_target_type, reset_target_id, target_label,
        {"reset_count": reset_count, "reset_all": reset_all, "company_id": company_id, "user_id": user_id},
    )
    db.commit()

    return {
        "ok": True,
        "reset_count": reset_count,
        "message": f"Reset {reset_count} current-month completed usage record(s) for {target_label}.",
    }


@auth_app.get("/admin/usage/dashboard")
def admin_usage_dashboard(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    company_id: Optional[int] = None,
    user_id: Optional[int] = None,
    status: Optional[str] = None,
    ai_intent: Optional[str] = None,
    admin_user: CurrentUser = Depends(require_admin_user),
    db: Session = Depends(get_auth_db),
):
    query, start, end = _filtered_usage_query(db, start_date, end_date, company_id, user_id, status, ai_intent)
    rows = query.all()

    status_counts = {}
    intent_counts = {}
    company_counts = {}
    user_counts = {}

    for row in rows:
        st = row.status or "unknown"
        intent = row.ai_intent or "unknown"
        status_counts[st] = status_counts.get(st, 0) + 1
        intent_counts[intent] = intent_counts.get(intent, 0) + 1

        is_scored_comprehensive = (st == "completed" and intent == "comprehensive" and row.compliance_score is not None)

        if row.company_id:
            company_counts.setdefault(row.company_id, {"company_id": row.company_id, "company_name": row.company_name or "", "total_records": 0, "completed_uploads": 0, "reset_records": 0, "score_sum": 0.0, "score_count": 0, "score_low": None, "score_high": None})
            company_counts[row.company_id]["total_records"] += 1
            if st == "completed":
                company_counts[row.company_id]["completed_uploads"] += 1
            if st == "reset_by_admin":
                company_counts[row.company_id]["reset_records"] += 1
            if is_scored_comprehensive:
                score = float(row.compliance_score)
                company_counts[row.company_id]["score_sum"] += score
                company_counts[row.company_id]["score_count"] += 1
                company_counts[row.company_id]["score_low"] = score if company_counts[row.company_id]["score_low"] is None else min(company_counts[row.company_id]["score_low"], score)
                company_counts[row.company_id]["score_high"] = score if company_counts[row.company_id]["score_high"] is None else max(company_counts[row.company_id]["score_high"], score)

        if row.user_id:
            user_counts.setdefault(row.user_id, {"user_id": row.user_id, "nspxn_id": row.nspxn_id or "", "company_name": row.company_name or "", "total_records": 0, "completed_uploads": 0, "reset_records": 0, "score_sum": 0.0, "score_count": 0, "score_low": None, "score_high": None})
            user_counts[row.user_id]["total_records"] += 1
            if st == "completed":
                user_counts[row.user_id]["completed_uploads"] += 1
            if st == "reset_by_admin":
                user_counts[row.user_id]["reset_records"] += 1
            if is_scored_comprehensive:
                score = float(row.compliance_score)
                user_counts[row.user_id]["score_sum"] += score
                user_counts[row.user_id]["score_count"] += 1
                user_counts[row.user_id]["score_low"] = score if user_counts[row.user_id]["score_low"] is None else min(user_counts[row.user_id]["score_low"], score)
                user_counts[row.user_id]["score_high"] = score if user_counts[row.user_id]["score_high"] is None else max(user_counts[row.user_id]["score_high"], score)

    for cid, item in list(company_counts.items()):
        company = db.query(Company).filter(Company.id == cid).first()
        if company:
            item["company_name"] = company.company_name
            item["plan_name"] = company.plan_name
            item["monthly_upload_limit"] = company.monthly_upload_limit
            item["uploads_remaining"] = max(int(company.monthly_upload_limit or 0) - int(item["completed_uploads"] or 0), 0)
            item["usage_percent"] = round((int(item["completed_uploads"] or 0) / int(company.monthly_upload_limit or 1)) * 100, 1) if int(company.monthly_upload_limit or 0) > 0 else 0
            item["billing_status"] = company.billing_status
            item["trial_end"] = company.trial_end.isoformat() if company.trial_end else None
        if int(item.get("score_count") or 0) > 0:
            item["avg_comprehensive_compliance_score"] = round(float(item.get("score_sum") or 0) / int(item.get("score_count") or 1), 1)
            item["lowest_comprehensive_compliance_score"] = round(float(item.get("score_low")), 1) if item.get("score_low") is not None else None
            item["highest_comprehensive_compliance_score"] = round(float(item.get("score_high")), 1) if item.get("score_high") is not None else None
        else:
            item["avg_comprehensive_compliance_score"] = None
            item["lowest_comprehensive_compliance_score"] = None
            item["highest_comprehensive_compliance_score"] = None

    for uid, item in list(user_counts.items()):
        user = db.query(AuthUser).filter(AuthUser.id == uid).first()
        if user:
            item["nspxn_id"] = user.nspxn_id
            item["email"] = user.email
            item["role"] = user.role
            item["is_active"] = user.is_active
            item["last_login"] = user.last_login.isoformat() if user.last_login else None
        if int(item.get("score_count") or 0) > 0:
            item["avg_comprehensive_compliance_score"] = round(float(item.get("score_sum") or 0) / int(item.get("score_count") or 1), 1)
            item["lowest_comprehensive_compliance_score"] = round(float(item.get("score_low")), 1) if item.get("score_low") is not None else None
            item["highest_comprehensive_compliance_score"] = round(float(item.get("score_high")), 1) if item.get("score_high") is not None else None
        else:
            item["avg_comprehensive_compliance_score"] = None
            item["lowest_comprehensive_compliance_score"] = None
            item["highest_comprehensive_compliance_score"] = None

    scored_rows = [float(row.compliance_score) for row in rows if row.status == "completed" and row.ai_intent == "comprehensive" and row.compliance_score is not None]

    return {
        "filters": {
            "start_date": start.isoformat() if start else None,
            "end_date": end.isoformat() if end else None,
            "company_id": company_id,
            "user_id": user_id,
            "status": status or "all",
            "ai_intent": ai_intent or "all",
        },
        "totals": {
            "total_records": len(rows),
            "completed_uploads": status_counts.get("completed", 0),
            "reset_records": status_counts.get("reset_by_admin", 0),
            "other_records": len(rows) - status_counts.get("completed", 0) - status_counts.get("reset_by_admin", 0),
            "avg_comprehensive_compliance_score": round(sum(scored_rows) / len(scored_rows), 1) if scored_rows else None,
            "comprehensive_score_count": len(scored_rows),
        },
        "status_counts": [{"status": k, "count": v} for k, v in sorted(status_counts.items())],
        "request_type_counts": [{"ai_intent": k, "count": v} for k, v in sorted(intent_counts.items())],
        "companies": sorted(company_counts.values(), key=lambda x: (-int(x.get("completed_uploads") or 0), x.get("company_name") or "")),
        "users": sorted(user_counts.values(), key=lambda x: (-int(x.get("completed_uploads") or 0), x.get("nspxn_id") or "")),
    }


@auth_app.get("/admin/usage/export")
def admin_usage_export(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    company_id: Optional[int] = None,
    user_id: Optional[int] = None,
    status: Optional[str] = None,
    ai_intent: Optional[str] = None,
    admin_user: CurrentUser = Depends(require_admin_user),
    db: Session = Depends(get_auth_db),
):
    query, start, end = _filtered_usage_query(db, start_date, end_date, company_id, user_id, status, ai_intent)
    rows = query.order_by(UsageLog.created_at.desc()).limit(10000).all()
    output = io.StringIO()
    writer = csv.writer(output)
    user_score_map = {}
    company_score_map = {}
    for row in rows:
        if row.status == "completed" and row.ai_intent == "comprehensive" and row.compliance_score is not None:
            if row.user_id:
                user_score_map.setdefault(row.user_id, []).append(float(row.compliance_score))
            if row.company_id:
                company_score_map.setdefault(row.company_id, []).append(float(row.compliance_score))

    def _avg(values):
        return round(sum(values) / len(values), 1) if values else ""

    writer.writerow([
        "created_at", "nspxn_id", "user_id", "company_name", "company_id", "ai_intent", "file_number", "status",
        "report_compliance_score", "score_source",
        "user_average_comprehensive_compliance_score", "user_comprehensive_score_count",
        "company_average_comprehensive_compliance_score", "company_comprehensive_score_count",
    ])
    for row in rows:
        user_scores = user_score_map.get(row.user_id, []) if row.user_id else []
        company_scores = company_score_map.get(row.company_id, []) if row.company_id else []
        writer.writerow([
            row.created_at.isoformat() if row.created_at else "",
            row.nspxn_id or "",
            row.user_id or "",
            row.company_name or "",
            row.company_id or "",
            row.ai_intent or "",
            row.file_number or "",
            row.status or "",
            row.compliance_score if row.compliance_score is not None else "",
            row.score_source or "",
            _avg(user_scores),
            len(user_scores),
            _avg(company_scores),
            len(company_scores),
        ])
    filename = f"nspxn_usage_{start.strftime('%Y%m%d') if start else 'all'}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.csv"
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@auth_app.get("/admin/activity")
def admin_activity(
    q: Optional[str] = None,
    action: Optional[str] = None,
    target_type: Optional[str] = None,
    limit: int = 100,
    admin_user: CurrentUser = Depends(require_admin_user),
    db: Session = Depends(get_auth_db),
):
    query = db.query(AdminActivityLog)
    if action:
        query = query.filter(AdminActivityLog.action == action)
    if target_type:
        query = query.filter(AdminActivityLog.target_type == target_type)
    if q:
        like = f"%{q.strip()}%"
        query = query.filter(
            (AdminActivityLog.admin_nspxn_id.ilike(like)) |
            (AdminActivityLog.action.ilike(like)) |
            (AdminActivityLog.target_label.ilike(like)) |
            (AdminActivityLog.details_json.ilike(like))
        )
    rows = query.order_by(AdminActivityLog.created_at.desc()).limit(max(1, min(int(limit or 100), 500))).all()
    return {"activity": [_activity_to_dict(r) for r in rows]}
