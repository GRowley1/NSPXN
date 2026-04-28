import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List

import jwt
from jwt import ExpiredSignatureError, InvalidTokenError
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, create_engine, func, inspect, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# NSPXN AUTH PHASE 2
# - Database-backed users
# - Company/account controls
# - Monthly upload limits
# - Trial expiration
# - Usage logging
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


def require_admin_user(current_user: CurrentUser = Depends(get_current_auth_user)) -> CurrentUser:
    if (current_user.role or "").lower() != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return current_user


def _company_to_dict(db: Session, company: Company) -> dict:
    used = count_company_uploads_this_month(db, company.id)
    limit = int(company.monthly_upload_limit or 0)
    return {
        "id": company.id,
        "company_name": company.company_name,
        "plan_name": company.plan_name,
        "monthly_upload_limit": company.monthly_upload_limit,
        "uploads_used_this_month": used,
        "uploads_remaining_this_month": max(limit - used, 0),
        "trial_start": company.trial_start.isoformat() if company.trial_start else None,
        "trial_end": company.trial_end.isoformat() if company.trial_end else None,
        "billing_status": company.billing_status,
        "is_active": company.is_active,
        "created_at": company.created_at.isoformat() if company.created_at else None,
    }


def _user_to_dict(db: Session, user: AuthUser) -> dict:
    company = get_company_for_user(db, user)
    return {
        "id": user.id,
        "nspxn_id": user.nspxn_id,
        "email": user.email,
        "company_id": company.id if company else user.company_id,
        "company_name": company.company_name if company else user.company_name,
        "role": user.role,
        "is_active": user.is_active,
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
        "created_at": entry.created_at.isoformat() if entry.created_at else None,
    }


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

