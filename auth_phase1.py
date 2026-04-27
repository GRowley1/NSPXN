import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from jwt import ExpiredSignatureError, InvalidTokenError
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import Boolean, Column, DateTime, Integer, String, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# NSPXN AUTH PHASE 1
# - Database-backed users
# - bcrypt password hashing
# - JWT bearer token login
# - Helper for ASGI router auth checks
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nspxn_auth.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "CHANGE_ME_IN_RENDER")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "480"))

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

auth_app = FastAPI(title="NSPXN Auth Phase 1")
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


class AuthUser(Base):
    __tablename__ = "auth_users"

    id = Column(Integer, primary_key=True, index=True)
    nspxn_id = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=True)
    password_hash = Column(String, nullable=False)
    company_name = Column(String, nullable=True)
    role = Column(String, default="user", nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)


class LoginRequest(BaseModel):
    nspxn_id: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    nspxn_id: str
    email: Optional[str] = None
    company_name: Optional[str] = None
    role: str


class CurrentUser(BaseModel):
    id: int
    nspxn_id: str
    email: Optional[str] = None
    company_name: Optional[str] = None
    role: str
    is_active: bool


def init_auth_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_auth_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    return pwd_context.verify(plain_password, password_hash)


def create_access_token(user: AuthUser) -> str:
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {
        "sub": user.nspxn_id,
        "user_id": user.id,
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


def get_user_by_nspxn_id(db: Session, nspxn_id: str) -> Optional[AuthUser]:
    clean_id = (nspxn_id or "").strip()
    if not clean_id:
        return None
    return db.query(AuthUser).filter(AuthUser.nspxn_id == clean_id).first()


def authenticate_user(db: Session, nspxn_id: str, password: str) -> Optional[AuthUser]:
    user = get_user_by_nspxn_id(db, nspxn_id)
    if not user:
        return None
    if not user.is_active:
        raise HTTPException(status_code=403, detail="This NSPXN account is inactive.")
    if not verify_password(password, user.password_hash):
        return None
    user.last_login = datetime.utcnow()
    db.commit()
    return user


def _to_current_user(user: AuthUser) -> CurrentUser:
    return CurrentUser(
        id=user.id,
        nspxn_id=user.nspxn_id,
        email=user.email,
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


def create_auth_user(
    db: Session,
    nspxn_id: str,
    password: str,
    email: Optional[str] = None,
    company_name: Optional[str] = None,
    role: str = "user",
    is_active: bool = True,
) -> AuthUser:
    clean_id = (nspxn_id or "").strip()
    if not clean_id:
        raise ValueError("nspxn_id is required")
    if not password:
        raise ValueError("password is required")
    existing = get_user_by_nspxn_id(db, clean_id)
    if existing:
        raise ValueError(f"NSPXN ID already exists: {clean_id}")

    user = AuthUser(
        nspxn_id=clean_id,
        email=email or None,
        password_hash=hash_password(password),
        company_name=company_name or None,
        role=role or "user",
        is_active=is_active,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@auth_app.on_event("startup")
def _startup_auth_db() -> None:
    init_auth_db()


@auth_app.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, db: Session = Depends(get_auth_db)):
    user = authenticate_user(db=db, nspxn_id=payload.nspxn_id, password=payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid NSPXN ID # or Password.")

    token = create_access_token(user)
    return LoginResponse(
        access_token=token,
        nspxn_id=user.nspxn_id,
        email=user.email,
        company_name=user.company_name,
        role=user.role,
    )


@auth_app.get("/me")
def me(current_user: CurrentUser = Depends(get_current_auth_user)):
    return {
        "nspxn_id": current_user.nspxn_id,
        "email": current_user.email,
        "company_name": current_user.company_name,
        "role": current_user.role,
        "is_active": current_user.is_active,
    }
