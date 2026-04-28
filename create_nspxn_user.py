import getpass

from auth_phase1 import (
    DEFAULT_MONTHLY_UPLOAD_LIMIT,
    DEFAULT_TRIAL_DAYS,
    SessionLocal,
    create_auth_user,
    get_or_create_company,
    init_auth_db,
)


def _input_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return int(default)
    return int(raw)


def main():
    init_auth_db()

    print("Create NSPXN Company/User")
    print("-------------------------")

    company_name = input("Company Name [NSPXN]: ").strip() or "NSPXN"
    plan_name = input("Plan Name [AI-4-IA]: ").strip() or "AI-4-IA"
    monthly_upload_limit = _input_int("Monthly Upload Limit", DEFAULT_MONTHLY_UPLOAD_LIMIT)
    billing_status = input("Billing Status [trial/active] [trial]: ").strip().lower() or "trial"
    trial_days = 0
    if billing_status == "trial":
        trial_days = _input_int("Trial Days", DEFAULT_TRIAL_DAYS)

    print("")
    print("User")
    print("----")
    nspxn_id = input("NSPXN ID #: ").strip()
    email = input("Email: ").strip() or None
    role = input("Role [user/admin]: ").strip() or "user"

    password = getpass.getpass("Temporary Password: ")
    confirm = getpass.getpass("Confirm Password: ")

    if password != confirm:
        print("Passwords do not match.")
        return
    if not password:
        print("Password is required.")
        return
    if len(password.encode("utf-8")) > 72:
        print("Password is too long for bcrypt. Use 72 bytes/characters or less.")
        return

    db = SessionLocal()
    try:
        company = get_or_create_company(
            db=db,
            company_name=company_name,
            plan_name=plan_name,
            monthly_upload_limit=monthly_upload_limit,
            trial_days=trial_days,
            billing_status=billing_status,
            is_active=True,
        )

        user = create_auth_user(
            db=db,
            nspxn_id=nspxn_id,
            password=password,
            email=email,
            company_name=company.company_name,
            role=role,
            is_active=True,
            plan_name=company.plan_name,
            monthly_upload_limit=company.monthly_upload_limit,
            trial_days=trial_days,
            billing_status=company.billing_status,
        )

        print("")
        print("Company/User created successfully.")
        print(f"Company ID: {company.id}")
        print(f"Company: {company.company_name}")
        print(f"Plan: {company.plan_name}")
        print(f"Billing Status: {company.billing_status}")
        print(f"Monthly Upload Limit: {company.monthly_upload_limit}")
        print(f"Trial End: {company.trial_end}")
        print(f"NSPXN ID #: {user.nspxn_id}")
        print(f"Email: {user.email}")
        print(f"Role: {user.role}")
    except Exception as exc:
        print(f"Could not create company/user: {exc}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
