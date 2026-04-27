import getpass

from auth_phase1 import SessionLocal, create_auth_user, init_auth_db


def main():
    init_auth_db()

    print("Create NSPXN Auth User")
    print("----------------------")

    nspxn_id = input("NSPXN ID #: ").strip()
    email = input("Email: ").strip() or None
    company_name = input("Company Name: ").strip() or None
    role = input("Role [user/admin]: ").strip() or "user"

    password = getpass.getpass("Temporary Password: ")
    confirm = getpass.getpass("Confirm Password: ")

    if password != confirm:
        print("Passwords do not match.")
        return
    if not password:
        print("Password is required.")
        return

    db = SessionLocal()
    try:
        user = create_auth_user(
            db=db,
            nspxn_id=nspxn_id,
            password=password,
            email=email,
            company_name=company_name,
            role=role,
            is_active=True,
        )
        print("")
        print("User created successfully.")
        print(f"NSPXN ID #: {user.nspxn_id}")
        print(f"Email: {user.email}")
        print(f"Company: {user.company_name}")
        print(f"Role: {user.role}")
    except Exception as exc:
        print(f"Could not create user: {exc}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
