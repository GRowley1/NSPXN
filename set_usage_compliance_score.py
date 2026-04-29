from auth_phase1 import SessionLocal, UsageLog, init_auth_db


def main():
    init_auth_db()
    print("Set / Backfill Compliance Score for Usage Log")
    print("----------------------------------------------")
    file_number = input("File Number to update: ").strip()
    if not file_number:
        print("File Number is required.")
        return

    raw_score = input("Compliance Score (0-100): ").strip()
    try:
        score = float(raw_score)
    except Exception:
        print("Compliance Score must be a number.")
        return
    if score < 0 or score > 100:
        print("Compliance Score must be between 0 and 100.")
        return

    nspxn_id = input("Optional NSPXN ID filter (press Enter to skip): ").strip()

    db = SessionLocal()
    try:
        q = db.query(UsageLog).filter(UsageLog.file_number == file_number)
        if nspxn_id:
            q = q.filter(UsageLog.nspxn_id == nspxn_id)
        rows = q.order_by(UsageLog.created_at.desc()).all()
        if not rows:
            print("No usage_log rows found for that file number.")
            return

        print(f"Found {len(rows)} matching usage row(s):")
        for idx, r in enumerate(rows, 1):
            print(f"[{idx}] id={r.id} created_at={r.created_at} nspxn_id={r.nspxn_id} company={r.company_name} intent={r.ai_intent} status={r.status} current_score={getattr(r, 'compliance_score', None)}")

        choice = input("Update which row? Enter number, or 'all' [1]: ").strip().lower() or "1"
        if choice == "all":
            selected = rows
        else:
            try:
                selected = [rows[int(choice) - 1]]
            except Exception:
                print("Invalid selection.")
                return

        for r in selected:
            r.compliance_score = score
            r.score_source = "manual_admin_backfill"
            if not r.ai_intent:
                r.ai_intent = "comprehensive"
        db.commit()
        print(f"Updated {len(selected)} usage row(s) with compliance score {score}.")
        print("Refresh the admin dashboard after deploy/hard refresh.")
    except Exception as exc:
        db.rollback()
        print(f"Could not update score: {exc}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
