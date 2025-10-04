from collections import defaultdict
from app.db import SessionLocal
from app.models import Model

def main():
    s = SessionLocal()
    try:
        rows = s.query(Model).all()
        by_repo = defaultdict(list)
        for r in rows:
            by_repo[r.repo_url].append(r)

        deleted = 0
        for repo, group in by_repo.items():
            if len(group) <= 1:
                continue
            # keep newest, delete the rest
            group.sort(key=lambda r: ((r.updated_at or r.created_at), r.id), reverse=True)
            keep = group[0]
            for extra in group[1:]:
                s.delete(extra)
                deleted += 1
        s.commit()
        print(f"Removed {deleted} duplicate rows.")
    finally:
        s.close()

if __name__ == "__main__":
    main()
