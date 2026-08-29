"""
SmartGrid PredictAI - Authentication & Role-Based Access Control
====================================================================
A small SQLite-backed user store for the Flask dashboard - real password
hashing (werkzeug, already a Flask dependency), real sessions, real
per-role data (technician assignments, inspection submissions, an activity
log) - not a cosmetic login screen in front of an otherwise-open app.

Six roles, modeling a least-privilege Eskom-style access structure (role
*keys* below are the stable internal identifiers; display names live in
ROLE_LABELS and have since been renamed to match the enterprise model):
    administrator (System Administrator)
                    - manage users, full visibility, system/dataset info
    engineer (Asset Management)
                    - run predictions, generate reports, use the AI Assistant,
                      assign transformers to technicians
    investigator (Revenue Protection / Loss Control)
                    - assign technicians to meters flagged for suspected theft
    technician (Field Technician)
                    - view assigned transformers/meters, submit inspections
                      and theft investigations
    manager (Management / Executive)
                    - read-only: Executive Overview, Asset Management, and
                      per-model dashboards; no assignment or write actions
    auditor (Auditor / Compliance)
                    - read-only: the activity log and inspection/investigation
                      history, nothing else - can't see or touch user accounts

This is intentionally not an "everyone sees everything" prototype: each role
only reaches the panels/APIs its PANEL_ROLES entry (dashboard/app.py) grants,
mirroring the least-privilege principle a real utility's IT/security team
would require.

The database lives at data/app.db, separate from the synthetic asset CSVs
in the same folder. init_db() creates the schema and seeds one demo account
per role the first time it runs, so a clean checkout works immediately - see
DEMO_ACCOUNTS below for the credentials.
"""

import sqlite3
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path

from flask import g, has_app_context, redirect, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "app.db"
UPLOAD_DIR = ROOT / "data" / "inspection_uploads"

# Role *keys* (used in session/DB) are kept stable even where the display
# label has since been renamed for the enterprise access model - only
# ROLE_LABELS changed, not the underlying identifier, so this rename carries
# zero risk to existing PANEL_ROLES/roles_required() checks.
ROLES = ["administrator", "engineer", "investigator", "technician", "manager", "auditor"]
ROLE_LABELS = {
    "administrator": "System Administrator",
    "engineer": "Asset Management",
    "investigator": "Revenue Protection / Loss Control",
    "technician": "Field Technician",
    "manager": "Management / Executive",
    "auditor": "Auditor / Compliance",
}

# Demo credentials for this prototype - printed to the console on first
# run and documented in the README, never a real production secret store.
DEMO_ACCOUNTS = [
    ("admin", "admin123", "Sidney Mpenyana", "administrator"),
    ("engineer", "engineer123", "Sipho Dlamini", "engineer"),
    ("investigator", "investigator123", "Naledi Sithole", "investigator"),
    ("technician", "tech123", "Lerato Mokoena", "technician"),
    ("manager", "manager123", "Thandiwe Nkosi", "manager"),
    ("auditor", "auditor123", "Piet van der Merwe", "auditor"),
]
# How many transformers/meters the demo technician account starts with
# assigned - real rows in the assignments/meter_assignments tables, not a
# value invented at answer time.
DEMO_ASSIGNMENT_COUNT = 12
DEMO_METER_ASSIGNMENT_COUNT = 5


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_db():
    """One connection per request, cached on flask.g - the standard Flask
    + sqlite3 pattern. Falls back to a fresh, short-lived connection when
    called with no Flask request active (PDF report generation reached from
    streamlit_app.py or a test, neither of which has an app context)."""
    if not has_app_context():
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


def close_db(_exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Creates the schema if missing and seeds demo accounts + the demo
    technician's assignments on first run. Safe to call on every app
    startup - every statement is idempotent."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            role TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            technician_id INTEGER NOT NULL REFERENCES users(id),
            transformer_id TEXT NOT NULL,
            assigned_at TEXT NOT NULL,
            UNIQUE(technician_id, transformer_id)
        );

        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transformer_id TEXT NOT NULL,
            technician_id INTEGER NOT NULL REFERENCES users(id),
            status TEXT NOT NULL,
            notes TEXT,
            photo_filename TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS meter_assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            technician_id INTEGER NOT NULL REFERENCES users(id),
            meter_id TEXT NOT NULL,
            assigned_at TEXT NOT NULL,
            UNIQUE(technician_id, meter_id)
        );

        CREATE TABLE IF NOT EXISTS meter_investigations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meter_id TEXT NOT NULL,
            technician_id INTEGER NOT NULL REFERENCES users(id),
            status TEXT NOT NULL,
            notes TEXT,
            photo_filename TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS activity_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER REFERENCES users(id),
            username TEXT,
            action TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.commit()

    existing = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if existing == 0:
        technician_id = None
        for username, password, full_name, role in DEMO_ACCOUNTS:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, full_name, role, is_active, created_at) "
                "VALUES (?, ?, ?, ?, 1, ?)",
                (username, generate_password_hash(password), full_name, role, _now()),
            )
            if role == "technician":
                technician_id = cur.lastrowid
        conn.commit()

        if technician_id is not None:
            transformer_csv = ROOT / "data" / "transformer_data.csv"
            if transformer_csv.exists():
                import pandas as pd

                ids = pd.read_csv(transformer_csv)["transformer_id"].tolist()
                sample = ids[:: max(1, len(ids) // DEMO_ASSIGNMENT_COUNT)][:DEMO_ASSIGNMENT_COUNT]
                for transformer_id in sample:
                    conn.execute(
                        "INSERT OR IGNORE INTO assignments (technician_id, transformer_id, assigned_at) "
                        "VALUES (?, ?, ?)",
                        (technician_id, transformer_id, _now()),
                    )
                conn.commit()

            meter_csv = ROOT / "data" / "meter_theft_scores.csv"
            if meter_csv.exists():
                import pandas as pd

                flagged = pd.read_csv(meter_csv)
                flagged = flagged[flagged["investigation_flag"] == 1]["meter_id"].tolist()
                sample = flagged[:DEMO_METER_ASSIGNMENT_COUNT]
                for meter_id in sample:
                    conn.execute(
                        "INSERT OR IGNORE INTO meter_assignments (technician_id, meter_id, assigned_at) "
                        "VALUES (?, ?, ?)",
                        (technician_id, meter_id, _now()),
                    )
                conn.commit()

        print("=" * 60)
        print("SmartGrid PredictAI - demo accounts seeded in data/app.db")
        for username, password, full_name, role in DEMO_ACCOUNTS:
            print(f"  {username:<12} / {password:<14} {ROLE_LABELS[role]}")
        print("=" * 60)

    else:
        # Backfill for a database seeded before the investigator role and
        # meter-investigation workflow existed - additive only, never
        # touches existing users/assignments/inspections.
        existing_usernames = {
            row[0] for row in conn.execute("SELECT username FROM users").fetchall()
        }
        for username, password, full_name, role in DEMO_ACCOUNTS:
            if username in existing_usernames:
                continue
            conn.execute(
                "INSERT INTO users (username, password_hash, full_name, role, is_active, created_at) "
                "VALUES (?, ?, ?, ?, 1, ?)",
                (username, generate_password_hash(password), full_name, role, _now()),
            )
            conn.commit()
            print(f"SmartGrid PredictAI - added new demo account: {username} / {password} ({ROLE_LABELS[role]})")

        technician_row = conn.execute(
            "SELECT id FROM users WHERE role = 'technician' ORDER BY created_at LIMIT 1"
        ).fetchone()
        has_meter_assignments = conn.execute("SELECT COUNT(*) FROM meter_assignments").fetchone()[0]
        if technician_row is not None and has_meter_assignments == 0:
            meter_csv = ROOT / "data" / "meter_theft_scores.csv"
            if meter_csv.exists():
                import pandas as pd

                flagged = pd.read_csv(meter_csv)
                flagged = flagged[flagged["investigation_flag"] == 1]["meter_id"].tolist()
                for meter_id in flagged[:DEMO_METER_ASSIGNMENT_COUNT]:
                    conn.execute(
                        "INSERT OR IGNORE INTO meter_assignments (technician_id, meter_id, assigned_at) "
                        "VALUES (?, ?, ?)",
                        (technician_row[0], meter_id, _now()),
                    )
                conn.commit()

    conn.close()


# --------------------------------------------------------------------------
# User CRUD
# --------------------------------------------------------------------------
def verify_login(username, password):
    row = get_db().execute(
        "SELECT * FROM users WHERE username = ? AND is_active = 1", (username,)
    ).fetchone()
    if row is None or not check_password_hash(row["password_hash"], password):
        return None
    return dict(row)


def get_user(user_id):
    row = get_db().execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def list_users():
    rows = get_db().execute(
        "SELECT id, username, full_name, role, is_active, created_at FROM users ORDER BY created_at"
    ).fetchall()
    return [dict(r) for r in rows]


def create_user(username, password, full_name, role):
    if role not in ROLES:
        raise ValueError(f"unknown role: {role}")
    if not username or not password or not full_name:
        raise ValueError("username, password, and full name are required")
    db = get_db()
    existing = db.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
    if existing:
        raise ValueError(f"username '{username}' is already taken")
    cur = db.execute(
        "INSERT INTO users (username, password_hash, full_name, role, is_active, created_at) "
        "VALUES (?, ?, ?, ?, 1, ?)",
        (username, generate_password_hash(password), full_name, role, _now()),
    )
    db.commit()
    return cur.lastrowid


def set_user_active(user_id, is_active):
    db = get_db()
    db.execute("UPDATE users SET is_active = ? WHERE id = ?", (1 if is_active else 0, user_id))
    db.commit()


# --------------------------------------------------------------------------
# Technician assignments + inspections
# --------------------------------------------------------------------------
def list_technicians():
    rows = get_db().execute(
        "SELECT id, username, full_name FROM users WHERE role = 'technician' AND is_active = 1 ORDER BY full_name"
    ).fetchall()
    return [dict(r) for r in rows]


def get_assigned_transformer_ids(technician_id):
    rows = get_db().execute(
        "SELECT transformer_id FROM assignments WHERE technician_id = ? ORDER BY assigned_at", (technician_id,)
    ).fetchall()
    return [r["transformer_id"] for r in rows]


def assign_transformer(technician_id, transformer_id):
    """Idempotent - re-assigning a transformer a technician already has is a
    no-op rather than a duplicate row or an error, since the UNIQUE
    constraint on (technician_id, transformer_id) plus INSERT OR IGNORE
    means the same click twice just leaves things as they were."""
    db = get_db()
    db.execute(
        "INSERT OR IGNORE INTO assignments (technician_id, transformer_id, assigned_at) VALUES (?, ?, ?)",
        (technician_id, transformer_id, _now()),
    )
    db.commit()


def unassign_transformer(technician_id, transformer_id):
    db = get_db()
    db.execute(
        "DELETE FROM assignments WHERE technician_id = ? AND transformer_id = ?",
        (technician_id, transformer_id),
    )
    db.commit()


def submit_inspection(transformer_id, technician_id, status, notes, photo_filename):
    db = get_db()
    cur = db.execute(
        "INSERT INTO inspections (transformer_id, technician_id, status, notes, photo_filename, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (transformer_id, technician_id, status, notes, photo_filename, _now()),
    )
    db.commit()
    return cur.lastrowid


def get_inspection_history(transformer_id, limit=10):
    rows = get_db().execute(
        "SELECT i.*, u.full_name AS technician_name FROM inspections i "
        "JOIN users u ON u.id = i.technician_id "
        "WHERE i.transformer_id = ? ORDER BY i.created_at DESC LIMIT ?",
        (transformer_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_recent_inspections_by_technician(technician_id, days=90, limit=50):
    """Every inspection a technician has personally submitted in the last
    `days` days (default ~3 months, comfortably covers "inspected 2 months
    ago"), newest first - not scoped to their *current* assignments, since a
    transformer can be reassigned after being inspected."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    rows = get_db().execute(
        "SELECT i.*, u.full_name AS technician_name FROM inspections i "
        "JOIN users u ON u.id = i.technician_id "
        "WHERE i.technician_id = ? AND i.created_at >= ? ORDER BY i.created_at DESC LIMIT ?",
        (technician_id, cutoff, limit),
    ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# Technician meter assignments + theft investigations - same shape as the
# transformer assignments/inspections above, kept as separate tables/
# functions rather than a shared "entity_id" column since meters have no
# feeder/substation hierarchy and a different investigation status
# vocabulary (confirmed_theft/false_positive instead of needs_followup).
# --------------------------------------------------------------------------
def get_assigned_meter_ids(technician_id):
    rows = get_db().execute(
        "SELECT meter_id FROM meter_assignments WHERE technician_id = ? ORDER BY assigned_at", (technician_id,)
    ).fetchall()
    return [r["meter_id"] for r in rows]


def assign_meter(technician_id, meter_id):
    db = get_db()
    db.execute(
        "INSERT OR IGNORE INTO meter_assignments (technician_id, meter_id, assigned_at) VALUES (?, ?, ?)",
        (technician_id, meter_id, _now()),
    )
    db.commit()


def unassign_meter(technician_id, meter_id):
    db = get_db()
    db.execute(
        "DELETE FROM meter_assignments WHERE technician_id = ? AND meter_id = ?",
        (technician_id, meter_id),
    )
    db.commit()


def submit_meter_investigation(meter_id, technician_id, status, notes, photo_filename):
    db = get_db()
    cur = db.execute(
        "INSERT INTO meter_investigations (meter_id, technician_id, status, notes, photo_filename, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (meter_id, technician_id, status, notes, photo_filename, _now()),
    )
    db.commit()
    return cur.lastrowid


def get_meter_investigation_history(meter_id, limit=10):
    rows = get_db().execute(
        "SELECT i.*, u.full_name AS technician_name FROM meter_investigations i "
        "JOIN users u ON u.id = i.technician_id "
        "WHERE i.meter_id = ? ORDER BY i.created_at DESC LIMIT ?",
        (meter_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_recent_meter_investigations_by_technician(technician_id, days=90, limit=50):
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    rows = get_db().execute(
        "SELECT i.*, u.full_name AS technician_name FROM meter_investigations i "
        "JOIN users u ON u.id = i.technician_id "
        "WHERE i.technician_id = ? AND i.created_at >= ? ORDER BY i.created_at DESC LIMIT ?",
        (technician_id, cutoff, limit),
    ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# Maintenance queue - real backlog counts for the Fleet Dashboard's queue
# widget, not invented numbers. "Pending" here means the assignment's most
# recent submission (if any) hasn't reached a resolved end state yet -
# inspected for transformers, confirmed_theft/false_positive for meters.
# "Unassigned emergency" means the model already flagged it as the worst
# tier and nobody has been dispatched to it at all - the actionable gap a
# planner most needs to see.
# --------------------------------------------------------------------------
def count_pending_transformer_inspections():
    row = get_db().execute(
        """
        SELECT COUNT(*) FROM assignments a
        WHERE COALESCE(
            (SELECT i.status FROM inspections i
             WHERE i.transformer_id = a.transformer_id AND i.technician_id = a.technician_id
             ORDER BY i.created_at DESC LIMIT 1),
            'pending'
        ) != 'inspected'
        """
    ).fetchone()
    return row[0]


def count_pending_meter_investigations():
    row = get_db().execute(
        """
        SELECT COUNT(*) FROM meter_assignments a
        WHERE COALESCE(
            (SELECT i.status FROM meter_investigations i
             WHERE i.meter_id = a.meter_id AND i.technician_id = a.technician_id
             ORDER BY i.created_at DESC LIMIT 1),
            'pending'
        ) IN ('pending', 'investigating')
        """
    ).fetchone()
    return row[0]


def count_unassigned_emergency_transformers():
    import pandas as pd

    path = ROOT / "data" / "transformer_risk_scores.csv"
    if not path.exists():
        return 0
    scores = pd.read_csv(path)
    emergency_ids = set(scores.loc[scores["risk_tier"] == "emergency", "transformer_id"])
    assigned_ids = {r["transformer_id"] for r in get_db().execute("SELECT DISTINCT transformer_id FROM assignments")}
    return len(emergency_ids - assigned_ids)


def count_unassigned_emergency_meters():
    import pandas as pd

    path = ROOT / "data" / "meter_theft_scores.csv"
    if not path.exists():
        return 0
    scores = pd.read_csv(path)
    emergency_ids = set(scores.loc[scores["priority_tier"] == "emergency", "meter_id"])
    assigned_ids = {r["meter_id"] for r in get_db().execute("SELECT DISTINCT meter_id FROM meter_assignments")}
    return len(emergency_ids - assigned_ids)


def count_overdue_transformer_maintenance():
    """Real backlog count for Asset Management - how many transformers have
    already passed the model's own next_maintenance_date, not a decorative
    number. No DB involved, purely a CSV comparison against today's date."""
    import pandas as pd

    path = ROOT / "data" / "transformer_risk_scores.csv"
    if not path.exists():
        return 0
    scores = pd.read_csv(path)
    if "next_maintenance_date" not in scores.columns:
        return 0
    due_dates = pd.to_datetime(scores["next_maintenance_date"])
    return int((due_dates < pd.Timestamp(datetime.now().date())).sum())


# --------------------------------------------------------------------------
# Activity log
# --------------------------------------------------------------------------
def log_activity(user, action):
    db = get_db()
    db.execute(
        "INSERT INTO activity_log (user_id, username, action, created_at) VALUES (?, ?, ?, ?)",
        (user["id"] if user else None, user["username"] if user else "system", action, _now()),
    )
    db.commit()


def recent_activity(limit=50):
    rows = get_db().execute(
        "SELECT * FROM activity_log ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# Fleet-wide audit views - unlike get_recent_inspections_by_technician /
# get_recent_meter_investigations_by_technician (scoped to one technician's
# own submissions), these are unfiltered across every technician - built for
# the auditor role, which needs to see everyone's submissions, not just its
# own.
# --------------------------------------------------------------------------
def recent_inspections_fleet(limit=30):
    rows = get_db().execute(
        "SELECT i.*, u.full_name AS technician_name FROM inspections i "
        "JOIN users u ON u.id = i.technician_id "
        "ORDER BY i.created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def recent_meter_investigations_fleet(limit=30):
    rows = get_db().execute(
        "SELECT i.*, u.full_name AS technician_name FROM meter_investigations i "
        "JOIN users u ON u.id = i.technician_id "
        "ORDER BY i.created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# AI Assistant conversation history - stored server-side per user rather than
# in the session cookie (which has a ~4KB limit multi-turn history would
# blow through), and so it survives a page reload. Only user/assistant text
# turns are kept - not the tool_use/tool_result exchange within a turn, since
# each new question re-runs the real tools for fresh data anyway; there's no
# need to replay stale tool output from a prior turn.
# --------------------------------------------------------------------------
def save_chat_message(user_id, role, content):
    db = get_db()
    db.execute(
        "INSERT INTO chat_messages (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (user_id, role, content, _now()),
    )
    db.commit()


def get_chat_history(user_id, limit=20):
    rows = get_db().execute(
        "SELECT role, content FROM chat_messages WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def clear_chat_history(user_id):
    db = get_db()
    db.execute("DELETE FROM chat_messages WHERE user_id = ?", (user_id,))
    db.commit()


# --------------------------------------------------------------------------
# Dataset stats (for the admin Settings panel)
# --------------------------------------------------------------------------
def dataset_stats():
    import pandas as pd

    files = [
        ("Transformers", "transformer_data.csv"),
        ("Transformer risk scores", "transformer_risk_scores.csv"),
        ("Meters", "meter_data.csv"),
        ("Meter theft scores", "meter_theft_scores.csv"),
        ("Feeders", "feeder_data.csv"),
        ("Feeder outage scores", "feeder_outage_scores.csv"),
    ]
    stats = []
    for label, filename in files:
        path = ROOT / "data" / filename
        if not path.exists():
            stats.append({"label": label, "filename": filename, "rows": None, "modified": None})
            continue
        rows = sum(1 for _ in open(path, encoding="utf-8")) - 1
        modified = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        stats.append({"label": label, "filename": filename, "rows": rows, "modified": modified})
    return stats


def fleet_counts():
    """Real row counts straight off disk - used on the (unauthenticated)
    login page, so the fleet-size stats there track data/generate_data.py's
    actual output instead of a number typed into the template. None for a
    file that hasn't been generated yet."""

    def count(filename):
        path = ROOT / "data" / filename
        if not path.exists():
            return None
        return sum(1 for _ in open(path, encoding="utf-8")) - 1

    return {
        "transformers": count("transformer_data.csv"),
        "meters": count("meter_data.csv"),
        "feeders": count("feeder_data.csv"),
    }


# --------------------------------------------------------------------------
# Flask session helpers / decorators
# --------------------------------------------------------------------------
def current_user():
    user_id = session.get("user_id")
    if user_id is None:
        return None
    return get_user(user_id)


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def roles_required(*roles):
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if not session.get("user_id"):
                return redirect(url_for("login", next=request.path))
            if session.get("role") not in roles:
                return "Forbidden - your role does not have access to this page.", 403
            return view(*args, **kwargs)

        return wrapped

    return decorator
