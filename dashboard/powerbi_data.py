"""
SmartGrid PredictAI - Power BI data source
===========================================
Shared join logic for exposing clean, already-merged tables to Power BI, in
two ways:

  1. Power BI Desktop's "Get Data > Python script" connector, which runs this
     file directly and picks up the module-level DataFrames below.
  2. The /api/powerbi/<table> routes in app.py, which call the get_*()
     functions fresh per request so Power BI Service can refresh over HTTP
     without needing local file access.

Desktop usage - Get Data > More... > Other > Python script > paste:

    import sys
    sys.path.append(r"C:\\Users\\SIDNEY MPENYANA\\Desktop\\SmartGrid-PredictAI\\dashboard")
    from powerbi_data import *

Power BI then lists transformers, meters, feeders, technicians, assignments,
inspections, activity_log as selectable tables. Requires the Python
environment configured under Power BI's File > Options > Python scripting to
have pandas installed (the interpreter this project already uses is fine).
"""

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DB_PATH = DATA / "app.db"


def _db_query(sql):
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Fleet risk tables - raw asset attributes merged with model risk scores,
# one row per asset, ready for risk-tier breakdowns and map visuals.
# --------------------------------------------------------------------------
def get_transformers():
    return pd.read_csv(DATA / "transformer_data.csv").merge(
        pd.read_csv(DATA / "transformer_risk_scores.csv"), on="transformer_id", suffixes=("", "_score")
    )


def get_meters():
    return pd.read_csv(DATA / "meter_data.csv").merge(
        pd.read_csv(DATA / "meter_theft_scores.csv"), on="meter_id"
    )


def get_feeders():
    return pd.read_csv(DATA / "feeder_data.csv").merge(
        pd.read_csv(DATA / "feeder_outage_scores.csv"), on="feeder_id", suffixes=("", "_score")
    )


# --------------------------------------------------------------------------
# Field operations tables - from the SQLite app.db, technician names
# resolved so Power BI doesn't need a separate users lookup.
# --------------------------------------------------------------------------
def get_technicians():
    return _db_query(
        "SELECT id AS technician_id, username, full_name, is_active, created_at "
        "FROM users WHERE role = 'technician'"
    )


def get_assignments():
    return _db_query(
        "SELECT a.id, a.technician_id, u.full_name AS technician_name, "
        "a.transformer_id, a.assigned_at "
        "FROM assignments a JOIN users u ON u.id = a.technician_id"
    )


def get_inspections():
    return _db_query(
        "SELECT i.id, i.transformer_id, i.technician_id, u.full_name AS technician_name, "
        "i.status, i.notes, i.photo_filename, i.created_at "
        "FROM inspections i JOIN users u ON u.id = i.technician_id"
    )


def get_activity_log():
    return _db_query("SELECT id, user_id, username, action, created_at FROM activity_log")


TABLES = {
    "transformers": get_transformers,
    "meters": get_meters,
    "feeders": get_feeders,
    "technicians": get_technicians,
    "assignments": get_assignments,
    "inspections": get_inspections,
    "activity_log": get_activity_log,
}

# Module-level variables for the Power BI Desktop "Python script" connector,
# which detects top-level DataFrame variables after running the script.
transformers = get_transformers()
meters = get_meters()
feeders = get_feeders()
technicians = get_technicians()
assignments = get_assignments()
inspections = get_inspections()
activity_log = get_activity_log()

if __name__ == "__main__":
    for name, df in [
        ("transformers", transformers),
        ("meters", meters),
        ("feeders", feeders),
        ("technicians", technicians),
        ("assignments", assignments),
        ("inspections", inspections),
        ("activity_log", activity_log),
    ]:
        print(f"{name}: {df.shape}")
