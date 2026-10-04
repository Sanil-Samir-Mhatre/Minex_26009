"""
Rule-based recommendation engine.
Transparent, explainable rules — no LLM dependency.
"""
import pandas as pd
from typing import List, Dict, Any


# ── Threshold constants ────────────────────────────────────────────────────────
THRESHOLDS = {
    "downtime_high":     4.0,    # hrs/day
    "downtime_critical": 7.0,
    "blast_delay_high":  3.0,
    "blast_delay_critical": 5.0,
    "rainfall_moderate": 20.0,   # mm/day
    "rainfall_high":     50.0,
    "equip_avail_low":   75.0,   # %
    "equip_avail_critical": 60.0,
    "shortfall_medium":  10.0,   # %
    "shortfall_high":    20.0,
    "shortfall_critical":30.0,
}


def _badge(level: str) -> str:
    colours = {
        "CRITICAL": "🔴",
        "HIGH":     "🟠",
        "MEDIUM":   "🟡",
        "LOW":      "🟢",
        "INFO":     "🔵",
    }
    return colours.get(level, "⚪")


def generate_recommendations(
    mine: str,
    planned_tpd: float,
    predicted_tpd: float,
    shortfall_pct: float,
    downtime_hrs: float,
    blast_delay_hrs: float,
    rainfall_mm: float,
    equip_avail_pct: float,
    maintenance_event: int = 0,
) -> List[Dict[str, Any]]:
    """
    Returns a list of recommendation dicts sorted by severity.

    Each dict:
        severity       : CRITICAL / HIGH / MEDIUM / LOW / INFO
        factor         : Short factor name
        evidence       : Quantified observation
        recommendation : Actionable text
        icon           : Emoji badge
    """
    recs = []
    T = THRESHOLDS

    # ── Equipment downtime ────────────────────────────────────────────────────
    if downtime_hrs >= T["downtime_critical"]:
        recs.append({
            "severity": "CRITICAL",
            "factor": "Equipment Downtime",
            "evidence": f"{downtime_hrs:.1f} hrs/day downtime (threshold: {T['downtime_critical']} hrs)",
            "recommendation": (
                "Immediately mobilise standby equipment. Trigger emergency maintenance SOP. "
                "Reallocate heavy machinery from low-priority zones to critical production faces. "
                "Escalate to Mine Manager."
            ),
        })
    elif downtime_hrs >= T["downtime_high"]:
        recs.append({
            "severity": "HIGH",
            "factor": "Equipment Downtime",
            "evidence": f"{downtime_hrs:.1f} hrs/day downtime",
            "recommendation": (
                "Reallocate available equipment to high-priority production zones. "
                "Advance preventive maintenance scheduling to off-peak shifts. "
                "Review equipment rotation plan."
            ),
        })

    # ── Blasting delay ────────────────────────────────────────────────────────
    if blast_delay_hrs >= T["blast_delay_critical"]:
        recs.append({
            "severity": "CRITICAL",
            "factor": "Blasting Delay",
            "evidence": f"{blast_delay_hrs:.1f} hrs delay (critical threshold: {T['blast_delay_critical']} hrs)",
            "recommendation": (
                "Invoke contingency blasting schedule. Pre-position explosive materials. "
                "Reschedule downstream loading and haulage fleet to prevent cascading delays. "
                "Coordinate with drilling crew for faster face preparation."
            ),
        })
    elif blast_delay_hrs >= T["blast_delay_high"]:
        recs.append({
            "severity": "HIGH",
            "factor": "Blasting Delay",
            "evidence": f"{blast_delay_hrs:.1f} hrs blasting delay",
            "recommendation": (
                "Prioritise delayed blasting operations. Reschedule downstream loading/haulage. "
                "Consider night blasting if permitted and weather conditions are favourable."
            ),
        })

    # ── Rainfall ─────────────────────────────────────────────────────────────
    if rainfall_mm >= T["rainfall_high"]:
        recs.append({
            "severity": "HIGH",
            "factor": "Heavy Rainfall",
            "evidence": f"{rainfall_mm:.1f} mm rainfall today",
            "recommendation": (
                "Activate mine drainage protocol. Suspend open-face blasting during active rainfall. "
                "Redeploy workers to underground sections or covered maintenance tasks. "
                "Review shift allocation and material movement schedule."
            ),
        })
    elif rainfall_mm >= T["rainfall_moderate"]:
        recs.append({
            "severity": "MEDIUM",
            "factor": "Moderate Rainfall",
            "evidence": f"{rainfall_mm:.1f} mm rainfall today",
            "recommendation": (
                "Monitor mine drainage status. Adjust haulage routes to avoid waterlogged areas. "
                "Ensure ore stockpile covers are in place to prevent grade dilution."
            ),
        })

    # ── Equipment availability ────────────────────────────────────────────────
    if equip_avail_pct < T["equip_avail_critical"]:
        recs.append({
            "severity": "CRITICAL",
            "factor": "Low Equipment Availability",
            "evidence": f"{equip_avail_pct:.1f}% availability (critical < {T['equip_avail_critical']}%)",
            "recommendation": (
                "Declare equipment crisis. Source third-party hire for shovels/dumpers immediately. "
                "Cancel non-essential maintenance work orders. "
                "Concentrate remaining fleet on highest-grade ore faces."
            ),
        })
    elif equip_avail_pct < T["equip_avail_low"]:
        recs.append({
            "severity": "HIGH",
            "factor": "Equipment Availability",
            "evidence": f"{equip_avail_pct:.1f}% availability (low threshold: {T['equip_avail_low']}%)",
            "recommendation": (
                "Prioritise critical equipment maintenance. Consider redeployment from lower-priority mines. "
                "Review spare-parts inventory and procurement lead times."
            ),
        })

    # ── Maintenance event ─────────────────────────────────────────────────────
    if maintenance_event:
        recs.append({
            "severity": "MEDIUM",
            "factor": "Planned Maintenance Event",
            "evidence": "Maintenance event flagged for today",
            "recommendation": (
                "Ensure production plan accounts for reduced capacity during maintenance window. "
                "Pre-position ore stockpile to buffer downstream processing. "
                "Stagger equipment servicing to minimise simultaneous downtime."
            ),
        })

    # ── Overall shortfall ─────────────────────────────────────────────────────
    if shortfall_pct >= T["shortfall_critical"]:
        recs.append({
            "severity": "CRITICAL",
            "factor": "Predicted Production Shortfall",
            "evidence": f"{shortfall_pct:.1f}% shortfall predicted — trigger recovery plan",
            "recommendation": (
                "Activate high-risk production recovery plan. "
                "Increase shift length or add night shifts where feasible. "
                "Prioritise highest-grade ore blocks to maintain revenue despite volume shortfall. "
                "Report to MOIL Operations Control Centre."
            ),
        })
    elif shortfall_pct >= T["shortfall_high"]:
        recs.append({
            "severity": "HIGH",
            "factor": "Production Shortfall Risk",
            "evidence": f"{shortfall_pct:.1f}% shortfall predicted",
            "recommendation": (
                "Review blasting and loading sequence for acceleration opportunities. "
                "Increase ore haulage frequency. Alert processing plant to expect reduced feed."
            ),
        })
    elif shortfall_pct >= T["shortfall_medium"]:
        recs.append({
            "severity": "MEDIUM",
            "factor": "Production Shortfall Risk",
            "evidence": f"{shortfall_pct:.1f}% shortfall predicted",
            "recommendation": (
                "Monitor real-time production pace against hourly target. "
                "Optimise truck dispatch cycles. Flag for evening production review meeting."
            ),
        })
    else:
        recs.append({
            "severity": "LOW",
            "factor": "Production On Track",
            "evidence": f"{shortfall_pct:.1f}% below plan — within acceptable range",
            "recommendation": (
                "Continue current production schedule. "
                "Maintain equipment preventive maintenance programme. "
                "Log any near-miss incidents for continuous improvement."
            ),
        })

    # ── Compound risk ─────────────────────────────────────────────────────────
    n_high = sum(1 for r in recs if r["severity"] in ("HIGH", "CRITICAL"))
    if n_high >= 3:
        recs.append({
            "severity": "CRITICAL",
            "factor": "Multiple High-Risk Factors",
            "evidence": f"{n_high} simultaneous risk factors detected",
            "recommendation": (
                "Multiple constraints are simultaneously elevated. "
                "Call emergency production coordination meeting. "
                "Allocate extra resources and escalate to senior management. "
                "Invoke the Mine Emergency Production Recovery Protocol."
            ),
        })

    # Sort by severity
    order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
    recs.sort(key=lambda r: order.get(r["severity"], 5))

    # Add icon
    for r in recs:
        r["icon"] = _badge(r["severity"])

    return recs


def generate_mine_summary(ops_df: pd.DataFrame, predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate mine-level risk summary for the alert table.
    """
    rows = []
    for mine in ops_df["mine"].unique():
        mine_pred = predictions_df[predictions_df["mine"] == mine]
        if mine_pred.empty:
            continue

        latest = mine_pred.sort_values("date").iloc[-1]
        recs = generate_recommendations(
            mine=mine,
            planned_tpd=latest.get("planned_tpd", 1000),
            predicted_tpd=latest.get("predicted_tpd", 900),
            shortfall_pct=latest.get("shortfall_pct", 10),
            downtime_hrs=latest.get("equipment_downtime_hrs", 2),
            blast_delay_hrs=latest.get("blasting_delay_hrs", 1),
            rainfall_mm=latest.get("rainfall_mm", 5),
            equip_avail_pct=latest.get("equipment_avail_pct", 85),
            maintenance_event=latest.get("maintenance_event", 0),
        )
        top_rec = recs[0] if recs else {}
        rows.append({
            "Mine":             mine,
            "Risk Level":       latest.get("risk_label", "LOW"),
            "Expected Shortfall (t)": latest.get("shortfall_tpd", 0),
            "Shortfall %":      latest.get("shortfall_pct", 0),
            "Main Cause":       top_rec.get("factor", "—"),
            "Recommended Action": top_rec.get("recommendation", "Continue monitoring")[:120] + "…",
        })

    return pd.DataFrame(rows)
