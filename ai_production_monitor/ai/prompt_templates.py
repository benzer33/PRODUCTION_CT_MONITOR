"""
ai/prompt_templates.py
Prompt templates for the Anthropic Claude integration.

Design principle
----------------
Raw video is NEVER sent to the API — only structured statistics (numbers,
percentages, timestamps).  This keeps cost low, protects privacy, and
satisfies typical factory data-governance requirements.

Templates
---------
- build_session_analysis_prompt : comprehensive session summary analysis
- build_zone_improvement_prompt : targeted advice for a single bottleneck zone
- build_quick_insight_prompt    : short real-time insight for single cycle
- build_summary_prompt          : Thai-language AI Summary Screen prompt
"""

from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_pct(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.1f}%"


def _fmt_sec(v: float) -> str:
    return f"{v:.2f}s"


# ---------------------------------------------------------------------------
# Session Analysis Prompt (legacy / English)
# ---------------------------------------------------------------------------

def build_session_analysis_prompt(
    station_name: str,
    session_stats: dict[str, Any],
    golden_standard: dict[str, Any],
    zone_names: dict[int, str],
    alert_history: list[dict],
    operator_name: str | None = None,
) -> str:
    """
    Build a detailed analysis prompt for a completed monitoring session.

    Parameters
    ----------
    station_name    : human-readable station/line name
    session_stats   : output of DatabaseManager.get_session_stats()
    golden_standard : {zone_id: standard_sec, "total": total_sec}
    zone_names      : {zone_id: "Zone 1 — Pick Part A", ...}
    alert_history   : list of alert dicts from AlertManager
    operator_name   : optional operator identifier

    Returns
    -------
    Full prompt string ready to send to Claude
    """
    op_str = f"Operator: {operator_name}" if operator_name else "Operator: anonymous"
    total   = session_stats.get("total_cycles", 0)
    passed  = session_stats.get("pass_cycles", 0)
    failed  = session_stats.get("fail_cycles", 0)
    seq_err = session_stats.get("seq_error_cycles", 0)
    avg_t   = session_stats.get("avg_cycle_time", 0)
    min_t   = session_stats.get("min_cycle_time", 0)
    max_t   = session_stats.get("max_cycle_time", 0)
    std_t   = session_stats.get("stdev_cycle_time", 0)
    avg_dev = session_stats.get("avg_deviation_pct", 0)
    times   = session_stats.get("cycle_times", [])
    zone_avgs = session_stats.get("zone_avg_times", {})

    std_total = golden_standard.get("total", 0)

    zone_rows = []
    for zid_str, avg in zone_avgs.items():
        zid  = int(zid_str)
        name = zone_names.get(zid, f"Zone {zid}")
        std  = golden_standard.get(str(zid), golden_standard.get(zid, 0))
        dev  = ((avg - std) / std * 100) if std > 0 else 0
        zone_rows.append(
            f"  - {name}: avg={_fmt_sec(avg)}, standard={_fmt_sec(std)}, "
            f"deviation={_fmt_pct(dev)}"
        )
    zone_table = "\n".join(zone_rows) if zone_rows else "  (no zone data)"

    alert_counts: dict[str, int] = {}
    for a in alert_history:
        key = a.get("type", "UNKNOWN")
        alert_counts[key] = alert_counts.get(key, 0) + 1
    alert_summary = json.dumps(alert_counts, indent=2) if alert_counts else "None"

    times_str = ", ".join(_fmt_sec(t) for t in times[:30])
    if len(times) > 30:
        times_str += f" ... ({len(times) - 30} more)"

    lines = [
        "You are an industrial engineering assistant specialised in manufacturing process optimisation.",
        "Analyse the following production monitoring session data and provide:",
        "1. A concise executive summary (2-3 sentences)",
        "2. Key performance highlights (what went well)",
        "3. Identified bottlenecks or problem areas with specific data evidence",
        "4. Concrete, actionable improvement recommendations (prioritised by impact)",
        "5. Estimated efficiency gain if recommendations are followed",
        "",
        "IMPORTANT: Be specific with numbers. Reference the zone names and times provided.",
        "Use clear, professional language suitable for a factory floor supervisor.",
        "Do NOT invent data not present below.",
        "",
        "=== SESSION DATA ===",
        f"Station : {station_name}",
        op_str,
        "",
        "Cycle Summary:",
        f"  Total cycles     : {total}",
        f"  Pass             : {passed}  ({passed/max(total, 1)*100:.0f}%)",
        f"  Fail (slow)      : {failed}",
        f"  Sequence errors  : {seq_err}",
        "",
        "Cycle Time Statistics (seconds):",
        f"  Average          : {_fmt_sec(avg_t)}",
        f"  Min              : {_fmt_sec(min_t)}",
        f"  Max              : {_fmt_sec(max_t)}",
        f"  Std deviation    : {_fmt_sec(std_t)}",
        f"  Standard (golden): {_fmt_sec(std_total)}",
        f"  Avg deviation    : {_fmt_pct(avg_dev)}",
        "",
        "Per-Zone Average Times vs Standard:",
        zone_table,
        "",
        "Cycle Time Series:",
        f"  [{times_str}]",
        "",
        "Alert History:",
        alert_summary,
        "===================",
        "",
        "Respond in clear paragraphs with section headers. Keep total response under 500 words.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Zone-specific improvement prompt
# ---------------------------------------------------------------------------

def build_zone_improvement_prompt(
    zone_name: str,
    zone_times: list[float],
    standard_time: float,
    sequence_errors: list[str],
) -> str:
    """Targeted prompt asking for improvement suggestions for one bottleneck zone."""
    avg = sum(zone_times) / len(zone_times) if zone_times else 0
    dev = ((avg - standard_time) / standard_time * 100) if standard_time > 0 else 0
    over_count = sum(1 for t in zone_times if t > standard_time * 1.1)

    lines = [
        "You are a lean manufacturing consultant.",
        "One specific work zone is consistently slow. Provide targeted improvement recommendations.",
        "",
        f"Zone: {zone_name}",
        f"Standard time    : {_fmt_sec(standard_time)}",
        f"Average actual   : {_fmt_sec(avg)}",
        f"Average deviation: {_fmt_pct(dev)}",
        f"Times over 110% of standard: {over_count}/{len(zone_times)} cycles",
        f"All times (s)    : {[round(t, 2) for t in zone_times[:20]]}",
        f"Sequence errors  : {sequence_errors[:5] if sequence_errors else 'None'}",
        "",
        "Give 3-5 specific, actionable recommendations to reduce time in this zone.",
        "Consider: ergonomics, tooling layout, part presentation, training gaps, fixture design.",
        "Keep response under 200 words.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick insight (single cycle, real-time feedback)
# ---------------------------------------------------------------------------

def build_quick_insight_prompt(
    cycle_number: int,
    cycle_time: float,
    standard_time: float,
    zone_times: dict[str, float],
    zone_standards: dict[str, float],
    sequence_errors: list[str],
    dtw_score: float,
) -> str:
    """Short prompt for real-time single-cycle feedback."""
    dev = ((cycle_time - standard_time) / standard_time * 100) if standard_time > 0 else 0

    zone_lines = []
    for zid, t in zone_times.items():
        std  = zone_standards.get(zid, 0)
        zdev = ((t - std) / std * 100) if std > 0 else 0
        zone_lines.append(f"  Zone {zid}: {_fmt_sec(t)} (std {_fmt_sec(std)}, {_fmt_pct(zdev)})")

    lines = [
        "Provide a one-paragraph (max 80 words) coaching comment for this production cycle.",
        "Be encouraging but specific about what to improve.",
        "",
        f"Cycle #{cycle_number}:",
        f"  Total time     : {_fmt_sec(cycle_time)} (standard {_fmt_sec(standard_time)}, {_fmt_pct(dev)})",
        f"  DTW similarity : {dtw_score:.0f}/100",
        "  Zone breakdown :",
    ] + zone_lines + [
        f"  Sequence errors: {', '.join(sequence_errors) if sequence_errors else 'None'}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# build_summary_prompt — Thai output for AI Summary Screen
# ---------------------------------------------------------------------------

def build_summary_prompt(session_data: dict) -> str:
    """
    แปลง session_data (output จาก DatabaseManager.get_session_full_data())
    เป็น text prompt ที่ส่งให้ Anthropic API

    ข้อมูลที่ส่ง: สถิติ cycle, deviation, alert counts, sequence violations
    ไม่ส่ง: รูปภาพ, วิดีโอ, trajectory raw data
    """
    stats   = session_data.get("stats", {})
    a_summ  = session_data.get("alert_summary", {})
    v_summ  = session_data.get("violation_summary", {})
    session = session_data.get("session")
    cycles  = session_data.get("cycles", [])

    station_name  = getattr(session, "station_id", "N/A") if session else "N/A"
    operator_name = getattr(session, "operator_name", None) if session else None
    op_str = f"ผู้ปฏิบัติงาน: {operator_name}" if operator_name else "ผู้ปฏิบัติงาน: ไม่ระบุ"

    total    = stats.get("total_cycles", 0)
    passed   = stats.get("pass_cycles", 0)
    failed   = stats.get("fail_cycles", 0)
    seq_err  = stats.get("seq_error_cycles", 0)
    avg_t    = stats.get("avg_cycle_time", 0.0)
    min_t    = stats.get("min_cycle_time", 0.0)
    max_t    = stats.get("max_cycle_time", 0.0)
    std_dev  = stats.get("stdev_cycle_time", 0.0)
    avg_dev  = stats.get("avg_deviation_pct", 0.0)
    zone_avgs = stats.get("zone_avg_times", {})

    zone_rows: list[str] = []
    for zid_str, avg in zone_avgs.items():
        zone_rows.append(f"  โซน {zid_str}: เฉลี่ย {_fmt_sec(avg)}")
    zone_table = "\n".join(zone_rows) if zone_rows else "  (ไม่มีข้อมูล zone)"

    total_alerts   = a_summ.get("total", 0)
    warn_count     = a_summ.get("warnings", 0)
    crit_count     = a_summ.get("criticals", 0)
    by_zone_alerts = a_summ.get("by_zone", {})
    alert_zone_rows = "\n".join(
        f"    โซน {z}: {cnt} ครั้ง" for z, cnt in by_zone_alerts.items()
    ) if by_zone_alerts else "    ไม่มี"

    total_viols = v_summ.get("total", 0)
    skip_cnt    = v_summ.get("skip_count", 0)
    oor_cnt     = v_summ.get("out_of_order_count", 0)
    rep_cnt     = v_summ.get("repeat_count", 0)
    aff_cycles  = v_summ.get("affected_cycles", 0)

    completed = [c for c in cycles if c.status != "in_progress"]
    cycle_rows: list[str] = []
    for i, c in enumerate(completed[:30], 1):
        t     = c.cycle_time_sec or 0.0
        mark  = "V" if c.status == "pass" else "X"
        dev_s = _fmt_pct(c.deviation_pct) if c.deviation_pct is not None else "N/A"
        cycle_rows.append(f"  รอบ {i:2d}: {_fmt_sec(t)}  dev={dev_s}  {mark}  {c.status}")
    if len(completed) > 30:
        cycle_rows.append(f"  ... (และอีก {len(completed) - 30} รอบ)")
    cycle_table = "\n".join(cycle_rows) if cycle_rows else "  (ไม่มี cycle ที่เสร็จสมบูรณ์)"

    output_lines = [
        "คุณเป็นที่ปรึกษาด้านวิศวกรรมอุตสาหการและการปรับปรุงกระบวนการผลิต",
        "โปรดวิเคราะห์ข้อมูลการ monitoring การผลิตต่อไปนี้และตอบเป็นภาษาไทยทั้งหมด",
        "",
        "โครงสร้างคำตอบ (ใช้หัวข้อตามนี้):",
        "## 1. ภาพรวมประสิทธิภาพ",
        "## 2. จุดที่มีปัญหาบ่อยที่สุด / รุนแรงที่สุด",
        "## 3. สาเหตุที่เป็นไปได้",
        "## 4. ข้อเสนอแนะที่เป็นรูปธรรม (เรียงตามความสำคัญ)",
        "## 5. สรุปและเป้าหมายที่แนะนำ",
        "",
        "กฎสำคัญ:",
        "- อ้างอิงตัวเลขจากข้อมูลด้านล่างทุกครั้งที่เป็นไปได้",
        "- ใช้ภาษาที่หัวหน้าไลน์ผลิตที่ไม่ใช่สายเทคนิคเข้าใจได้",
        "- ห้ามสร้างข้อมูลที่ไม่มีในข้อมูลด้านล่าง",
        "- ความยาวคำตอบรวม: 400-600 คำ",
        "",
        "=== ข้อมูล SESSION ===",
        f"สถานี : {station_name}",
        op_str,
        "",
        "สรุปภาพรวม:",
        f"  รอบทั้งหมด       : {total} รอบ",
        f"  ผ่านมาตรฐาน      : {passed} รอบ ({passed / max(total, 1) * 100:.0f}%)",
        f"  ไม่ผ่าน (ช้าเกิน) : {failed} รอบ",
        f"  ผิดลำดับขั้นตอน  : {seq_err} รอบ",
        "",
        "สถิติเวลา cycle (วินาที):",
        f"  เฉลี่ย           : {_fmt_sec(avg_t)}",
        f"  ต่ำสุด           : {_fmt_sec(min_t)}",
        f"  สูงสุด           : {_fmt_sec(max_t)}",
        f"  ค่าเบี่ยงเบน     : {_fmt_sec(std_dev)}",
        f"  เบี่ยงเบนจาก standard เฉลี่ย: {_fmt_pct(avg_dev)}",
        "",
        "เวลาเฉลี่ยต่อโซน:",
        zone_table,
        "",
        "การแจ้งเตือน (Alert):",
        f"  รวมทั้งหมด       : {total_alerts} ครั้ง",
        f"  ระดับ WARNING    : {warn_count} ครั้ง",
        f"  ระดับ CRITICAL   : {crit_count} ครั้ง",
        "  แยกตามโซน:",
        alert_zone_rows,
        "",
        "การละเมิดลำดับขั้นตอน (Sequence Violation):",
        f"  รวม              : {total_viols} ครั้ง ใน {aff_cycles} รอบ",
        f"  ข้ามขั้น (SKIP)  : {skip_cnt} ครั้ง",
        f"  สลับลำดับ        : {oor_cnt} ครั้ง",
        f"  ทำซ้ำ (REPEAT)   : {rep_cnt} ครั้ง",
        "",
        "รายละเอียดแต่ละรอบ:",
        cycle_table,
        "=====================",
        "",
        "โปรดวิเคราะห์และให้คำแนะนำตามโครงสร้างที่กำหนด",
    ]
    return "\n".join(output_lines)
