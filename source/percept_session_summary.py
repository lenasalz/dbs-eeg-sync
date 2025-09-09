
#!/usr/bin/env python3
"""
percept_session_summary.py

Parse a Medtronic Percept (PC/RC+) Session Report JSON and produce a compact overview:
- Session start/end, duration
- Battery info
- Active group(s) and stimulation settings (rate, PW, amplitude) per hemisphere
- Presence and counts of BrainSenseTimeDomain and BrainSenseLfp recordings
- For each recording (when available): type, hemisphere, channel, sensing frequency, stim frequency, band power info, duration
- MostRecentInSessionSignalCheck summary (channels, frequencies, PSD length)
Outputs a pretty-printed summary and optional CSV/Excel.

Usage:
  python percept_session_summary.py \
      /path/to/Report_Json_Session_Report_YYYYMMDDTHHMMSS.json \
      [--csv out.csv] [--xlsx out.xlsx]

Arguments:
  json_path         Path to a Medtronic Percept JSON "Session Report" export.
  --csv out.csv     (optional) Export all tables concatenated into a single CSV with a
                    __section__ column (stim, recordings, blocks, signal_checks).
  --xlsx out.xlsx   (optional) Export separate Excel sheets: "stim", "recordings",
                    "blocks", and "signal_checks".

Notes:
  • Columns that are entirely NaN are dropped from all tables.
  • "Blocks" group rows that share the same start_time and type (e.g., bilateral TD pairs).
  • Amplitude is flattened from electrode_state when per-program amplitude is not provided.
  • Durations are backfilled when possible:
      – TimeDomain: duration_s = n_samples / sample_rate_hz
      – LfpBandPower: duration_s = n_points / sensing_freq_hz
  • Console preview shows a balanced sample (first 5 TimeDomain + first 5 LfpBandPower),
    then prints the Blocks and Signal Checks.

Examples:
  # Basic console summary
  python percept_session_summary.py \
      "/path/Report_Json_Session_Report_20240719T121230.json"

  # Save CSV and Excel outputs
  python percept_session_summary.py \
      "/path/Report_Json_Session_Report_20240719T121230.json" \
      --csv outputs/percept_session_summary.csv \
      --xlsx outputs/percept_session_summary.xlsx

This script is defensive against slightly different field names across software versions.
"""

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

try:
    import pandas as pd
except Exception:
    pd = None

def _parse_iso(ts: Optional[str]) -> Optional[dt.datetime]:
    if not ts:
        return None
    try:
        # Python 3.11: dt.datetime.fromisoformat handles Z as well
        return dt.datetime.fromisoformat(ts.replace("Z","+00:00"))
    except Exception:
        try:
            return dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S%z")
        except Exception:
            return None

def _duration(start: Optional[dt.datetime], end: Optional[dt.datetime]) -> Optional[dt.timedelta]:
    if start and end:
        return end - start
    return None

def _td_to_s(td: Optional[dt.timedelta]) -> Optional[float]:
    if td is None: return None
    return td.total_seconds()

def _safe(d: Dict, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def _iter_list(d: Dict, *keys) -> List[Any]:
    val = _safe(d, *keys, default=[])
    return val if isinstance(val, list) else []

# --- Additional helper functions for amplitude, side, etc.
def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _approx_amp_from_electrode_state(electrode_state):
    """Return the first Negative contact's amplitude (mA) if present."""
    try:
        for e in electrode_state or []:
            if (e.get("ElectrodeStateResult") or "").endswith("Negative"):
                amp = e.get("ElectrodeAmplitudeInMilliAmps")
                if amp is not None:
                    return _safe_float(amp)
    except Exception:
        pass
    return None

def _side_from_channel(ch: Optional[str]) -> Optional[str]:
    if not ch:
        return None
    s = str(ch).upper()
    if "_LEFT" in s or s.endswith("LEFT") or " LEFT" in s:
        return "Left"
    if "_RIGHT" in s or s.endswith("RIGHT") or " RIGHT" in s:
        return "Right"
    return None

def session_meta(js: Dict) -> Dict[str, Any]:
    start = _parse_iso(js.get("SessionDate"))
    end = _parse_iso(js.get("SessionEndDate"))
    duration = _duration(start, end)
    battery = _safe(js, "BatteryInformation", "BatteryPercentage")
    est_months = _safe(js, "BatteryInformation", "EstimatedBatteryLifeMonths")
    active_group_name = None
    active_group_id = None

    # Try to find active group from Groups.Final or Groups.Initial
    for phase in ("Final", "Initial"):
        groups = _iter_list(js, "Groups", phase)
        for g in groups:
            if g.get("ActiveGroup"):
                active_group_name = g.get("GroupName")
                active_group_id = g.get("GroupId")
                break
        if active_group_id:
            break

    return {
        "session_start": start.isoformat() if start else None,
        "session_end": end.isoformat() if end else None,
        "session_duration_s": _td_to_s(duration),
        "session_duration_h": (duration.total_seconds()/3600.0) if duration else None,
        "battery_pct": battery,
        "battery_est_months": est_months,
        "active_group_name": active_group_name,
        "active_group_id": active_group_id,
    }

def extract_group_stim(js: Dict) -> List[Dict[str, Any]]:
    """Extract stimulation settings per group / hemisphere from Groups.Final then Groups.Initial."""
    out = []
    def scan(groups: List[Dict[str,Any]], phase: str):
        for g in groups:
            gname = g.get("GroupName")
            gid = g.get("GroupId")
            mode = g.get("Mode")
            active = g.get("ActiveGroup")
            gs = g.get("ProgramSettings") or {}
            # Two possible shapes: {LeftHemisphere:{Programs:[...]}, RightHemisphere:{...}}
            # or a SensingChannel array with per-hemisphere entries including RateInHertz etc.
            # We extract both if present.
            for hemi_key, hemi_label in (("LeftHemisphere","Left"), ("RightHemisphere","Right")):
                hemi = gs.get(hemi_key) or {}
                programs = hemi.get("Programs") or []
                for p in programs:
                    row = {
                        "phase": phase,
                        "group_name": gname,
                        "group_id": gid,
                        "active_group": active,
                        "hemisphere": hemi_label,
                        "program_id": p.get("ProgramId"),
                        "stim_rate_hz": p.get("RateInHertz"),
                        "pulse_width_us": p.get("PulseWidthInMicroSecond"),
                        "amplitude_mA": p.get("AmplitudeInMilliAmps") if p.get("AmplitudeInMilliAmps") is not None else _approx_amp_from_electrode_state(p.get("ElectrodeState")),
                        "upper_limit_mA": p.get("UpperLimitInMilliAmps"),
                        "lower_limit_mA": p.get("LowerLimitInMilliAmps"),
                        "electrode_state": p.get("ElectrodeState"),
                    }
                    out.append(row)
            # SensingChannel entries (often include RateInHertz, PulseWidth, SensingSetup.FrequencyInHertz)
            for sc in gs.get("SensingChannel", []) or []:
                row = {
                    "phase": phase,
                    "group_name": gname,
                    "group_id": gid,
                    "active_group": active,
                    "hemisphere": sc.get("HemisphereLocation"),
                    "program_id": sc.get("ProgramId"),
                    "stim_rate_hz": sc.get("RateInHertz"),
                    "pulse_width_us": sc.get("PulseWidthInMicroSecond"),
                    "amplitude_mA": _approx_amp_from_electrode_state(sc.get("ElectrodeState")),
                    "upper_limit_mA": sc.get("UpperLimitInMilliAmps"),
                    "lower_limit_mA": sc.get("LowerLimitInMilliAmps"),
                    "electrode_state": sc.get("ElectrodeState"),
                    "sensing_channel": sc.get("Channel"),
                    "sensing_freq_hz": _safe(sc,"SensingSetup","FrequencyInHertz"),
                    "sensing_avg_ms": _safe(sc,"SensingSetup","AveragingDurationInMilliSeconds"),
                }
                out.append(row)

    final_groups = _iter_list(js, "Groups", "Final")
    if final_groups:
        scan(final_groups, "Final")
    initial_groups = _iter_list(js, "Groups", "Initial")
    if initial_groups:
        scan(initial_groups, "Initial")
    return out

def extract_signal_checks(js: Dict) -> List[Dict[str,Any]]:
    out = []
    for sc in _iter_list(js, "MostRecentInSessionSignalCheck"):
        row = {
            "channel": sc.get("Channel"),
            "artifact_status": sc.get("ArtifactStatus"),
            "n_freq_bins": len(sc.get("SignalFrequencies") or []),
            "n_psd_vals": len(sc.get("SignalPsdValues") or []),
            "has_peaks": bool(sc.get("PeakFrequencies")),
        }
        out.append(row)
    return out

def extract_brainsense_time_domain(js: Dict) -> List[Dict[str,Any]]:
    # This block tries multiple likely keys, as names vary by export version.
    candidates = [
        ("BrainSenseTimeDomain",),
        ("BrainSense", "TimeDomain"),
        ("TimeDomainData",),
    ]
    td_list = []
    for path in candidates:
        td_list = _iter_list(js, *path)
        if td_list:
            break
    out = []
    for td in td_list:
        start = _parse_iso(td.get("FirstPacketDateTime") or td.get("StartTime") or td.get("DateTime"))
        end   = _parse_iso(td.get("LastPacketDateTime") or td.get("EndTime"))
        dur_s = _td_to_s(_duration(start, end))
        row = {
            "type": "TimeDomain",
            "hemisphere": td.get("HemisphereLocation"),
            "channel": td.get("Channel") or td.get("SensingChannel") or td.get("ElectrodeConfig"),
            "sample_rate_hz": td.get("SampleRateInHz") or td.get("SampleRateHz"),
            "band_pass_hz": td.get("BandPassFilter"),
            "stimulation_status": td.get("StimulationStatus") or td.get("StimStatus"),
            "start_time": start.isoformat() if start else None,
            "end_time": end.isoformat() if end else None,
            "duration_s": dur_s,
            "n_samples": td.get("NumberOfSamples") or td.get("NumberOfPoints"),
        }
        # Backfill duration for TD when possible
        if row.get("duration_s") is None:
            ns = row.get("n_samples")
            sr = row.get("sample_rate_hz")
            if ns is not None and sr:
                try:
                    row["duration_s"] = float(ns) / float(sr)
                except Exception:
                    pass
        out.append(row)
    return out

def extract_brainsense_lfp(js: Dict) -> List[Dict[str,Any]]:
    # LFP band power streams / FFT streams
    candidates = [
        ("BrainSenseLfp",),
        ("BrainSense", "Lfp"),
        ("LfpData",),
    ]
    lfp_list = []
    for path in candidates:
        lfp_list = _iter_list(js, *path)
        if lfp_list:
            break
    out = []
    for l in lfp_list:
        start = _parse_iso(l.get("FirstPacketDateTime") or l.get("StartTime") or l.get("DateTime"))
        end   = _parse_iso(l.get("LastPacketDateTime") or l.get("EndTime"))
        dur_s = _td_to_s(_duration(start, end))
        # Common fields seen in exports
        row = {
            "type": "LfpBandPower",
            "hemisphere": l.get("HemisphereLocation"),
            "channel": l.get("Channel") or l.get("SensingChannel"),
            "sensing_freq_hz": l.get("SampleRateHz") or l.get("SensingFrequencyInHertz") or l.get("UpdateRateHz"),
            "band_center_hz": l.get("CenterFrequency") or l.get("BandCenterFrequency"),
            "band_width_hz": l.get("BandWidth") or l.get("BandWidthHz"),
            "band_low_hz": l.get("LowerCorner") or l.get("BandLowerHz"),
            "band_high_hz": l.get("UpperCorner") or l.get("BandUpperHz"),
            "stimulation_status": l.get("StimulationStatus") or l.get("StimStatus"),
            "start_time": start.isoformat() if start else None,
            "end_time": end.isoformat() if end else None,
            "duration_s": dur_s,
            "n_points": l.get("NumberOfPoints") or l.get("NumberOfSamples"),
            "mean_band_power": l.get("MeanBandPower") or l.get("AverageBandPower"),
            "notes": l.get("Notes"),
        }
        # Backfill duration for LFP band-power when possible
        if row.get("duration_s") is None:
            np_ = row.get("n_points")
            fr = row.get("sensing_freq_hz")
            if np_ is not None and fr:
                try:
                    row["duration_s"] = float(np_) / float(fr)
                except Exception:
                    pass
        out.append(row)
    return out

# --- Grouping function for recording blocks ---
from collections import defaultdict

def group_recording_blocks(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group raw recording rows into blocks by exact start_time and type.
    This collapses bilateral pairs that start at the same time into a single block.
    """
    blocks = defaultdict(list)
    for r in recs:
        key = (r.get("type"), r.get("start_time"))
        blocks[key].append(r)
    out = []
    for (rtype, start), rows in blocks.items():
        chans = [str(x.get("channel")) for x in rows if x.get("channel")]
        sides = sorted({ _side_from_channel(x) or "?" for x in chans })
        end_times = [x.get("end_time") for x in rows if x.get("end_time")]
        # Choose end as the max end_time if present
        end = None
        try:
            if end_times:
                end_dt = max(_parse_iso(t) for t in end_times if t)
                if end_dt:
                    end = end_dt.isoformat()
        except Exception:
            end = end_times[-1] if end_times else None
        # Duration: prefer explicit duration if identical across rows; else take max available
        durs = [x.get("duration_s") for x in rows if x.get("duration_s") is not None]
        dur = None
        if durs:
            try:
                dur = max(float(d) for d in durs)
            except Exception:
                dur = durs[0]
        out.append({
            "type": rtype,
            "start_time": start,
            "end_time": end,
            "duration_s": dur,
            "n_streams": len(rows),
            "channels": ",".join(sorted(chans)),
        })
    # Sort by start_time then type
    out.sort(key=lambda x: (x.get("start_time") or "", x.get("type") or ""))
    return out

def build_overview(js: Dict) -> Dict[str, Any]:
    meta = session_meta(js)
    stim = extract_group_stim(js)
    td = extract_brainsense_time_domain(js)
    lfp = extract_brainsense_lfp(js)
    sig = extract_signal_checks(js)

    recordings = td + lfp
    blocks = group_recording_blocks(recordings)
    overview = {
        "meta": meta,
        "counts": {
            "n_time_domain": len(td),
            "n_lfp": len(lfp),
            "has_time_domain": bool(td),
            "has_lfp": bool(lfp),
            "n_blocks": len(blocks),
        },
        "stim_rows": stim,
        "recordings": recordings,
        "blocks": blocks,
        "signal_checks": sig,
    }
    return overview

def to_tables(overview: Dict[str, Any]) -> Tuple[Optional["pd.DataFrame"], Optional["pd.DataFrame"], Optional["pd.DataFrame"], Optional["pd.DataFrame"]]:
    if pd is None:
        return None, None, None, None
    def drop_all_nan(df):
        if df is None or df.empty:
            return df
        return df.loc[:, [c for c in df.columns if not df[c].isna().all()]]

    stim_df = pd.DataFrame(overview.get("stim_rows") or [])
    rec_df = pd.DataFrame(overview.get("recordings") or [])
    blk_df = pd.DataFrame(overview.get("blocks") or [])
    sig_df = pd.DataFrame(overview.get("signal_checks") or [])
    # Order columns if present
    if not stim_df.empty:
        stim_cols = [c for c in [
            "phase","active_group","group_name","group_id","hemisphere","program_id",
            "stim_rate_hz","pulse_width_us","amplitude_mA","upper_limit_mA","lower_limit_mA",
            "sensing_channel","sensing_freq_hz","sensing_avg_ms","electrode_state"
        ] if c in stim_df.columns]
        stim_df = stim_df[stim_cols]
    if not rec_df.empty:
        rec_cols = [c for c in [
            "type","hemisphere","channel","sensing_freq_hz","sample_rate_hz",
            "band_center_hz","band_width_hz","band_low_hz","band_high_hz","mean_band_power",
            "stimulation_status","start_time","end_time","duration_s","n_samples","n_points","notes"
        ] if c in rec_df.columns]
        rec_df = rec_df[rec_cols]
    if not blk_df.empty:
        blk_cols = [c for c in ["type","start_time","end_time","duration_s","n_streams","channels"] if c in blk_df.columns]
        blk_df = blk_df[blk_cols]
    if not sig_df.empty:
        sig_cols = [c for c in ["channel","artifact_status","n_freq_bins","n_psd_vals","has_peaks"] if c in sig_df.columns]
        sig_df = sig_df[sig_cols]

    # Drop columns that are entirely NaN
    stim_df = drop_all_nan(stim_df)
    rec_df = drop_all_nan(rec_df)
    blk_df = drop_all_nan(blk_df)
    sig_df = drop_all_nan(sig_df)

    return stim_df, rec_df, blk_df, sig_df

def pretty_print(overview: Dict[str, Any]) -> None:
    meta = overview["meta"]
    counts = overview["counts"]
    def _fmt(v, nd=2):
        if v is None: return "—"
        if isinstance(v, float):
            return f"{v:.{nd}f}"
        return str(v)
    print("SESSION")
    print("  Start:          ", meta.get("session_start","—"))
    print("  End:            ", meta.get("session_end","—"))
    print("  Duration (h):   ", _fmt(meta.get("session_duration_h")))
    print("  Battery (%):    ", _fmt(meta.get("battery_pct"),0))
    print("  Est. months:    ", _fmt(meta.get("battery_est_months"),0))
    print("  Active group:   ", meta.get("active_group_name","—"), f"({meta.get('active_group_id','—')})")
    print()
    print("RECORDINGS")
    print("  TimeDomain:     ", counts["n_time_domain"])
    print("  LfpBandPower:   ", counts["n_lfp"])
    print("  Blocks (by start time): ", overview.get("counts",{}).get("n_blocks","—"))
    print("  Blocks group rows that start at the same time (e.g., bilateral TD into one block).")
    print()
    print("HINTS")
    print("  • Use the STIM table to see stimulation and sensing frequencies per hemisphere.")
    print("  • The RECORDINGS table shows per-recording sensing frequency and band-power windows (for LFP).")
    print("  • SIGNAL CHECKS list PSD bin counts per channel within the session.")
    print()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", help="Path to Percept Session Report JSON")
    ap.add_argument("--csv", help="Optional: export overview tables to a single CSV (multi-sheet CSV not supported).")
    ap.add_argument("--xlsx", help="Optional: export overview tables to Excel workbook.")
    args = ap.parse_args()

    p = Path(args.json_path).expanduser()
    with p.open("r", encoding="utf-8") as f:
        js = json.load(f)

    overview = build_overview(js)
    pretty_print(overview)

    # DataFrames if pandas available
    stim_df, rec_df, blk_df, sig_df = to_tables(overview)
    if pd is not None and not (stim_df is None and rec_df is None and blk_df is None and sig_df is None):
        print("TABLES")
        if stim_df is not None and not stim_df.empty:
            print("  STIM (first 10 rows):")
            print(stim_df.head(10).to_string(index=False))
        if rec_df is not None and not rec_df.empty:
            # Show a balanced preview: first 5 TD and first 5 LFP if possible
            print("  RECORDINGS (sample):")
            try:
                td_sample = rec_df[rec_df["type"]=="TimeDomain"].head(5)
                lfp_sample = rec_df[rec_df["type"]=="LfpBandPower"].head(5)
                print("    TimeDomain (first 5):")
                if not td_sample.empty:
                    print(td_sample.to_string(index=False))
                print("    LfpBandPower (first 5):")
                if not lfp_sample.empty:
                    print(lfp_sample.to_string(index=False))
            except Exception:
                print("  RECORDINGS (first 10 rows):")
                print(rec_df.head(10).to_string(index=False))
        if blk_df is not None and not blk_df.empty:
            print("  BLOCKS (first 10 rows):")
            print(blk_df.head(10).to_string(index=False))
        if sig_df is not None and not sig_df.empty:
            print("  SIGNAL_CHECKS:")
            print(sig_df.to_string(index=False))

        # Exports
        if args.csv:
            frames = []
            if not (stim_df is None or stim_df.empty):
                tmp = stim_df.copy(); tmp.insert(0,"__section__","stim"); frames.append(tmp)
            if not (rec_df is None or rec_df.empty):
                tmp = rec_df.copy(); tmp.insert(0,"__section__","recordings"); frames.append(tmp)
            if not (blk_df is None or blk_df.empty):
                tmp = blk_df.copy(); tmp.insert(0,"__section__","blocks"); frames.append(tmp)
            if not (sig_df is None or sig_df.empty):
                tmp = sig_df.copy(); tmp.insert(0,"__section__","signal_checks"); frames.append(tmp)
            if frames:
                pd.concat(frames, ignore_index=True).to_csv(args.csv, index=False)
                print(f"Saved CSV -> {args.csv}")
        if args.xlsx:
            with pd.ExcelWriter(args.xlsx, engine="xlsxwriter") as xw:
                if not (stim_df is None or stim_df.empty):
                    stim_df.to_excel(xw, sheet_name="stim", index=False)
                if not (rec_df is None or rec_df.empty):
                    rec_df.to_excel(xw, sheet_name="recordings", index=False)
                if not (blk_df is None or blk_df.empty):
                    blk_df.to_excel(xw, sheet_name="blocks", index=False)
                if not (sig_df is None or sig_df.empty):
                    sig_df.to_excel(xw, sheet_name="signal_checks", index=False)
            print(f"Saved Excel -> {args.xlsx}")

if __name__ == "__main__":
    main()



