#!/usr/bin/env python3
"""
Pipeline.py  —  Unified STS Computer-Vision Pipeline Runner
════════════════════════════════════════════════════════════
Runs the full 4-stage pipeline in order:

  Stage 0  Yolo_Crop.py            crop bounding boxes + build Stage 1 dataset
  Stage 1  Object_Detection.py     train YOLOv8s generic detector
  Stage 2  Object_Classifier.py    train EfficientNet-B0 specific classifier
  Stage 3  Eval_ComputerVision.py  evaluate both models

SKIP / RESUME LOGIC
  Stage 0  Skipped if BOTH of these exist:
             output/stage1_dataset/data.yaml
             output/cropped_dataset/  (with at least one sub-folder)
  Stage 1  Skipped if runs/detect/sts_detector/weights/best.pt exists.
           Training resumes from last.pt automatically (handled internally
           by Object_Detection.py).
  Stage 2  Skipped if output/stage2_checkpoints/best.pt exists.
           Training resumes from last.pt automatically (handled internally
           by Object_Classifier.py).
  Stage 3  Always re-runs (evaluation is fast and idempotent).

STATE TRACKING
  pipeline_state.json  written/updated after every stage (JSON, project root)
  logs/pipeline.log    full stdout + stderr captured from every stage

FORCE FLAGS
  --force-crop       Re-run Stage 0 even if cropped_dataset/ already exists
  --force-detect     Re-run Stage 1 even if detector best.pt exists
  --force-classify   Re-run Stage 2 even if classifier best.pt exists

OTHER FLAGS
  --skip-eval        Skip Stage 3 entirely
  --device  STR      Forwarded to Eval_ComputerVision.py  (e.g. cpu, 0, 0,1)
  --topn    INT      Forwarded to Eval_ComputerVision.py  (default in script: 30)
  --n-samples INT    Forwarded to Eval_ComputerVision.py  (default in script: 16)

FAILURE BEHAVIOUR
  Any stage that exits with a non-zero return code immediately aborts the
  entire pipeline.  The failure is recorded in pipeline_state.json and
  logs/pipeline.log, and this process exits with the same return code.
  On re-submission, stages whose outputs already exist are skipped
  automatically so training is never repeated from scratch.

USAGE
  python Pipeline.py
  python Pipeline.py --force-detect --force-classify
  python Pipeline.py --skip-eval
  python Pipeline.py --device 0 --topn 20 --n-samples 32
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────── PATHS ──────────────────────────────────

# Resolve relative to this script's own directory so the pipeline works
# no matter what the current working directory is when it is launched.
ROOT_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "output"

# ── Stage-completion sentinels ────────────────────────────────────────────────
CROP_YAML     = OUTPUT_DIR / "stage1_dataset" / "data.yaml"
CROP_DIR      = OUTPUT_DIR / "cropped_dataset"
DETECT_BEST   = ROOT_DIR  / "runs" / "detect" / "sts_detector" / "weights" / "best.pt"
CLASSIFY_BEST = OUTPUT_DIR / "stage2_checkpoints" / "best.pt"

# ── Logging / state ───────────────────────────────────────────────────────────
STATE_FILE   = ROOT_DIR / "pipeline_state.json"
LOG_DIR      = ROOT_DIR / "logs"
PIPELINE_LOG = LOG_DIR  / "pipeline.log"

# ── Stage script paths ────────────────────────────────────────────────────────
SCRIPTS: Dict[str, Path] = {
    "crop":     ROOT_DIR / "Yolo_Crop.py",
    "detect":   ROOT_DIR / "Object_Detection.py",
    "classify": ROOT_DIR / "Object_Classifier.py",
    "eval":     ROOT_DIR / "Eval_ComputerVision.py",
}

STAGE_LABELS: Dict[str, str] = {
    "crop":     "Stage 0 — Yolo_Crop          (cropping + dataset build)",
    "detect":   "Stage 1 — Object_Detection   (YOLOv8s generic detector) ",
    "classify": "Stage 2 — Object_Classifier  (EfficientNet-B0 classifier)",
    "eval":     "Stage 3 — Eval_ComputerVision (joint evaluator)          ",
}

_WIDTH = 70   # visual separator width


# ─────────────────────────────────── HELPERS ─────────────────────────────────

def _ensure_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _sep(char: str = "═") -> str:
    return char * _WIDTH


def _log(msg: str, echo: bool = True) -> None:
    """Append a timestamped line to PIPELINE_LOG and optionally to stdout."""
    line = f"[{_ts()}] {msg}"
    if echo:
        print(line, flush=True)
    with open(PIPELINE_LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ─────────────────────────────── STATE FILE ──────────────────────────────────

def _load_state() -> Dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            _log("  ⚠  pipeline_state.json is unreadable — starting with empty state.")
    return {}


def _save_state(state: Dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _record(
    state: Dict,
    stage: str,
    status: str,
    elapsed: float,
    returncode: int,
) -> None:
    """Persist one stage result to disk immediately."""
    state[stage] = {
        "status":     status,   # "success" | "failed" | "skipped"
        "timestamp":  _ts(),
        "elapsed_s":  round(elapsed, 1),
        "returncode": returncode,
    }
    _save_state(state)


# ──────────────────────────────── SKIP LOGIC ─────────────────────────────────

def _crop_complete() -> bool:
    """True when Yolo_Crop.py outputs are fully present."""
    if not CROP_YAML.exists():
        return False
    if not CROP_DIR.is_dir():
        return False
    return any(d.is_dir() for d in CROP_DIR.iterdir())


def _detect_complete() -> bool:
    return DETECT_BEST.exists()


def _classify_complete() -> bool:
    return CLASSIFY_BEST.exists()


def _should_run(
    stage: str,
    force: bool,
    state: Dict,
) -> Tuple[bool, str]:
    """
    Return (should_run, human_readable_reason).

    Decision priority
    -----------------
    1. CLI force flag            → always run
    2. File-system sentinel      → skip if outputs already exist
    3. eval stage                → always run
    4. Previous failure in state → re-run so the user gets a fresh attempt
    5. Default                   → run (outputs not found)
    """
    if force:
        return True, "forced via CLI flag"

    if stage == "crop" and _crop_complete():
        return False, (
            "output/stage1_dataset/data.yaml and "
            "output/cropped_dataset/ already exist"
        )
    if stage == "detect" and _detect_complete():
        return False, "runs/detect/sts_detector/weights/best.pt already exists"
    if stage == "classify" and _classify_complete():
        return False, "output/stage2_checkpoints/best.pt already exists"
    if stage == "eval":
        return True, "evaluation always re-runs"

    prior = state.get(stage, {})
    if prior.get("status") == "failed":
        ts = prior.get("timestamp", "unknown time")
        return True, f"re-running after previous failure at {ts}"

    return True, "outputs not found — running fresh"


# ─────────────────────────────── SUBPROCESS ──────────────────────────────────

def _run_stage(
    stage: str,
    extra_args: Optional[List[str]] = None,
) -> Tuple[int, float]:
    """
    Run *stage* as a child process.

    * stdout and stderr are merged and streamed line-by-line to the
      terminal AND appended to PIPELINE_LOG simultaneously.
    * Returns (returncode, elapsed_seconds).
    * Compatible with Python 3.9 on both Windows and Linux.
    """
    cmd: List[str] = [sys.executable, str(SCRIPTS[stage])]
    if extra_args:
        cmd.extend(extra_args)

    _log(f"  Command : {' '.join(cmd)}")

    # Write a section header to the log only (not terminal — too noisy)
    with open(PIPELINE_LOG, "a", encoding="utf-8") as fh:
        fh.write(f"\n{_sep('-')}\n")
        fh.write(f"  SUBPROCESS OUTPUT — {stage}   started {_ts()}\n")
        fh.write(f"{_sep('-')}\n\n")

    t0 = time.time()

    # stdout=PIPE + stderr=STDOUT → single merged stream, line-buffered.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,              # line-buffered (meaningful for text=True)
        cwd=str(ROOT_DIR),
    )

    assert proc.stdout is not None  # guaranteed by stdout=PIPE

    with open(PIPELINE_LOG, "a", encoding="utf-8") as fh:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            fh.write(line)
            fh.flush()

    proc.wait()
    return proc.returncode, time.time() - t0


# ─────────────────────────────── SUMMARY TABLE ───────────────────────────────

def _print_summary(rows: List[Dict], total_elapsed: float) -> None:
    _log("")
    _log(_sep())
    _log("  PIPELINE SUMMARY")
    _log(_sep())

    w_stage   = max(len(r["stage"])   for r in rows) + 2
    w_status  = max(len(r["status"])  for r in rows) + 2
    w_elapsed = max(len(r["elapsed"]) for r in rows) + 2

    _log(
        f"  {'Stage':<{w_stage}}"
        f"  {'Status':<{w_status}}"
        f"  {'Time':<{w_elapsed}}"
        f"  Note"
    )
    _log(_sep("─"))

    for r in rows:
        note = f"({r['reason']})" if r["reason"] else ""
        _log(
            f"  {r['stage']:<{w_stage}}"
            f"  {r['status']:<{w_status}}"
            f"  {r['elapsed']:<{w_elapsed}}"
            f"  {note}"
        )

    _log(_sep("─"))
    hours, rem = divmod(int(total_elapsed), 3600)
    mins, secs = divmod(rem, 60)
    _log(f"  Total elapsed : {hours:02d}h {mins:02d}m {secs:02d}s")
    _log(_sep())


# ─────────────────────────────────── MAIN ────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Unified STS CV pipeline: "
            "Crop → Detect → Classify → Eval"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Force flags ───────────────────────────────────────────────────────────
    ap.add_argument(
        "--force-crop", action="store_true",
        help="Re-run Stage 0 even if cropped_dataset/ already exists",
    )
    ap.add_argument(
        "--force-detect", action="store_true",
        help="Re-run Stage 1 even if detector best.pt already exists",
    )
    ap.add_argument(
        "--force-classify", action="store_true",
        help="Re-run Stage 2 even if classifier best.pt already exists",
    )

    # ── Control flags ─────────────────────────────────────────────────────────
    ap.add_argument(
        "--skip-eval", action="store_true",
        help="Skip Stage 3 (evaluation) entirely",
    )

    # ── Eval pass-through arguments ───────────────────────────────────────────
    ap.add_argument(
        "--device", type=str, default=None,
        help="Device forwarded to Eval_ComputerVision.py (e.g. cpu, 0, 0,1)",
    )
    ap.add_argument(
        "--topn", type=int, default=None,
        help="--topn forwarded to Eval_ComputerVision.py (default in script: 30)",
    )
    ap.add_argument(
        "--n-samples", type=int, default=None, dest="n_samples",
        help="--n-samples forwarded to Eval_ComputerVision.py (default: 16)",
    )

    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    _ensure_dirs()

    state            = _load_state()
    t_pipeline_start = time.time()

    # ── Banner ────────────────────────────────────────────────────────────────
    _log(_sep())
    _log("  STS CV PIPELINE — UNIFIED RUNNER")
    _log(_sep())
    _log(f"  Python        : {sys.version.split()[0]}")
    _log(f"  Root dir      : {ROOT_DIR}")
    _log(f"  State file    : {STATE_FILE}")
    _log(f"  Log file      : {PIPELINE_LOG}")
    _log(f"  Force crop    : {args.force_crop}")
    _log(f"  Force detect  : {args.force_detect}")
    _log(f"  Force classify: {args.force_classify}")
    _log(f"  Skip eval     : {args.skip_eval}")
    _log(_sep())

    # ── Pre-flight: make sure all scripts exist before starting ───────────────
    missing = [
        str(path)
        for key, path in SCRIPTS.items()
        if not path.exists()
    ]
    if missing:
        for m in missing:
            _log(f"❌  Script not found: {m}")
        _log("  Aborting — fix missing scripts and re-submit.")
        sys.exit(1)

    # ── Build ordered stage list ──────────────────────────────────────────────
    # Each tuple: (stage_key, force_bool, extra_cli_args_or_None)
    stages: List[Tuple[str, bool, Optional[List[str]]]] = [
        ("crop",     args.force_crop,     None),
        ("detect",   args.force_detect,   None),
        ("classify", args.force_classify, None),
    ]

    if not args.skip_eval:
        eval_extra: List[str] = []
        if args.device   is not None:
            eval_extra += ["--device",    args.device]
        if args.topn     is not None:
            eval_extra += ["--topn",      str(args.topn)]
        if args.n_samples is not None:
            eval_extra += ["--n-samples", str(args.n_samples)]
        stages.append(("eval", False, eval_extra if eval_extra else None))

    # ── Execute stages in order ───────────────────────────────────────────────
    summary_rows: List[Dict] = []

    for stage_key, force, extra in stages:
        label          = STAGE_LABELS[stage_key]
        run_flag, why  = _should_run(stage_key, force, state)

        _log("")
        _log(_sep("─"))
        _log(f"  {label}")
        _log(_sep("─"))

        # ── SKIP ─────────────────────────────────────────────────────────────
        if not run_flag:
            _log(f"  ⏭  SKIPPED — {why}")
            _record(state, stage_key, "skipped", 0.0, 0)
            summary_rows.append({
                "stage":   label,
                "status":  "⏭  skipped",
                "elapsed": "—",
                "reason":  why,
            })
            continue

        # ── RUN ──────────────────────────────────────────────────────────────
        _log(f"  ▶  RUNNING — {why}")
        returncode, elapsed = _run_stage(stage_key, extra)

        # ── SUCCESS ──────────────────────────────────────────────────────────
        if returncode == 0:
            _log(f"  ✅  Completed in {elapsed:.1f}s")
            _record(state, stage_key, "success", elapsed, returncode)
            summary_rows.append({
                "stage":   label,
                "status":  "✅  success",
                "elapsed": f"{elapsed:.1f}s",
                "reason":  "",
            })
            continue

        # ── FAILURE ──────────────────────────────────────────────────────────
        _log(
            f"❌  FAILED (exit code {returncode}) after {elapsed:.1f}s"
        )
        _record(state, stage_key, "failed", elapsed, returncode)
        summary_rows.append({
            "stage":   label,
            "status":  "❌  FAILED",
            "elapsed": f"{elapsed:.1f}s",
            "reason":  f"exit code {returncode}",
        })

        _print_summary(summary_rows, time.time() - t_pipeline_start)
        _log(
            f"\n  Pipeline aborted at '{stage_key}'.  "
            "Fix the error above, then re-submit —\n"
            "  stages that already completed will be skipped automatically."
        )
        sys.exit(returncode)

    # ── All stages finished ───────────────────────────────────────────────────
    _print_summary(summary_rows, time.time() - t_pipeline_start)
    _log("\n  Pipeline finished successfully. ✅\n")


if __name__ == "__main__":
    main()
