#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline.sh  —  SBATCH wrapper for the unified STS CV pipeline
#
#  Submit with:
#    sbatch Pipeline.sh                         # normal run (auto-resume)
#    sbatch Pipeline.sh --force-detect          # force re-train detector
#    sbatch Pipeline.sh --force-classify        # force re-train classifier
#    sbatch Pipeline.sh --force-crop            # force re-crop dataset
#    sbatch Pipeline.sh --skip-eval             # skip evaluation stage
#    sbatch Pipeline.sh --device 0 --topn 20   # pass args to evaluator
#
#  All arguments after "Pipeline.sh" are forwarded verbatim to Pipeline.py.
#
#  Time budget breakdown (8 h total):
#    Stage 0  Yolo_Crop            ~0.5 h
#    Stage 1  Object_Detection     ~3.0 h   (100 epochs, YOLOv8s, 2× V100)
#    Stage 2  Object_Classifier    ~2.5 h   (100 epochs, EfficientNet-B0)
#    Stage 3  Eval_ComputerVision  ~0.5 h
#  Adjust --time if your dataset is significantly larger than expected.
# ══════════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=sts_pipeline
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:V100:2
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err

# ---------------------------------------------------------------------------
# Safety: abort the job on any unhandled error or undefined variable
# ---------------------------------------------------------------------------
set -euo pipefail

# ---------------------------------------------------------------------------
# Create logs/ before SLURM tries to open the log files
# ---------------------------------------------------------------------------
mkdir -p logs

# ---------------------------------------------------------------------------
# (Optional) activate your conda / virtualenv here, e.g.:
#   source activate sts_cv
#   source /path/to/venv/bin/activate
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Print environment info for easier debugging of failed jobs
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  STS CV PIPELINE — SLURM JOB"
echo "============================================================"
echo "  Job ID       : ${SLURM_JOB_ID}"
echo "  Node         : $(hostname)"
echo "  GPUs         : ${CUDA_VISIBLE_DEVICES:-not set}"
echo "  Python       : $(python --version 2>&1)"
echo "  Working dir  : $(pwd)"
echo "  Arguments    : $*"
echo "  Started at   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Launch the pipeline — all CLI args ($@) are forwarded to Pipeline.py
# ---------------------------------------------------------------------------
python Pipeline.py "$@"

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  Job finished at : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Exit code       : ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
