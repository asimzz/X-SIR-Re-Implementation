#!/bin/bash

# Exit on error or unset variable
set -e
set -u

echo "🔍 Computing z-scores for human (non-watermarked) data..."
bash compute_human_zscores.sh

echo "🧬 Generating model outputs with watermark and applying translation attacks..."
bash generate_with_watermark_translate.sh

echo "✅ All processes completed successfully."
