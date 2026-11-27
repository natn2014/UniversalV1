#!/usr/bin/env bash
# Opens the D92 photos in X-AnyLabeling with the fixed class list.
# Polygons are saved as <photo>.json next to each photo (labelme format),
# which tools/labelme_to_yoloseg.py turns into a YOLO-seg dataset.
#   tools/label_d92.sh [FOLDER]      (default D92-Image/D92)
# X-AnyLabeling lives in its own venv (~/xanylabeling-venv) to keep the app's .venv clean:
#   python3 -m venv ~/xanylabeling-venv && ~/xanylabeling-venv/bin/pip install "x-anylabeling-cvhub[cpu]"
cd "$(dirname "$0")/.." || exit 1
FOLDER="${1:-D92-Image/D92}"
exec ~/xanylabeling-venv/bin/xanylabeling --filename "$FOLDER" --labels glass_front,glass_back,sponge \
    --validatelabel exact --nodata --autosave --no-auto-update-check
