"""Execute training notebooks in order without changing them; executed copies go to <AM_CHECKPOINTS>/logs/.

    python tools/run_notebooks.py esim/01_word2vec.ipynb siamese/train.ipynb esim/02_train.ipynb

Folders follow helpers/paths.py (the AM_* environment variables are passed on to each notebook's kernel).
"""
import os
import sys
import time

import nbformat
from nbclient import NotebookClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(os.environ.get("AM_CHECKPOINTS", os.path.join(ROOT, "checkpoints")), "logs")
os.makedirs(LOGS, exist_ok=True)

for rel in sys.argv[1:]:
    path = os.path.join(ROOT, rel)
    nb = nbformat.read(path, as_version=4)
    start = time.time()
    print(f"[{time.strftime('%H:%M')}] start {rel}", flush=True)
    try:
        NotebookClient(nb, timeout=None, kernel_name="python3",
                       resources={"metadata": {"path": os.path.dirname(path)}}).execute()
    finally:
        nbformat.write(nb, os.path.join(LOGS, rel.replace("/", "__")))
    print(f"[{time.strftime('%H:%M')}] done {rel} ({(time.time() - start) / 60:.0f} min)", flush=True)
