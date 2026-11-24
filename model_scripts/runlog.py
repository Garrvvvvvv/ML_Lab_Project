"""Experiment tracking: every training run is written to results/runs/<id>/ so a model can be traced
back to the exact data (SHA-256 of every capture), settings and code version (git commit) that made it."""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import sklearn

from features import PROJECT_ROOT


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_manifest(pcap_dirs):
    """SHA-256 and size of every capture used (training pool and hold-out)."""
    items = []
    for d in pcap_dirs:
        for p in sorted(Path(d).glob("*")):
            if p.suffix.lower() in (".pcap", ".pcapng"):
                items.append({"file": p.relative_to(PROJECT_ROOT).as_posix(),
                              "sha256": sha256_file(p), "bytes": p.stat().st_size})
    return items


def combined_hash(manifest):
    """One short fingerprint for the whole dataset."""
    h = hashlib.sha256()
    for item in sorted(manifest, key=lambda i: i["file"]):
        h.update((item["file"] + item["sha256"]).encode())
    return h.hexdigest()


def git_state():
    def run(*args):
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=15)
    try:
        commit = run("rev-parse", "HEAD").stdout.strip() or None
        dirty = bool(run("status", "--porcelain").stdout.strip())
        return {"commit": commit, "uncommitted_changes": dirty}
    except Exception:
        return {"commit": None, "uncommitted_changes": None}


def write_run(results_dir, cfg, metrics, manifest, headline):
    """Create results/runs/<timestamp>/ and return (run_id, run_dir)."""
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(results_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "run_id": run_id,
        "time": datetime.now().isoformat(timespec="seconds"),
        "git": git_state(),
        "data_hash": combined_hash(manifest),
        "params": cfg,
        "argv": sys.argv[1:],
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
        "headline_metrics": headline,
    }
    (run_dir / "run.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_id, run_dir, info
