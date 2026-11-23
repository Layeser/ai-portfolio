"""Share trained weights so anyone can use the model without training it.

    python scripts/weights.py export      # gather the files listed in weights.json into release/,
                                          # write weights.sha256 (committed) and the model card
    python scripts/weights.py upload      # publish release/ (GitHub Release or Hugging Face Hub)
    python scripts/weights.py download    # fetch the files to where the code expects them, verify SHA-256

`weights.json`, next to this script's parent folder:

    {"backend": "github", "github": {"repo": "<owner>/<repo>", "tag": "<release tag>"},
     "files": {"<release name>": "<path in the project>"}}
    {"backend": "huggingface", "repo_id": "<user>/<model>", "private": true, "files": {...}}

* github (default): files are attached to a GitHub Release of the code repository (up to 2 GB per file,
  outside the git history, so the repository stays light). Upload uses the `gh` CLI.
* huggingface: model repository on the Hub. Use "private": true when the weights could reveal private
  training data (e.g. a TF-IDF vocabulary). Upload needs HF_TOKEN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "weights.json").read_text())
BACKEND = CONFIG.get("backend", "github")
RELEASE = ROOT / "release"
CHECKSUMS = ROOT / "weights.sha256"
CARD = ROOT / "MODEL_CARD.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_name(name: str) -> str:
    """GitHub release assets cannot contain '/': tfidf/model.joblib -> tfidf__model.joblib."""
    return name.replace("/", "__") if BACKEND == "github" else name


def export() -> None:
    RELEASE.mkdir(exist_ok=True)
    lines = []
    for name, local in CONFIG["files"].items():
        src = ROOT / local
        if not src.exists():
            sys.exit(f"missing {local}: train the model first")
        dst = RELEASE / asset_name(name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        lines.append(f"{sha256(dst)}  {name}")
    CHECKSUMS.write_text("\n".join(lines) + "\n")
    if CARD.exists():
        shutil.copy2(CARD, RELEASE / "README.md")  # the Hub shows README.md as the model card
    size = sum(f.stat().st_size for f in RELEASE.rglob("*") if f.is_file()) / 1e6
    print(f"release/ ready: {len(lines)} files, {size:.0f} MB; checksums in {CHECKSUMS.name}")


def upload() -> None:
    if BACKEND == "github":
        gh = CONFIG["github"]
        if shutil.which("gh") is None:
            sys.exit("install the GitHub CLI (https://cli.github.com) and run `gh auth login`")
        files = [str(RELEASE / asset_name(name)) for name in CONFIG["files"]]
        exists = subprocess.run(["gh", "release", "view", gh["tag"], "--repo", gh["repo"]],
                                capture_output=True).returncode == 0
        if exists:
            subprocess.run(["gh", "release", "upload", gh["tag"], *files, "--repo", gh["repo"], "--clobber"],
                           check=True)
        else:
            notes = ["--notes-file", str(CARD)] if CARD.exists() else ["--notes", "Model weights"]
            subprocess.run(["gh", "release", "create", gh["tag"], *files, "--repo", gh["repo"],
                            "--title", gh.get("title", gh["tag"]), *notes], check=True)
        print(f"uploaded to https://github.com/{gh['repo']}/releases/tag/{gh['tag']}")
        return

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("set HF_TOKEN (a write token from https://huggingface.co/settings/tokens)")
    api = HfApi(token=token)
    api.create_repo(CONFIG["repo_id"], repo_type="model", private=CONFIG.get("private", False), exist_ok=True)
    api.upload_folder(folder_path=RELEASE, repo_id=CONFIG["repo_id"], repo_type="model")
    print(f"uploaded to https://huggingface.co/{CONFIG['repo_id']}")


def fetch(name: str) -> Path:
    """Download one released file and return its local (cached) path."""
    if BACKEND == "github":
        gh = CONFIG["github"]
        cache = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "model-weights" / gh["tag"]
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / asset_name(name)
        if not target.exists():
            url = f"https://github.com/{gh['repo']}/releases/download/{gh['tag']}/{asset_name(name)}"
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, target.with_suffix(target.suffix + ".part"))
            target.with_suffix(target.suffix + ".part").rename(target)
        return target

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(CONFIG["repo_id"], name, repo_type="model"))


def download() -> None:
    expected = dict(reversed(line.split("  ", 1)) for line in CHECKSUMS.read_text().splitlines() if line)
    for name, local in CONFIG["files"].items():
        cached = fetch(name)
        if sha256(cached) != expected[name]:
            sys.exit(f"checksum mismatch for {name}: refusing to use it")
        target = ROOT / local
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, target)
        print(f"ok  {name} -> {local}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["export", "upload", "download"])
    {"export": export, "upload": upload, "download": download}[parser.parse_args().command]()
