"""Build the KYC training manifests for Asmita (owner: Saloni).

    python data/scripts/prepare_kyc_data.py --oulu data/raw/oulu_npu --replay data/raw/replay_attack \
                                            --midv data/raw/midv
    python data/scripts/prepare_kyc_data.py            # no sources present -> writes empty manifests (headers only)

Outputs: data/processed/liveness_manifest.csv, data/processed/document_manifest.csv

Rules
  * We only index files that really exist. We never fabricate document images or videos.
  * Subject IDs are namespaced by dataset (e.g. "oulu_npu:user_07") and preserved, so splits are SUBJECT-DISJOINT.
  * Missing information stays missing: tampering_label is "UNAVAILABLE" unless the source provides it. MIDV
    does not by itself supply forgery labels. If you add a forgery dataset or your own labelled tampered
    set, register it as a new source rather than inferring labels.
  * Filename conventions below are parsed from the datasets' published naming schemes. VERIFY them against your
    downloaded copy; files that do not match are skipped and counted, never guessed.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from common import DEFAULT_SEED, PROCESSED_DIR, ensure_dirs, hash_split

LIVENESS_COLS = ["video_id", "source_dataset", "subject_id", "label", "attack_type", "split", "storage_uri",
                 "file_available", "data_origin"]
DOCUMENT_COLS = ["document_id", "source_dataset", "source_subject_id", "document_type", "storage_uri",
                 "tampering_label", "annotation_uri", "split", "data_origin"]
VIDEO_EXT = {".avi", ".mov", ".mp4", ".mkv"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def oulu_rows(root: Path) -> tuple[list[dict], int]:
    """OULU-NPU: files named Phone_Session_User_PAI.avi. PAI 1=real, 2-3=print, 4-5=video replay.
    Official subject-disjoint split by user ID: 1-20 train, 21-35 dev, 36-55 test (verify vs protocol files)."""
    rx = re.compile(r"^(\d+)_(\d+)_(\d+)_(\d+)$")
    rows, skipped = [], 0
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in VIDEO_EXT:
            continue
        m = rx.match(p.stem)
        if not m:
            skipped += 1
            continue
        user, pai = int(m.group(3)), int(m.group(4))
        split = "train" if user <= 20 else ("validation" if user <= 35 else "test")
        attack = {1: "none", 2: "print", 3: "print", 4: "replay", 5: "replay"}.get(pai, "unknown")
        rows.append({"video_id": f"oulu_npu:{p.stem}", "source_dataset": "oulu_npu",
                     "subject_id": f"oulu_npu:user_{user:02d}", "label": "REAL" if pai == 1 else "SPOOF",
                     "attack_type": attack, "split": split, "storage_uri": str(p), "file_available": True,
                     "data_origin": "IMPORTED"})
    return rows, skipped


def replay_rows(root: Path) -> tuple[list[dict], int]:
    """Replay-Attack: official train/ devel/ test/ folders (already subject-disjoint); 'client###' in the name;
    attack videos start with 'attack' and live under an 'attack' directory."""
    rows, skipped = [], 0
    split_dir = {"train": "train", "devel": "validation", "test": "test"}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in VIDEO_EXT:
            continue
        parts = [x.lower() for x in p.parts]
        split = next((split_dir[d] for d in parts if d in split_dir), None)
        m = re.search(r"client(\d+)", p.name.lower()) or re.search(r"client(\d+)", "/".join(parts))
        if split is None or m is None:
            skipped += 1
            continue
        spoof = p.name.lower().startswith("attack") or "attack" in parts
        attack = "none"
        if spoof:
            attack = "print" if "print" in p.name.lower() else ("replay" if any(
                k in p.name.lower() for k in ("mobile", "highdef", "tablet", "video")) else "unknown")
        rows.append({"video_id": f"replay_attack:{p.stem}", "source_dataset": "replay_attack",
                     "subject_id": f"replay_attack:client{int(m.group(1)):03d}",
                     "label": "SPOOF" if spoof else "REAL", "attack_type": attack, "split": split,
                     "storage_uri": str(p), "file_available": True, "data_origin": "IMPORTED"})
    return rows, skipped


def midv_rows(root: Path, seed: int) -> tuple[list[dict], int]:
    """MIDV: index images; group = document-type folder + clip folder. Splits are by group (hash), never by image.
    tampering_label stays UNAVAILABLE: the base MIDV annotations are quads/fields, not forgery labels."""
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXT:
            continue
        rel = p.relative_to(root).parts
        doc_type = rel[0] if len(rel) > 1 else "UNKNOWN"
        clip = "/".join(rel[:-1]) or "root"
        ann = next((str(a) for a in p.parent.glob("*.json")), "")
        rows.append({"document_id": f"midv:{'/'.join(rel)}", "source_dataset": "midv",
                     "source_subject_id": f"midv:{clip}", "document_type": doc_type, "storage_uri": str(p),
                     "tampering_label": "UNAVAILABLE", "annotation_uri": ann,
                     "split": hash_split("midv:" + clip, seed), "data_origin": "IMPORTED"})
    return rows, 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oulu"), ap.add_argument("--replay"), ap.add_argument("--midv")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args(argv)
    ensure_dirs()

    live, docs = [], []
    for name, path, fn in (("oulu_npu", args.oulu, oulu_rows), ("replay_attack", args.replay, replay_rows)):
        if path:
            rows, skipped = fn(Path(path))
            live += rows
            print(f"{name}: indexed {len(rows):,} videos, skipped {skipped:,} unrecognised files")
    if args.midv:
        rows, _ = midv_rows(Path(args.midv), args.seed)
        docs += rows
        print(f"midv: indexed {len(rows):,} document images (tampering labels: UNAVAILABLE)")

    pd.DataFrame(live, columns=LIVENESS_COLS).to_csv(PROCESSED_DIR / "liveness_manifest.csv", index=False)
    pd.DataFrame(docs, columns=DOCUMENT_COLS).to_csv(PROCESSED_DIR / "document_manifest.csv", index=False)
    if not live and not docs:
        print("No KYC sources supplied: wrote header-only manifests. Nothing was fabricated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
