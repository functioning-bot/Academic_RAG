"""
ingestion/batch_ingest.py
-------------------------
One-shot batch ingestion of a folder of PDFs.

The normal entry point (`watcher.py`) is a continuous folder-watcher. For a
known fixed set of PDFs this script calls the same `parse_pdf_agentic` pipeline
directly, once per file: LlamaParse → (figures) → chunk → index into Qdrant.

Text-only by default: ENABLE_IMAGE_DESCRIPTION is forced false so no Ollama /
vision model is required. Tables still come through (LlamaParse emits them as
markdown); only image/diagram description chunks are skipped.

Usage (from repo root or anywhere):
    python ingestion/batch_ingest.py
    python ingestion/batch_ingest.py --src ingestion/new_pdfs
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

# Text-only: must be set BEFORE importing watcher (it reads this at import time).
os.environ["ENABLE_IMAGE_DESCRIPTION"] = "false"

_INGESTION_DIR = Path(__file__).resolve().parent
if str(_INGESTION_DIR) not in sys.path:
    sys.path.insert(0, str(_INGESTION_DIR))

# Load .env (LLAMA_CLOUD_API_KEY, QDRANT_URL). load_dotenv does not override
# already-set vars, so the ENABLE_IMAGE_DESCRIPTION above stays false.
try:
    from dotenv import load_dotenv
    _REPO = _INGESTION_DIR.parent
    for _ep in (_REPO / "brain" / ".env", _REPO / ".env"):
        if _ep.exists():
            load_dotenv(_ep)
            break
except ImportError:
    pass

os.environ["ENABLE_IMAGE_DESCRIPTION"] = "false"  # belt-and-suspenders

import watcher  # noqa: E402  (import after env is configured)


def main() -> int:
    ap = argparse.ArgumentParser("Batch-ingest a folder of PDFs")
    ap.add_argument("--src", default=str(_INGESTION_DIR / "new_pdfs"),
                    help="Folder containing PDFs to ingest")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"[batch] source folder not found: {src}")
        return 1

    pdfs = sorted(src.glob("*.pdf"))
    if not pdfs:
        print(f"[batch] no PDFs found in {src}")
        return 1

    print(f"[batch] {len(pdfs)} PDF(s) to ingest from {src}")
    print(f"[batch] image description enabled: {watcher.ENABLE_IMAGE_DESCRIPTION}")

    ok, failed = 0, []
    for i, pdf in enumerate(pdfs, 1):
        print(f"\n{'=' * 72}")
        print(f"[batch] ({i}/{len(pdfs)}) {pdf.name}")
        print(f"{'=' * 72}")
        try:
            asyncio.run(watcher.parse_pdf_agentic(str(pdf)))
            ok += 1
            print(f"[batch] OK: {pdf.name}")
        except Exception as exc:
            failed.append((pdf.name, str(exc)))
            print(f"[batch] FAILED: {pdf.name} -> {exc}")

    print(f"\n{'=' * 72}")
    print(f"[batch] done. succeeded={ok}  failed={len(failed)}")
    for name, err in failed:
        print(f"  [fail] {name}: {err}")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
