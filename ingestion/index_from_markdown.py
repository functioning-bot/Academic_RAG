"""
ingestion/index_from_markdown.py
--------------------------------
Resume ingestion from already-parsed Markdown files.

`batch_ingest.py` parsed all PDFs through LlamaParse successfully (the .md files
were saved), but crashed on a Windows console-encoding error while printing a
'✓' character — before chunking/indexing ran. This script picks up from the
saved .md files: chunk → index into Qdrant. No re-parsing, no LlamaParse cost.

stdout/stderr are reconfigured to UTF-8 so the pipeline's '✓'/'✗' prints do not
crash under the cp1252 console codec.

Usage:
    python ingestion/index_from_markdown.py
    python ingestion/index_from_markdown.py --src ingestion/new_pdfs
"""
import argparse
import sys
from pathlib import Path

# Fix Windows cp1252 console crash on unicode prints (the original bug).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_INGESTION_DIR = Path(__file__).resolve().parent
if str(_INGESTION_DIR) not in sys.path:
    sys.path.insert(0, str(_INGESTION_DIR))

try:
    from dotenv import load_dotenv
    _REPO = _INGESTION_DIR.parent
    for _ep in (_REPO / "brain" / ".env", _REPO / ".env"):
        if _ep.exists():
            load_dotenv(_ep)
            break
except ImportError:
    pass

from chunker import chunk_markdown
import indexer


def main() -> int:
    ap = argparse.ArgumentParser("Index already-parsed Markdown into Qdrant")
    ap.add_argument("--src", default=str(_INGESTION_DIR / "new_pdfs"),
                    help="Folder containing parsed .md files")
    args = ap.parse_args()

    src = Path(args.src)
    md_files = sorted(src.glob("*.md"))
    if not md_files:
        print(f"[index-md] no .md files found in {src}")
        return 1

    print(f"[index-md] {len(md_files)} markdown file(s) to chunk + index")
    ok, failed = 0, []

    for i, md in enumerate(md_files, 1):
        # The chunker tags chunks with source_file; match the watcher's
        # convention of using the original PDF filename (stem + .pdf).
        source_file = md.stem + ".pdf"
        print(f"\n{'=' * 72}")
        print(f"[index-md] ({i}/{len(md_files)}) {source_file}")
        print(f"{'=' * 72}")
        try:
            text = md.read_text(encoding="utf-8")
            chunks = chunk_markdown(text, source_file=source_file)
            if not chunks:
                print(f"[index-md] WARNING: 0 chunks produced for {source_file}")
            indexer.index_chunks(chunks)
            ok += 1
        except Exception as exc:
            failed.append((source_file, str(exc)))
            print(f"[index-md] FAILED: {source_file} -> {exc}")

    print(f"\n{'=' * 72}")
    print(f"[index-md] done. succeeded={ok}  failed={len(failed)}")
    for name, err in failed:
        print(f"  [fail] {name}: {err}")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
