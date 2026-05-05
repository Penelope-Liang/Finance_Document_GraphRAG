from __future__ import annotations

import argparse
import json
from pathlib import Path


def _shorten(text: str, limit: int) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect failed direct-extraction retrieval results."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to batch results json produced by run_direct_extract_qa_batch.py",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all rows instead of only failed rows.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=5,
        help="Max retrieved items to show per question.",
    )
    parser.add_argument(
        "--content-limit",
        type=int,
        default=320,
        help="Max characters of each retrieved content item to print.",
    )
    args = parser.parse_args()

    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    for row in rows:
        failed = row.get("answer") is None or row.get("error") is not None
        if not args.show_all and not failed:
            continue

        print("=" * 100)
        print("question_id =", row.get("question_id"))
        print("doc_id      =", row.get("doc_id"))
        print("gold_answer =", row.get("gold_answer"))
        print("answer      =", row.get("answer"))
        print("evidence    =", row.get("evidence"))
        print("error       =", row.get("error"))

        raw_answer = row.get("raw_answer")
        if raw_answer:
            print("raw_answer  =", _shorten(str(raw_answer), 500))

        rr = row.get("retriever_result") or {}
        metadata = rr.get("metadata")
        if metadata:
            print("retriever metadata =", json.dumps(metadata, ensure_ascii=False)[:1000])

        items = rr.get("items") or []
        print(f"retrieved_items = {len(items)}")
        for idx, item in enumerate(items[: args.max_items], start=1):
            content = str(item.get("content") or "")
            item_meta = item.get("metadata") or {}
            print(f"[{idx}] metadata = {json.dumps(item_meta, ensure_ascii=False)}")
            print(f"[{idx}] content  = {_shorten(content, args.content_limit)}")


if __name__ == "__main__":
    main()
