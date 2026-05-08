#!/usr/bin/env python3
"""
Manual val-page tally helper. Just run with:
    python3 scripts/manual_tally.py 137 28 34
                                    ^   ^  ^
                                   page correct total

Saves to outputs/manual_tally.json so you can build it up across pages.
At the end, prints aggregate accuracy.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TALLY = ROOT / "outputs" / "manual_tally.json"


def load() -> dict:
    if TALLY.exists():
        return json.loads(TALLY.read_text())
    return {}


def save(data: dict) -> None:
    TALLY.parent.mkdir(parents=True, exist_ok=True)
    TALLY.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def show(data: dict) -> None:
    print(f"\n{'Page':>6}  {'correct':>8}  {'total':>6}  {'acc':>6}")
    print("-" * 36)
    sum_c = 0
    sum_t = 0
    for page, r in sorted(data.items(), key=lambda kv: int(kv[0])):
        acc = r["correct"] / r["total"] if r["total"] else 0
        print(f"{page:>6}  {r['correct']:>8}  {r['total']:>6}  {acc:>6.3f}")
        sum_c += r["correct"]
        sum_t += r["total"]
    if sum_t:
        agg = sum_c / sum_t
        print("-" * 36)
        print(f"{'TOTAL':>6}  {sum_c:>8}  {sum_t:>6}  {agg:>6.3f}")
        print(f"\nWeighted character accuracy: {agg:.1%} ({sum_c}/{sum_t})")


def main() -> None:
    if len(sys.argv) == 1:
        show(load())
        return
    if len(sys.argv) == 2 and sys.argv[1] in ("show", "ls"):
        show(load())
        return
    if len(sys.argv) == 3 and sys.argv[1] == "rm":
        data = load()
        if sys.argv[2] in data:
            del data[sys.argv[2]]
            save(data)
            print(f"removed {sys.argv[2]}")
        show(data)
        return
    if len(sys.argv) != 4:
        print("Usage:")
        print("  python3 scripts/manual_tally.py <page> <correct> <total>")
        print("  python3 scripts/manual_tally.py show")
        print("  python3 scripts/manual_tally.py rm <page>")
        sys.exit(1)

    page, correct, total = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    data = load()
    data[page] = {"correct": correct, "total": total}
    save(data)
    print(f"saved page {page}: {correct}/{total} = {correct/total:.1%}")
    show(data)


if __name__ == "__main__":
    main()
