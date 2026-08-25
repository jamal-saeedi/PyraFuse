"""CLI for building skin/fabric/background label masks.

    python -m pyrafuse.data --split val
    python -m pyrafuse.data --split train

See scripts/prepare_data.py for the full pipeline (downloads + label build).
"""

import argparse

from .masks import build_split


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N masks (smoke test)")
    args = p.parse_args()
    build_split(args.split, args.limit)


if __name__ == "__main__":
    main()
