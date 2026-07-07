#!/usr/bin/env python3
import argparse
from pathlib import Path


TARGET_RELATIVE_FILES = (
    "wan/modules/s2v/motioner.py",
    "wan/modules/s2v/model_s2v.py",
)


def patch_text(text: str) -> tuple[str, int]:
    replacements = (
        ("torch.cuda.amp.autocast()", 'torch.amp.autocast("cuda")'),
        ("torch.cuda.amp.autocast(", 'torch.amp.autocast("cuda", '),
    )
    patched = text
    total = 0
    for old, new in replacements:
        count = patched.count(old)
        if count:
            patched = patched.replace(old, new)
            total += count
    return patched, total


def patch_file(path: Path, dry_run: bool) -> int:
    original = path.read_text(encoding="utf-8")
    patched, count = patch_text(original)
    if count and not dry_run:
        path.write_text(patched, encoding="utf-8")
    print(f"{path}: autocast replacements={count}")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch Wan2.2 deprecated torch.cuda.amp.autocast calls.")
    parser.add_argument("repo_dir", nargs="?", default="/opt/Wan2.2")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    total = 0
    for relative in TARGET_RELATIVE_FILES:
        path = repo_dir / relative
        if not path.exists():
            raise FileNotFoundError(f"Expected Wan2.2 file not found: {path}")
        total += patch_file(path, args.dry_run)

    if total == 0:
        print("No deprecated autocast calls found; patch noop.")
        return 0
    print(f"total autocast replacements={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
