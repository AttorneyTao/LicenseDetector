"""One-time migration: remove log records older than seven days.

Run only while the service is stopped. The default mode is a read-only report;
pass --apply to rewrite the explicitly selected log directory.
"""

import argparse
import os
import re
import stat
import tempfile
from datetime import datetime, timedelta
from pathlib import Path


LOG_NAME = re.compile(r"^[A-Za-z0-9_]+\.log(?:\.\d{4}-\d{2}-\d{2}\.\d{3,})?$")
RECORD_START = re.compile(rb"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - ")


def prune_file(path: Path, cutoff: datetime, apply: bool = False) -> tuple[int, int]:
    """Stream a log, retaining recent records and their continuation lines."""
    cutoff_bytes = cutoff.strftime("%Y-%m-%d %H:%M:%S").encode("ascii")
    keep = False
    retained = removed = 0
    temp_path = None
    output = None
    try:
        if apply:
            output = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False)
            temp_path = Path(output.name)
            os.fchmod(output.fileno(), stat.S_IMODE(path.stat().st_mode))
        with path.open("rb") as source:
            for line in source:
                if RECORD_START.match(line):
                    keep = line[:19] >= cutoff_bytes
                if keep:
                    retained += len(line)
                    if output:
                        output.write(line)
                else:
                    removed += len(line)
        if output:
            output.flush()
            os.fsync(output.fileno())
            output.close()
            output = None
            if removed:
                if retained == 0 and ".log." in path.name:
                    path.unlink()
                    temp_path.unlink()
                else:
                    os.replace(temp_path, path)
            else:
                temp_path.unlink()
            temp_path = None
    finally:
        if output:
            output.close()
        if temp_path and temp_path.exists():
            temp_path.unlink()
    return retained, removed


def prune_directory(log_dir: Path, cutoff: datetime, apply: bool = False) -> tuple[int, int]:
    if not log_dir.is_dir():
        raise ValueError(f"Not a log directory: {log_dir}")
    retained = removed = 0
    for path in sorted(log_dir.iterdir()):
        if not LOG_NAME.fullmatch(path.name) or not path.is_file() or path.is_symlink():
            continue
        kept_here, removed_here = prune_file(path, cutoff, apply)
        retained += kept_here
        removed += removed_here
        print(f"{path.name}: retain={kept_here} remove={removed_here}")
    return retained, removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--apply", action="store_true", help="rewrite logs; default is dry-run")
    args = parser.parse_args()
    if args.days <= 0:
        parser.error("--days must be positive")
    cutoff = datetime.now() - timedelta(days=args.days)
    retained, removed = prune_directory(args.log_dir, cutoff, args.apply)
    print(f"cutoff={cutoff.isoformat(timespec='seconds')} retained={retained} removed={removed} apply={args.apply}")


if __name__ == "__main__":
    main()
