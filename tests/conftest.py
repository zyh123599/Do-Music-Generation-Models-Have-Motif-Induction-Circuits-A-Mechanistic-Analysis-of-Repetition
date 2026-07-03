import sys
from pathlib import Path

# Allow running the tests without installing the package.
ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT),):
    if p not in sys.path:
        sys.path.insert(0, p)
