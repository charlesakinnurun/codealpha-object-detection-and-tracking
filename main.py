"""Root-level convenience entry point.

Forwards to ``src.main`` so the exact CLI form ``python main.py --source webcam``
works from the repository root (``python src/main.py`` also works).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())