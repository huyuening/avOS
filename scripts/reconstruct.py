import sys
from pathlib import Path

AVOS_DIR = Path(__file__).resolve().parents[1] / "avos"
if str(AVOS_DIR) not in sys.path:
    sys.path.insert(0, str(AVOS_DIR))

from scenario_reconstructor.reconstruction import main


if __name__ == "__main__":
    raise SystemExit(main())
