"""Allow running as: python -m experiments.benchmark --config <yaml>"""

import sys

from experiments.benchmark.cli import main

if __name__ == "__main__":
    sys.exit(main())
