"""AIBrain Framework

    This file is a starting point for running the AIBrain application.
"""


# region Imported Dependencies------------------------------------------------------------------------------------------
import argparse
from brain.core import Sys
# endregion Imported Dependencies


if __name__ == '__main__':
    # Input Parameters
    parser = argparse.ArgumentParser(description='AIBrain')
    parser.add_argument('--cfg', '-c', type=str, help='The configuration file path', default='../cfg/cfg.properties')
    args = parser.parse_args()

    # Application Initialization
    app = Sys(a_cfg=args.cfg)

    # Application Run
    app.run()


