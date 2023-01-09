import os.path as osp

# Where experiment outputs are saved by default:
USER_CONFIG_DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))),'data')

# Whether to automatically insert a date and time stamp into the names of
# save directories:
USER_CONFIG_DATESTAMP = False

# Whether GridSearch provides automatically-generated default shorthands:
USER_CONFIG_DEFAULT_SHORTHAND = True

# Tells the GridSearch how many seconds to pause for before launching
# experiments.
USER_CONFIG_WAIT_BEFORE_LAUNCH = 5

USER_CONFIG_DIV_LINE_WIDTH = 80