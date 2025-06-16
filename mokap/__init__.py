import logging
import sys

logger = logging.getLogger('mokap')
logger.setLevel(logging.INFO)

# A handler for printing to the console (stderr)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

# file_handler = logging.FileHandler("mokap.log", mode='w')
# file_handler.setLevel(logging.INFO) # only log INFO and above to the file

formatter = logging.Formatter(
    # '%(asctime)s - [%(levelname)s] %(message)s'
    '[%(levelname)s] %(message)s'
)
console_handler.setFormatter(formatter)
# file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
# logger.addHandler(file_handler)