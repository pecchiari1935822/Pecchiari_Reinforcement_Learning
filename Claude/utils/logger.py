import logging
import sys
from pathlib import Path
from config.settings import LOG_LEVEL

# Crea directory per i log
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Configura il logger
logger = logging.getLogger("BladeOptimization")
logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

# Handler per console (stdout)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

# Handler per file
file_handler = logging.FileHandler(LOG_DIR / "optimization.log")
file_handler.setLevel(logging.DEBUG)

# Formato log
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

if __name__ == "__main__":
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")