import logging
import sys

# Logging:  ANSI color codes
colors_dict = {
    "I": 32,  # green
    "W": 33,  # orange/yellow
    "E": 31,  # red
}

level_dict = {
    logging.INFO: "I",
    logging.WARNING: "W",
    logging.ERROR: "E",
    logging.CRITICAL: "C",
}


class LoggingColor(logging.Formatter):
    """Formatter for Python Logging (inject colors """
    def format(self, record):
        level_letter = level_dict.get(record.levelno, "?")
        code = colors_dict.get(level_letter)
        tag = f"\033[{code}m[{level_letter}]\033[0m" if code else f"[{level_letter}]"
        return f"[3DFos] {tag} {record.getMessage()}"


def setup_logging() -> None:
    """Configure logging for the 3DFos CLI."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(LoggingColor())
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)
