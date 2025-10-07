from __future__ import annotations
import logging
import os

# Simple ANSI color codes; auto-disable on non-TTY/CI
USE_COLOR = os.getenv("LOG_COLOR", "1") == "1" and os.getenv("NO_COLOR") is None

COLORS = {
    "RESET": "\033[0m",
    "GRAY": "\033[90m",
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "MAGENTA": "\033[95m",
    "CYAN": "\033[96m",
}

LEVEL_COLOR = {
    logging.DEBUG: "GRAY",
    logging.INFO: "CYAN",
    logging.WARNING: "YELLOW",
    logging.ERROR: "RED",
    logging.CRITICAL: "MAGENTA",
}


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if not USE_COLOR:
            return msg
        color = COLORS.get(LEVEL_COLOR.get(record.levelno, "RESET"), "")
        reset = COLORS["RESET"]
        return f"{color}{msg}{reset}"


def install_color_handler(logger: logging.Logger, level: int = logging.INFO) -> None:
    if any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        return
    handler = logging.StreamHandler()
    handler.setLevel(level)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handler.setFormatter(ColorFormatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(level)
