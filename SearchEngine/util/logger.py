import logging


class LogFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    green = "\x1b[32;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(message)s"

    FORMATS = {
        logging.DEBUG: green + "[#] " + reset + format,
        logging.INFO: grey + "[*] " + reset + format,
        logging.WARNING: yellow + "[!] " + reset + format,
        logging.ERROR: red + "[!!] " + reset + format,
        logging.CRITICAL: bold_red + "[!!!] " + reset + format
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = logging.getLogger("IR")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(LogFormatter())

logger.addHandler(ch)
