import logging


def init_logger(log_file: str, loglevel: str):
    logger = logging.getLogger()
    loglevel = getattr(logging, loglevel.upper())
    logger.setLevel(loglevel)
    # create file handler which logs event debug messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(loglevel)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(loglevel)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
