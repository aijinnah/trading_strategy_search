import logging


def setup_logger(name, log_file, level=logging.INFO):
    """Function to set up a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a file handler for the logger
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Create a console handler for the logger
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Set the formatter
    formatter = logging.Formatter("%(message)s")

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
