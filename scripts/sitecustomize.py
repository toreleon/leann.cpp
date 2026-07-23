"""Keep third-party HTTP client request logs out of timed benchmark output."""

import logging

for logger_name in ("httpx", "httpcore", "openai"):
    logging.getLogger(logger_name).setLevel(logging.WARNING)
