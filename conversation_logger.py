import logging
import queue
from logging.handlers import QueueHandler, QueueListener
from datetime import datetime

log_queue = queue.Queue()
file_handler = logging.FileHandler(f"logs/conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))

listener = QueueListener(log_queue, file_handler)
listener.start()

conv_logger = logging.getLogger("conversation")
conv_logger.setLevel(logging.INFO)
conv_logger.addHandler(QueueHandler(log_queue))