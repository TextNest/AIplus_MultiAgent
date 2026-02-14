
import logging
import os

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "agent.log")


logger = logging.getLogger(name='AgentLog')
logger.setLevel(logging.INFO)


formatter = logging.Formatter('|%(asctime)s|%(levelname)s| - %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')

file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

if logger.hasHandlers():
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
