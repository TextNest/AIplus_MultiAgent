import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DUMMY_MODE = os.getenv("DUMMY_MODE", "False").lower() == "true"
    REPORT_FORMAT = os.getenv("REPORT_FORMAT", "markdown") # markdown, html, pdf, pptx

config = Config()
