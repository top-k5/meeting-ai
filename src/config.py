from dotenv import load_dotenv
load_dotenv()

import os

DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
AUDIO_FILE_PATH = os.getenv('AUDIO_FILE_PATH')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

PROCESSED_DIR = "data/processed"
RESULT_DIR = "data/result"
