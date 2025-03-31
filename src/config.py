from dotenv import load_dotenv
load_dotenv()

import os

DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
AUDIO_FILE_PATH = os.getenv('AUDIO_FILE_PATH')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

AUDIO_FILE_PATH = 'data/raw/就職面接でたぶんスーツ着てない奴.mp4'
PROCESSED_DIR = "data/processed"
RESULT_DIR = "data/result"
