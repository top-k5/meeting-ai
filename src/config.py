from dotenv import load_dotenv
load_dotenv()

import os

DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
AUDIO_FILE_PATH = os.getenv('AUDIO_FILE_PATH')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
AMIVOICE_API_KEY = os.getenv('AMIVOICE_API_KEY')

AUDIO_FILE_PATH = 'data/raw/就職面接でたぶんスーツ着てない奴_10minutes.mp3'
PROCESSED_DIR = "data/processed"
TRANSCRIPTION_DIR = "data/transcription"
RESULT_DIR = "data/result"
