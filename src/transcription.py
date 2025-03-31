import os
import time
from .config import *
import pandas as pd
from openai import OpenAI
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)

def transcribe(audio_file_path: str, model: str = "gpt-4o-mini-transcribe"):
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        deepgram_client = DeepgramClient(DEEPGRAM_API_KEY)
        base_name = os.path.basename(audio_file_path).split('.')[0]
        print(f'ファイル読み込み中: {base_name}')
        audio_file = open(audio_file_path, "rb")

        start_time = time.perf_counter()
        print(f'文字起こし中...')
        prompt = "この音声は、WEB会議（Teasm）の音声（日本語）です。「えー」「えっと」「えーっと」「あー」「そのー」「んーと」などのような、フィラーや言葉のひげと呼ばれる音声も省略せずに、発言を全てを文字起こししてください。"

        if model == "deepgram":
            payload: FileSource = {"buffer": audio_file}
            # Deepgramのオプションを設定
            options = PrerecordedOptions(
                model="whisper-large",
                language="ja",
                diarize=True,
                utterances=True,
            )
            transcription = deepgram_client.listen.prerecorded.v("1").transcribe_file(payload, options, timeout=300)

        elif model == "gpt-4o-mini-transcribe":
            transcription = openai_client.audio.transcriptions.create(
                model=model, 
                file=audio_file, 
                prompt=prompt, # promptを指定すると、文字起こしがむしろ省略されてしまうことあり、特にwhisper-1
                # response_format="text"
            )

        elif model == "whisper-1":
            transcription = openai_client.audio.transcriptions.create(
                model=model, 
                file=audio_file, 
                # prompt=prompt, # promptを指定すると、文字起こしがむしろ省略されてしまうことあり、特にwhisper-1
                # response_format="text"
            )

        else:
            raise ValueError(f"Invalid model: {model}")

        end_time = time.perf_counter()
                
        run_time = end_time - start_time
        print(f"実行時間: {run_time:.2f}秒")

        print(transcription)

    except Exception as e:
        print(f"Exception: {e}")