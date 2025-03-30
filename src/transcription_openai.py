import src
from openai import OpenAI
import os
import time
import pandas as pd

def transcribe_openai(audio_file_path: str, model: str = "gpt-4o-mini-transcribe"):
    base_name = os.path.basename(audio_file_path).split('.')[0]

    try:
        print(f'ファイル読み込み中: {base_name}')
        audio_file = open(src.AUDIO_FILE_PATH, "rb")

        start_time = time.perf_counter()
        print(f'文字起こし中...')
        # 結果を取得
        client = OpenAI()
        prompt = "この音声は、WEB会議（Teasm）の音声（日本語）です。「えー」「えっと」「えーっと」「あー」「そのー」「んーと」などのような、フィラーや言葉のひげと呼ばれる音声も省略せずに、発言を全てを文字起こししてください。"

        # promptを指定すると、文字起こしがむしろ省略されてしまうことあり、特にwhisper-1
        transcription = client.audio.transcriptions.create(
            model=model, 
            file=audio_file, 
            prompt=prompt,
            # response_format="text"
        )

        end_time = time.perf_counter()
                
        run_time = end_time - start_time
        print(f"実行時間: {run_time:.2f}秒")

    except Exception as e:
        print(f"Exception: {e}")
    
    print(transcription)