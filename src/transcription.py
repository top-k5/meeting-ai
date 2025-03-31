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

def transcribe_deepgram(audio_file_path: str):
    deepgram_client = DeepgramClient(DEEPGRAM_API_KEY)
    base_name = os.path.basename(audio_file_path).split('.')[0]

    try:
        print(f'ファイル読み込み中: {base_name}')
        with open(audio_file_path, "rb") as file:
            buffer_data = file.read()
        payload: FileSource = {
            "buffer": buffer_data,
        }

        # Deepgramのオプションを設定
        options = PrerecordedOptions(
            model="whisper-large",
            language="ja",
            diarize=True,
            utterances=True,
        )

        start_time = time.perf_counter()
        print(f'文字起こし中...')
        # 結果を取得
        response = deepgram_client.listen.prerecorded.v("1").transcribe_file(payload, options, timeout=300)

        end_time = time.perf_counter()

        for u in response.results.utterances:
            print(f"[Speaker: {u.speaker}] {u.transcript}")
                
        run_time = end_time - start_time
        print(f"実行時間: {run_time:.2f}秒")

        # 文字起こし結果を保存
        df_structured = pd.DataFrame(response.results.utterances)
        df_structured.drop(columns=["channel",  "words", "sentiment", "sentiment_score", "id"], inplace=True)

        print(f'文字起こし結果を保存中...')
        df_structured.to_csv(f"{PROCESSED_DIR}/structured_{base_name}.csv", index=False)

    except Exception as e:
        print(f"Exception: {e}")


# OpenAIを使用した文字起こし
def transcribe_openai(audio_file_path: str, model: str = "gpt-4o-mini-transcribe"):
    base_name = os.path.basename(audio_file_path).split('.')[0]

    try:
        print(f'ファイル読み込み中: {base_name}')
        audio_file = open(audio_file_path, "rb")

        start_time = time.perf_counter()
        print(f'文字起こし中...')
        # 結果を取得
        client = OpenAI(api_key=OPENAI_API_KEY)
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

        print(transcription)

    except Exception as e:
        print(f"Exception: {e}")
    
