import os
import time
from src.config import *
import pandas as pd

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)

def transcribe(audio_file_path: str):
    deepgram_client = DeepgramClient(DEEPGRAM_API_KEY)
    base_name = os.path.basename(audio_file_path)

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

    except Exception as e:
        print(f"Exception: {e}")

    df_structured = pd.DataFrame(response.results.utterances)
    df_structured.drop(columns=["channel",  "words", "sentiment", "sentiment_score", "id"], inplace=True)
    
    print(f'文字起こし結果を保存中...')
    df_structured.to_csv(f"{OUTPUT_DIR}/structured_{base_name}.csv", index=False)

