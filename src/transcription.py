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
import requests
import pickle

def transcribe(audio_file_path: str, api: str = "openai", model: str = "gpt-4o-mini-transcribe"):
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        deepgram_client = DeepgramClient(DEEPGRAM_API_KEY)
        base_name = os.path.basename(audio_file_path)
        print(f'ファイル読み込み中: {base_name}')
        base_name = base_name.split('.')[0]
        audio_file = open(audio_file_path, "rb")

        start_time = time.perf_counter()
        print(f'文字起こし中...')
        prompt = "この音声は、WEB会議の音声（日本語）です。「えー」「えっと」「えーっと」「あー」「そのー」「んーと」やその他のフィラーも省略せずに、発言を全てを文字起こししてください。"

        if api == "deepgram":
            payload: FileSource = {"buffer": audio_file}
            # Deepgramのオプションを設定
            options = PrerecordedOptions(
                model=model,
                language="ja",
                diarize=True,
                utterances=True,
                filler_words=True,
            )
            transcription = deepgram_client.listen.prerecorded.v("1").transcribe_file(payload, options, timeout=300)

            for u in transcription.results.utterances:
                print(f"[Speaker: {u.speaker}] {u.transcript}")

        elif api == "openai":
            transcription = openai_client.audio.transcriptions.create(
                model=model, 
                file=audio_file, 
                prompt=prompt, # promptを指定すると、文字起こしがむしろ省略されてしまうことあり、特にwhisper-1
                # response_format="text"
            )
            print(transcription)

        elif api == "openai" and model == "whisper-1":
            transcription = openai_client.audio.transcriptions.create(
                model=model, 
                file=audio_file, 
                # prompt=prompt, # promptを指定すると、文字起こしがむしろ省略されてしまうことあり、特にwhisper-1
                # response_format="text"
            )
            print(transcription)

        elif api == "amivoice" and model == '-a-general':
            url = "https://acp-api.amivoice.com/v1/recognize"
            params = {
                "d": f"grammarFileNames={model} keepFillerToken=1 speakerDiarization=True"
            }
            files = {
                "u": AMIVOICE_API_KEY,
                "a": audio_file
            }
            response = requests.post(url, params=params, files=files)
            # レスポンスを処理
            if response.status_code == 200:
                transcription = response.json()
                print("認識結果:", transcription["text"])
                
            else:
                print("エラー:", response.status_code, response.text)

        else:
            raise ValueError(f"Invalid model: {model}")

        end_time = time.perf_counter()
                
        run_time = end_time - start_time
        print(f"実行時間: {run_time:.2f}秒")

        # 結果の保存
        pkl_name = f"{base_name}_{api}_{model}.pkl"
        print(f"結果の保存中: {pkl_name}")
        pickle.dump(transcription, open(os.path.join(TRANSCRIPTION_DIR, pkl_name), "wb"))
    except Exception as e:
        print(f"Exception: {e}")