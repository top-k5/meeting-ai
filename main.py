import src

def main():
    src.transcribe(src.AUDIO_FILE_PATH)
    src.judge_conclusion_first()

if __name__ == "__main__":
    main()
