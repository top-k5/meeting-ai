import ffmpeg

def trim_mp4(input_file, start_time, end_time, output_file):
    """
    指定したMP4ファイルをトリミングする関数

    :param input_file: 入力MP4ファイルのパス
    :param start_time: トリミングの開始時刻（秒）
    :param end_time: トリミングの終了時刻（秒）
    :param output_file: 出力MP4ファイルのパス
    :return: 出力ファイルのパス
    """
    # ffmpegを使用してトリミングを実行
    (
        ffmpeg
        .input(input_file, ss=start_time, to=end_time)
        .output(output_file)
        .run(overwrite_output=True)
    )
    
    return output_file
