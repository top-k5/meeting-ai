# 参考
# ・langchainの基本的な流れ
# https://note.com/unco3/n/nb8dc295db03b

# ・Chainの基本
# https://qiita.com/taka_yayoi/items/f09678fe6dcd57c8d2b3

from .config import *
import pandas as pd

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
import openai
# from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

def judge_conclusion_first():
    # 文字起こしデータの読み込み
    print(f"文字起こしデータの読み込み中...")
    df_transcription = pd.read_csv(f"{PROCESSED_DIR}/sample_data.csv")[:10]

    # プロンプトテンプレートの作成
    template = ChatPromptTemplate([
        ("system", "あなたは、WEB会議の質を高めるためのコンサルタントです。あなたは、userから与えられたWEB会議内の発言が結論ファーストであるかどうかを判定し、結論ファーストならば1、そうでなければ0として出力してください。\n{format_instructions}"),
        ("human", "{user_input}"),
    ])
    # モデルの設定
    model = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key= OPENAI_API_KEY)
    # 出力形式の設定
    first_conclusion_schema = ResponseSchema(name="conc_1st_flg", description="2値の結論ファーストフラグ。結論ファーストならば1、そうでなければ0とする。")
    output_parser = StructuredOutputParser.from_response_schemas([first_conclusion_schema])
    format_instructions = output_parser.get_format_instructions()

    # チェーンの作成
    chain = template | model | output_parser

    # 各発言に対してチェーンの実行
    print(f"各発言に対してチェーンの実行中...")
    results = []
    for _, row in df_transcription.iterrows():
        text = row["transcript"]
        result = chain.invoke(
            {
                "format_instructions": format_instructions,
                "user_input": text
            }
        )
        results.append(result['conc_1st_flg'])
        print(f"発言: {text}\n結論ファーストフラグ: {result['conc_1st_flg']}\n")

    df_transcription["is_conclusion_1st"] = results 

    print(f"結果を保存中...")
    df_transcription.to_csv(f"{RESULT_DIR}/result_sample_data.csv", index=False)