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


def judge_understood():
    # 文字起こしデータの読み込み
    print(f"文字起こしデータの読み込み中...")
    df_transcription = pd.read_csv(f"{PROCESSED_DIR}/sample_data.csv")[:10]

    # プロンプトテンプレートの作成
    template = ChatPromptTemplate([
        ("system", "あなたは、WEB会議の質を高めるためのコンサルタントです。あなたは、userから与えられたWEB会議内の発言履歴を元に、判定対象の発言が、その直後の別の発言者に理解されているかどうかを判定してください。判断材料として、判定対象の発言以前の会話履歴と、その直後の発言がuserから与えられます。\n{format_instructions}"),
        ("human", "判定対象の発言以前の会話履歴：\n{history}\n判定対象の発言：\n{target}\n判定対象の発言の直後の発言：\n{next}"),
    ])
    # モデルの設定
    model = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key= OPENAI_API_KEY)
    # 出力形式の設定
    understood_schema = ResponseSchema(name="understood_flg", description="2値の理解フラグ。判定対象の発言が、直後の発言者に理解されていれば1、そうでなければ0とする。")
    output_parser = StructuredOutputParser.from_response_schemas([understood_schema])
    format_instructions = output_parser.get_format_instructions()

    # チェーンの作成
    chain = template | model | output_parser

    # 各発言に対してチェーンの実行
    print(f"各発言に対してチェーンの実行中...")
    results = []

    for i in range(len(df_transcription)-1):
        target = f'{df_transcription["speaker"][i]}「 {df_transcription["transcript"][i]}」'
        next = f'{df_transcription["speaker"][i+1]}「 {df_transcription["transcript"][i+1]}」'
        history = ''
        for history_i in range(i):
            history += f'{df_transcription["speaker"][history_i]}「 {df_transcription["transcript"][history_i]}」'
        result = chain.invoke(
            {
                "format_instructions": format_instructions,
                "history": history,
                "target": target,
                "next": next
            }
        )
        results.append(result['understood_flg'])
        print(f"判定対象の発言: {target}\n直後の発言: {next}\n理解フラグ: {result['understood_flg']}\n")
    
    results.append(None)
    df_transcription["is_understood"] = results 

    print(f"結果を保存中...")
    # df_transcription.to_csv(f"{RESULT_DIR}/result_sample_data.csv", index=False)