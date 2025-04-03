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
from sklearn.metrics import accuracy_score, precision_score, recall_score

def read_transcription(file_name: str):
    if file_name.endswith(".csv"):
        return pd.read_csv(f"{PROCESSED_DIR}/{file_name}")
    elif file_name.endswith('.xlsx'):
        return pd.read_excel(f"{PROCESSED_DIR}/{file_name}")
    else:
        raise ValueError(f"Invalid file extension: {file_name}")
    
########################################################
# 結論ファーストの判定
########################################################
def judge_conclusion_first(file_name: str):
    # 文字起こしデータの読み込み
    print(f"文字起こしデータの読み込み中...")
    df_transcription = read_transcription(file_name)
    
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

    df_transcription["conc_1st_flg"] = results 

    print(f"結果を保存中...")
    df_transcription.to_csv(f"{RESULT_DIR}/{file_name.split('.')[0]}_conc_1st.csv", index=False)

########################################################
# フィラーの判定
########################################################
def judge_filler(file_name: str):
    # 文字起こしデータの読み込み
    print(f"文字起こしデータの読み込み中...")
    df_transcription = read_transcription(file_name)
    
    # プロンプトテンプレートの作成
    template = ChatPromptTemplate([
        ("system", "あなたは、WEB会議の質を高めるためのコンサルタントです。あなたは、userから与えられたWEB会議内の発言に「えー」「えっと」「えーっと」「あー」「そのー」「んーと」やその他のフィラーが含まれていれば1、そうでなければ0として出力してください。また、フィラーとして判定された単語も全て出力してください。さらに、発言に含まれるフィラーの個数も出力してください。\n{format_instructions}"),
        ("human", "{user_input}"),
    ])
    # モデルの設定
    model = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key= OPENAI_API_KEY)
    # 出力形式の設定
    filler_flg_schema = ResponseSchema(name="filler_flg", description="2値のフィラーフラグ。フィラーが含まれていれば1、そうでなければ0とする。")
    filler_words_schema = ResponseSchema(name="filler_words", description="フィラーとして判定された単語。複数個ある場合は、カンマ区切りで出力してください。存在しない場合は、空文字を出力してください。")
    filler_num_schema = ResponseSchema(name="filler_num", description="発言に含まれるフィラーの個数。整数のみ出力してください。")
    output_parser = StructuredOutputParser.from_response_schemas([filler_flg_schema, filler_words_schema, filler_num_schema])
    format_instructions = output_parser.get_format_instructions()

    # チェーンの作成
    chain = template | model | output_parser

    # 各発言に対してチェーンの実行
    print(f"使用するプロンプト:\n{template}")
    print(f"各発言に対してチェーンの実行中...")
    results_flg = []
    results_words = []
    results_num = []
    df_transcription["filler_flg_true"] = (df_transcription["filler_num_true"]>0).astype(int)
    for _, row in df_transcription.iterrows():
        text = row["transcript"]
        result = chain.invoke(
            {
                "format_instructions": format_instructions,
                "user_input": text
            }
        )
        results_flg.append(int(result['filler_flg']))
        results_words.append(result['filler_words'])
        results_num.append(int(result['filler_num']))
        print(f"発言: {text}\nフィラー有無(予測): {result['filler_flg']}\nフィラー有無(正解): {row['filler_flg_true']}\n抽出されたフィラー: {result['filler_words']}\nフィラー個数(予測): {result['filler_num']}\nフィラー個数(正解): {row['filler_num_true']}\n")

    df_transcription["filler_num"] = results_num
    df_transcription["filler_flg"] = results_flg
    df_transcription["filler_words"] = results_words
    # 正解率の計算
    pred = df_transcription['filler_flg']
    true = df_transcription['filler_flg_true']
    print(f"正解率: {accuracy_score(true, pred)*100}%")
    print(f"適合率: {precision_score(true, pred)*100}%")
    print(f"再現率: {recall_score(true, pred)*100}%")

    print(f"結果を保存中...")
    df_transcription.to_csv(f"{RESULT_DIR}/{file_name.split('.')[0]}_filler.csv", index=False)


########################################################
# 理解の判定
########################################################
def judge_understood(file_name: str):
    # 文字起こしデータの読み込み
    print(f"文字起こしデータの読み込み中...")
    df_transcription = read_transcription(file_name)

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
    df_transcription["understood_flg"] = results 

    print(f"結果を保存中...")
    df_transcription.to_csv(f"{RESULT_DIR}/{file_name.split('.')[0]}_understood.csv", index=False)

########################################################
# 逸脱の判定
########################################################
def judge_deviation(file_name: str):
    # 文字起こしデータの読み込み
    print(f"文字起こしデータの読み込み中...")
    df_transcription = read_transcription(file_name)

    # プロンプトテンプレートの作成
    template = ChatPromptTemplate([
        ("system", "あなたは、WEB会議の質を高めるためのコンサルタントです。あなたは、userから与えられたWEB会議内の発言履歴を元に、判定対象の発言が、逸脱のきっかけとなる発言であるかどうかを判定してください。さらに、判定対象の発言が本題から逸脱した内容であるかどうかも判定してください。なお、本題からの逸脱とは、会議を生産的に進める上で必要のない発言のことであり、その判断材料として判定対象の発言以前の会話履歴がuserから与えられます。\n{format_instructions}"),
        ("human", "判定対象の発言以前の会話履歴：\n{history}\n判定対象の発言：\n{target}"),
    ])
    # モデルの設定
    model = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key= OPENAI_API_KEY)
    # 出力形式の設定
    deviation_init_schema = ResponseSchema(name="deviation_init_flg", description="2値の逸脱きっかけフラグ。判定対象の発言が逸脱のきっかけとなる発言であれば1、そうでなければ0とする。")
    deviation_schema = ResponseSchema(name="deviation_flg", description="2値の逸脱フラグ。判定対象の発言が本題から逸脱していれば1、そうでなければ0とする。")
    output_parser = StructuredOutputParser.from_response_schemas([deviation_init_schema, deviation_schema])
    format_instructions = output_parser.get_format_instructions()

    # チェーンの作成
    chain = template | model | output_parser

    # 各発言に対してチェーンの実行
    print(f"各発言に対してチェーンの実行中...")
    results_deviation_init = []
    results_deviation = []

    for i in range(len(df_transcription)):
        target = f"{df_transcription['speaker'][i]}「 {df_transcription['transcript'][i]}」"
        history = ""
        for history_i in range(i):
            history += f"{df_transcription['speaker'][history_i]}「 {df_transcription['transcript'][history_i]}」"
        result = chain.invoke(
            {
                "format_instructions": format_instructions,
                "history": history,
                "target": target
            }
        )
        results_deviation_init.append(result['deviation_init_flg'])
        results_deviation.append(result['deviation_flg'])
        print(f"判定対象の発言: {target}\n逸脱きっかけフラグ: {result['deviation_init_flg']}\n逸脱フラグ: {result['deviation_flg']}\n")
    
    df_transcription["deviation_init_flg"] = results_deviation_init
    df_transcription["deviation_flg"] = results_deviation

    print(f"結果を保存中...")
    df_transcription.to_csv(f"{RESULT_DIR}/{file_name.split('.')[0]}_deviation.csv", index=False)