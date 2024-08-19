#####################################################################################################
# Python Module for Gen AI Solutions - Month End Process                                            #
# Author: Subhadip Kundu (Jade Global)                                                              #
# --------------------------------------------------------------------------------------------------#
#    Date      |     Author          |                   Comment                                    #
# ------------ + ------------------- + ------------------------------------------------------------ #
# 15-Jun-2024  | Subhadip Kundu      | Created the Initial Code                                     #
# 20-Jun-2024  | Subhadip Kundu      | Added Workflow and Dashboard Details                         #
# 24-Jun-2024  | Subhadip Kundu      | Change the Dashboard Details                                 #
#####################################################################################################

import os
import time
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
import snowflake.connector
from langchain.chains import RetrievalQA
from PIL import Image
from io import StringIO
from tabulate import tabulate
import plotly.express as px
from streamlit_option_menu import option_menu

from FewShotSettings import few_shot_settings
from ZeroShotAnalyzeSettings import zero_shot_analyze_settings
import UiPath_API_Queue_Load
import prompts

# Setup the Page
jadeimage = Image.open("assets/jadeglobalsmall.png")
st.set_page_config(page_title="Jade Global", page_icon=jadeimage, layout="wide")

# Declare Global Variables
load_dotenv()
api_key = os.getenv("OpenAI_Secret_Key")
st_UserName = os.getenv("Streamlit_User_Name")
st_Password = os.getenv("Streamlit_User_Credential")
llm_model_name = ""
year_selected = ""
month_selected = ""
prev_month = ""


# Class Few Shot Prompt for Text to SQL
class few_shot_prompt_utility:

    def __init__(self, examples, prefix, suffix, input_variables, example_template, example_variables):
        self.examples = examples
        self.prefix = prefix
        self.suffix = suffix
        self.input_variables = input_variables
        self.example_template = example_template
        self.example_variables = example_variables

    def get_prompt_template(self):
        example_prompt = PromptTemplate(
            input_variables=self.example_variables,
            template=self.example_template
        )
        return example_prompt

    def get_embeddings(self):
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        return embeddings

    def get_example_selector(self, embeddings):
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            self.examples,
            embeddings,
            FAISS,
            k=3
        )
        return example_selector

    def get_prompt(self, question, example_selector, example_prompt):
        prompt_template = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix=self.prefix,
            suffix=self.suffix,
            input_variables=self.input_variables
        )
        return prompt_template


# Class Zero Shot Prompt for Analysis Part
class zero_shot_analyze_utility:

    def __init__(self, question, ask, context, metadata):
        self.question = question
        self.ask = ask
        self.context = context
        self.metadata = metadata

    def get_analyze_prompt(self):
        template, variables = zero_shot_analyze_settings.get_prompt_template(self.ask, self.metadata)
        prompt_template = PromptTemplate(template=template, input_variables=variables)
        prompt_template.format(question=self.question, context=self.context)
        return prompt_template


# Initializing the Few_Shot Utility
def few_shot():
    prefix = few_shot_settings.get_prefix()
    suffix, input_variable = few_shot_settings.get_suffix()
    examples = few_shot_settings.get_examples()
    example_template, example_variables = few_shot_settings.get_example_template()
    fewShot = few_shot_prompt_utility(examples=examples, prefix=prefix, suffix=suffix, input_variables=input_variable,
                                      example_template=example_template, example_variables=example_variables)
    return fewShot

# Initializing the Large Language Model
def large_language_model(model_name):
    llm = ChatOpenAI(model_name=model_name, temperature=0, max_tokens=2000, openai_api_key=api_key)
    return llm

# Function for Text to Sql
def text_to_sql(user_question):
    try:
        question = user_question
        llm = large_language_model(llm_model_name)
        fewShot = few_shot()
        example_prompt = fewShot.get_prompt_template()
        embeddings = fewShot.get_embeddings()
        example_selector = fewShot.get_example_selector(embeddings)
        prompt_template = fewShot.get_prompt(question, example_selector, example_prompt)
        docsearch = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=docsearch.as_retriever(),
                                               chain_type_kwargs={"prompt": prompt_template})
        sql_query = qa_chain({"query": question})['result']
        return sql_query

    except Exception as er:
        return "Error in Text_to_Sql - " + str(er)


# Function to run the SQL query into Snowflake
def run_sql_query(query):
    try:
        snow_con = snowflake.connector.connect(
            account=os.getenv("Snowflake_Account_Name"),
            user=os.getenv("Snowflake_User_Name"),
            password=os.getenv("Snowflake_User_Credential"),
            role=os.getenv("Snowflake_User_Role"),
            warehouse=os.getenv("Snowflake_Warehouse_Name"),
            database=os.getenv("Snowflake_Database_Name"),
            schema=os.getenv("Snowflake_Schema_Name")
        )
        data = pd.read_sql(query, snow_con)
        return data
    except Exception as e:
        out_err = ["Error, Data for the provided question is not available in Database :" + str(e)]
        return out_err


def result_analysis(dataframe, question):
    analysis_question_part1 = '''Provide analysis of the data in tabular format below. \n '''
    analysis_question_prompt = '''\nUse "Ask" and "Metadata" information as supporting data for the analysis. This information is mentioned toward end of this text.
        Keep analysis strictly for business users working in the finance domain to understand nature of output. Limit your response accordingly.
        Few Rules to follow are:
        1. If the result for the query is in tabular format make sure the whole analysis is in same format.
        2. The analysis must be within 80-120 words. 
        3. Do not include supplied data into analysis.
        4. You can add some points from your end if you think that is relevant to the user question. 
        '''
    analysis_question = str(analysis_question_part1) + str(dataframe) + str(analysis_question_prompt)
    fewShot = few_shot()
    llm = large_language_model(llm_model_name)
    embeddings = fewShot.get_embeddings()
    docsearch = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = docsearch.similarity_search(question)
    metadata = ""
    for i in docs:
        metadata = metadata + "\n" + i.page_content
    zeroShotAnlyze = zero_shot_analyze_utility(analysis_question, question, "company finance data", metadata)
    analyze_prompt = zeroShotAnlyze.get_analyze_prompt()
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=docsearch.as_retriever(),
                                           chain_type_kwargs={"prompt": analyze_prompt})
    result = qa_chain({"query": analysis_question})['result']
    return result


def plot_chart(dataFrame):
    df = pd.DataFrame(dataFrame)
    fig = px.bar(df, x=df.columns[0], y=df.columns[-1], color=df.columns[1])
    return st.plotly_chart(fig, width=0, height=300, use_container_width=True)


def format_amount(amount):
    if amount >= 1_000_000:
        return f"$ {amount / 1_000_000:.2f}M"
    elif amount >= 1_000:
        return f"$ {amount / 1_000:.2f}K"
    else:
        return f"$ {amount}"


def chat_history(CSV_FILE):
    try:
        chat_history_df = pd.read_csv(CSV_FILE)
        return chat_history_df
    except FileNotFoundError:
        chat_history_df = pd.DataFrame(columns=["User_Chat_History"])
        chat_history_df.to_csv(CSV_FILE, index=False)
        return chat_history_df

def creds_entered():
    if len(st.session_state["streamlit_username"]) > 0 and len(st.session_state["streamlit_password"]) > 0:
        if st.session_state["streamlit_username"].strip() == st_UserName \
                and st.session_state["streamlit_password"].strip() == st_Password:
            st.session_state["authenticated"] = True
        else:
            st.session_state["authenticated"] = False
            st.error("Invalid Username/Password ")

def authenticate_user():
    if "authenticated" not in st.session_state:
        buff, col, buff2 = st.columns([1, 1, 1])
        col.text_input(label="Username", value="", key="streamlit_username", on_change=creds_entered)
        col.text_input(label="Password", value="", key="streamlit_password", type="password", on_change=creds_entered)
        return False
    else:
        if st.session_state["authenticated"]:
            return True
        else:
            buff, col, buff2 = st.columns([1, 1, 1])
            col.text_input(label="Username:", value="", key="streamlit_username", on_change=creds_entered)
            col.text_input(label="Password:", value="", key="streamlit_password", type="password",
                           on_change=creds_entered)
            return False

def main():
    try:
        if authenticate_user():
            his_file_1 = "Chat_History/jade_Chat_History_Snow.csv"
            his_file_2 = "Chat_History/jade_Chat_History_PDF.csv"
            chat_df_1 = chat_history(his_file_1)
            chat_df_2 = chat_history(his_file_2)
            st.markdown(
                """
                <style>
                    [data-testid=stSidebar] [data-testid=stImage]{
                        text-align: center;
                        display: block;
                        margin-left: auto;
                        margin-right: auto;
                        width: 100%;
                    }
                </style>
                """, unsafe_allow_html=True
            )

            if "messages" not in st.session_state.keys():
                st.session_state.messages = []
            if "messages1" not in st.session_state.keys():
                st.session_state.messages1 = []

            # Workflow Status
            if 'button1' not in st.session_state:
                st.session_state.button1 = 0
            if 'button2' not in st.session_state:
                st.session_state.button2 = 0
            if 'button3' not in st.session_state:
                st.session_state.button3 = 0
            if 'button4' not in st.session_state:
                st.session_state.button4 = 0
            if 'button5' not in st.session_state:
                st.session_state.button5 = 0
            if 'button6' not in st.session_state:
                st.session_state.button6 = 0
            if 'button7' not in st.session_state:
                st.session_state.button7 = 0
            if 'button8' not in st.session_state:
                st.session_state.button8 = 0
            if 'button9' not in st.session_state:
                st.session_state.button9 = 0
            if 'button10' not in st.session_state:
                st.session_state.button10 = 0
            if 'button11' not in st.session_state:
                st.session_state.button11 = 0
            if 'button12' not in st.session_state:
                st.session_state.button12 = 0

            # Setup the Widgets
            with st.sidebar:
                st.image('assets/jadeglobalbig.png', width=200)
                st.markdown("<h1 style='text-align: center;'>FinClose AI</h1>", unsafe_allow_html=True)
                st.write(" \n  \n  \n  \n")

            ### Add Option menu to select the source
            with st.sidebar:
                select_source = option_menu(menu_title="Menu",
                                            menu_icon="list",
                                            options=['Dashboard', 'Query Financial Data', 'Trigger Month End', 'Query Month End Reports'],
                                            icons=['graph-up-arrow', 'database', 'robot', 'filetype-pdf'],
                                            default_index=0)

            if select_source == 'Dashboard':
                st.title("Dashboard :")
                # Month Map
                month_map = {
                    'January': 1, 'February': 2, 'March': 3, 'April': 4,
                    'May': 5, 'June': 6, 'July': 7, 'August': 8,
                    'September': 9, 'October': 10, 'November': 11, 'December': 12
                }

                # Selling Data
                selling_sql_qry = """SELECT
                                        SELLING_YEAR||' '||SELLING_MONTH AS YEAR_MONTH,
                                        SELLING_COST AS REVENUE_AMOUNT
                                    FROM DEMO_DB.SC_FINCLOSE.SELLING_COST 
                                    ORDER BY SELLING_YEAR, 
                                    CASE SELLING_MONTH
                                        WHEN 'January' THEN 1
                                        WHEN 'February' THEN 2
                                        WHEN 'March' THEN 3
                                        WHEN 'April' THEN 4
                                        WHEN 'May' THEN 5
                                        WHEN 'June' THEN 6
                                    END;"""
                selling_sql_result = run_sql_query(selling_sql_qry)
                selling_df = pd.DataFrame(selling_sql_result)
                selling_df[['Year', 'MonthName']] = selling_df['YEAR_MONTH'].str.split(expand=True)
                selling_df['Year'] = selling_df['Year'].astype(int)
                selling_df.columns = selling_df.columns.str.replace('_', ' ')

                # Buying Data
                buying_sql_qry = """SELECT
                                        BUYING_YEAR||' '||BUYING_MONTH AS YEAR_MONTH,
                                        BUYING_COST AS EXPENSES_AMOUNT
                                    FROM DEMO_DB.SC_FINCLOSE.BUYING_COST 
                                    ORDER BY BUYING_YEAR, 
                                    CASE BUYING_MONTH
                                        WHEN 'January' THEN 1
                                        WHEN 'February' THEN 2
                                        WHEN 'March' THEN 3
                                        WHEN 'April' THEN 4
                                        WHEN 'May' THEN 5
                                        WHEN 'June' THEN 6
                                    END;"""
                buying_sql_result = run_sql_query(buying_sql_qry)
                buying_df = pd.DataFrame(buying_sql_result)
                buying_df[['Year', 'MonthName']] = buying_df['YEAR_MONTH'].str.split(expand=True)
                buying_df['Year'] = buying_df['Year'].astype(int)
                buying_df.columns = buying_df.columns.str.replace('_', ' ')

                # Distinct Year and Month List
                year_list = list(buying_df.Year.unique())[::-1]
                month_list = list(buying_df.MonthName.unique())

                # Margin Data
                df_merged = pd.merge(selling_df, buying_df, on=['YEAR MONTH', 'Year', 'MonthName'])
                df_merged['Margin Amount'] = df_merged['REVENUE AMOUNT'] - df_merged['EXPENSES AMOUNT']
                df_merged['Margin Amount'] = round((((df_merged['REVENUE AMOUNT'] - df_merged['EXPENSES AMOUNT']) / df_merged['REVENUE AMOUNT']) * 100), 2)
                df_merged['Month_Num'] = df_merged['MonthName'].map(month_map)
                df_margin = df_merged[['YEAR MONTH', 'Margin Amount', 'Year', 'Month_Num']]

                # DSO Data
                dso_sql_qry = """SELECT 
                                    R.PERIOD_YEAR, 
                                    R.PERIOD_MONTH,
                                    R.PERIOD_YEAR||' '||R.PERIOD_MONTH AS YEAR_MONTH,
                                    ROUND(((R.AMOUNT * DAY(LAST_DAY(TO_DATE(R.PERIOD_MONTH|| ' ' ||R.PERIOD_YEAR, 'MMMM YYYY'))))/S.AMOUNT), 2) AS DSO_AMOUNT
                                FROM 
                                    DEMO_DB.SC_FINCLOSE.DSO AS R
                                INNER JOIN
                                    DEMO_DB.SC_FINCLOSE.DSO AS S
                                ON R.PERIOD_YEAR = S.PERIOD_YEAR
                                AND R.PERIOD_MONTH = S.PERIOD_MONTH
                                AND R.TYPE = 'RECEIVABLES'
                                AND S.TYPE = 'CREDIT_SALES';"""
                dso_sql_result = run_sql_query(dso_sql_qry)
                dso_df = pd.DataFrame(dso_sql_result)
                dso_df.columns = dso_df.columns.str.replace('_', ' ')
                # # Outstanding Receivables Data
                # out_sql_qry = "SELECT AMOUNT AS OUTSTANDING_RECEIVABLES_AMOUNT FROM DEMO_DB.SC_FINCLOSE.OUTSTANDING_RECEIVABLES;"
                # out_sql_result = run_sql_query(out_sql_qry)
                # out_df = pd.DataFrame(out_sql_result)
                # out_df.columns = out_df.columns.str.replace('_', ' ')

                # Dashboard Main Panel
                col = st.columns((3), gap='medium')
                # First Column
                with col[0]:
                    global year_selected
                    year_selected = st.selectbox("Select a year :", options = year_list)
                # Second Column
                with col[1]:
                    global month_selected, prev_month
                    month_selected = st.selectbox("Select a month :", options = month_list)
                    if month_selected == 'January':
                        prev_month = 'January'
                    elif month_selected == 'February':
                        prev_month = 'January'
                    elif month_selected == 'March':
                        prev_month = 'February'
                    elif month_selected == 'April':
                        prev_month = 'March'
                    elif month_selected == 'May':
                        prev_month = 'April'
                    elif month_selected == 'June':
                        prev_month = 'May'
                # Third Column
                with col[2]:
                    st.markdown('')
                    st.markdown('')
                    st.markdown('')
                    st.markdown('')
                    st.markdown('')

                with col[0]:
                    st.subheader("Revenue:", divider='rainbow')
                    # Metric Graph
                    curr_sell_df = \
                        selling_df[(selling_df['Year'] == year_selected) & (selling_df['MonthName'] == month_selected)][
                            'REVENUE AMOUNT'].to_frame().reset_index(drop=True)
                    curr_sell_data = curr_sell_df.loc[0, 'REVENUE AMOUNT']
                    prev_sell_df = \
                        selling_df[(selling_df['Year'] == year_selected) & (selling_df['MonthName'] == prev_month)][
                            'REVENUE AMOUNT'].to_frame().reset_index(drop=True)
                    prev_sell_data = prev_sell_df.loc[0, 'REVENUE AMOUNT']
                    sell_data_diff = str(round((((curr_sell_data - prev_sell_data) / prev_sell_data) * 100), 2)) + '%'
                    curr_sell_amount = format_amount(curr_sell_data)
                    st.metric(label=str(year_selected) + " " + str(month_selected),
                              value=curr_sell_amount,
                              delta=sell_data_diff)
                    # Bar Graph
                    fig = px.bar(selling_df,
                                 x=selling_df.columns[0],
                                 y=selling_df.columns[1],
                                 #color=selling_df.columns[1]
                                 )
                    st.plotly_chart(fig, width=0, height=300, use_container_width=True)

                with col[1]:
                    st.subheader("Expenses:", divider='rainbow')
                    # Metric Graph
                    curr_buy_df = \
                        buying_df[(buying_df['Year'] == year_selected) & (buying_df['MonthName'] == month_selected)][
                            'EXPENSES AMOUNT'].to_frame().reset_index(drop=True)
                    curr_buy_data = curr_buy_df.loc[0, 'EXPENSES AMOUNT']
                    prev_buy_df = \
                        buying_df[(buying_df['Year'] == year_selected) & (buying_df['MonthName'] == prev_month)][
                            'EXPENSES AMOUNT'].to_frame().reset_index(drop=True)
                    prev_buy_data = prev_buy_df.loc[0, 'EXPENSES AMOUNT']
                    buy_data_diff = str(round((((curr_buy_data - prev_buy_data) / prev_buy_data) * 100), 2)) + '%'
                    curr_buy_amount = format_amount(curr_buy_data)
                    st.metric(label=str(year_selected) + " " + str(month_selected),
                              value=curr_buy_amount,
                              delta=buy_data_diff)
                    # Bar Graph
                    fig = px.bar(buying_df,
                                 x=buying_df.columns[0],
                                 y=buying_df.columns[1],
                                 #color=buying_df.columns[1]
                                 )
                    st.plotly_chart(fig, width=0, height=300, use_container_width=True)

                    # DSO Data
                    st.subheader("Days Sales Outstanding:", divider='rainbow')
                    # Metric Graph
                    curr_dso_df = \
                        dso_df[(dso_df['PERIOD YEAR'] == year_selected) & (dso_df['PERIOD MONTH'] == month_selected)][
                            'DSO AMOUNT'].to_frame().reset_index(drop=True)
                    curr_dso_data = curr_dso_df.loc[0, 'DSO AMOUNT']
                    prev_dso_df = \
                        dso_df[(dso_df['PERIOD YEAR'] == year_selected) & (dso_df['PERIOD MONTH'] == prev_month)][
                            'DSO AMOUNT'].to_frame().reset_index(drop=True)
                    prev_dso_data = prev_dso_df.loc[0, 'DSO AMOUNT']
                    dso_data_diff = str(round((((curr_dso_data - prev_dso_data) / prev_dso_data) * 100), 2)) + '%'
                    st.metric(label=str(year_selected) + " " + str(month_selected),
                              value=curr_dso_data,
                              delta=dso_data_diff)
                    # Bar Graph
                    fig = px.bar(dso_df,
                                 x=dso_df.columns[2],
                                 y=dso_df.columns[3],
                                 # color=df_margin.columns[3]
                                 )
                    st.plotly_chart(fig, width=0, height=300, use_container_width=True)

                with col[2]:
                    # Margin Data
                    st.subheader("Margin:", divider='rainbow')
                    prev_margin_data = round((((prev_sell_data - prev_buy_data) / prev_buy_data) * 100), 2)
                    curr_margin_data = round((((curr_sell_data - curr_buy_data) / curr_buy_data) * 100), 2)
                    margin_data_diff = str(round((((curr_margin_data - prev_margin_data) / prev_margin_data) * 100), 2)) + '%'
                    st.metric(label=str(year_selected) + " " + str(month_selected), value=str(curr_margin_data) + '%', delta=margin_data_diff)
                    # Bar Graph
                    fig = px.bar(df_margin,
                                 x=df_margin.columns[0],
                                 y=df_margin.columns[1],
                                 #color=df_margin.columns[1]
                                  )
                    st.plotly_chart(fig, width=0, height=300, use_container_width=True)

            elif select_source == 'Query Financial Data':
                ### Setup the Home Page
                str_input = st.chat_input("Enter your question:")
                st.markdown("<h2>AI Assistant :</h2>", unsafe_allow_html=True)
                st.markdown("""Welcome! I am Finance Assistant of your company. 
                            I possess the ability to extract information from your company's financial statements like expense, invoice, balance sheet etc. 
                            Please ask me questions and I will try my level best to provide accurate responses.""")
                ### Add a select box to add scope to choose the model
                llm_selected = st.selectbox("Choose a model :",
                                            options=["OpenAI - Gpt 4.0 Turbo", "OpenAI - Gpt 4.o", "OpenAI - Gpt 4.0",
                                                     "OpenAI - Gpt 3.5 Turbo"])
                global llm_model_name
                if llm_selected == "OpenAI - Gpt 4.0 Turbo":
                    llm_model_name = "gpt-4"
                elif llm_selected == "OpenAI - Gpt 4.o":
                    llm_model_name = "gpt-4"
                elif llm_selected == "OpenAI - Gpt 4.0":
                    llm_model_name = "gpt-4"
                elif llm_selected == "OpenAI - Gpt 3.5 Turbo":
                    llm_model_name = "gpt-3.5-turbo"
                else:
                    llm_model_name = "gpt-4"

                ### Save the User Chat History
                new_data = {"User_Chat_History": str_input}
                chat_df_1 = chat_df_1._append(new_data, ignore_index=True)
                chat_df_1 = chat_df_1.dropna().drop_duplicates()
                chat_df_1 = chat_df_1.sort_index(axis=0, ascending=False)
                chat_df_1.to_csv(his_file_1, index=False)

                ### Add a button to reset the User Chat History
                chat_reset = st.sidebar.button(":orange[Clear Chat History]", type="secondary", key="Clear_Chat_History")
                if chat_reset:
                    chat_df = pd.DataFrame(columns=["User_Chat_History"])
                    chat_df.to_csv(his_file_1, index=False)

                ### Give flexibility to the User to Select from the Chat History
                with st.sidebar:
                    for index, row in chat_df_1.iterrows():
                        if st.button(f"{row['User_Chat_History']}"):
                            str_input = str(row['User_Chat_History'])

                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        role = message["role"]
                        df_str = message["content"]
                        if role == "user":
                            st.markdown(df_str, unsafe_allow_html=True)
                            continue
                        if df_str.find("<separator>") > -1:
                            csv_str = df_str[:df_str.index("<separator>")]
                            analysis_str = df_str[df_str.index("<separator>") + len("<separator>"):]
                            csv = StringIO(csv_str)
                            df_data = pd.read_csv(csv, sep=',')
                            df_data.columns = df_data.columns.str.replace('_', ' ')
                            headers = df_data.columns
                            st.markdown(f'<p style="font-family:sans-serif; font-size:15px">{analysis_str}</p>',
                                        unsafe_allow_html=True)
                            if len(df_data.index) >= 2 and len(df_data.columns) >= 2 and len(df_data.columns) <= 3:
                                with st.expander("Graph:"):
                                    plot_chart(df_data)
                            with st.expander("Table Output:"):
                                st.markdown(
                                    tabulate(df_data, tablefmt="html", headers=headers, floatfmt=".2f", showindex=False),
                                    unsafe_allow_html=True)
                        else:
                            st.markdown(df_str)

                if prompt := str_input:
                    st.chat_message("user").markdown(prompt, unsafe_allow_html=True)
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("assistant"):
                        i = 0
                        while i <= 4:
                            sql_query = text_to_sql(str_input)
                            sql_result = run_sql_query(sql_query)
                            if "Error" not in str(sql_query) and "Error" not in str(sql_result):
                                break
                            i += 1
                        df = pd.DataFrame(sql_result)
                        df_analysis = str(df)
                        sql_result_analysis = result_analysis(df_analysis, str_input)
                        st.markdown(f'<p style="font-family:sans-serif; font-size:15px">{sql_result_analysis}</p>',
                                    unsafe_allow_html=True)
                        if len(df.index) >= 1 and len(df.columns) >= 2 and len(df.columns) <= 3:
                            with st.expander("Graph:"):
                                plot_chart(df)
                        if "Error" not in df_analysis:
                            df.columns = df.columns.str.replace('_', ' ')
                            headers = df.columns
                            with st.expander("Table Output:"):
                                st.markdown(tabulate(df, tablefmt="html", headers=headers, floatfmt=".2f", showindex=False),
                                            unsafe_allow_html=True)
                            with st.expander("The SQL query used for above question is:"):
                                st.write(sql_query)
                        out_data = df.to_csv(sep=',', index=False) + "<separator>" + sql_result_analysis
                        st.session_state.messages.append({"role": "assistant", "content": out_data})

            elif select_source == 'Query Month End Reports':
                str_input = st.chat_input("Enter your question:")
                st.markdown("<h2>AI Assistant :</h2>", unsafe_allow_html=True)
                st.markdown("""Welcome! I'm your AI assistant specialized to extract information from exception report insights. 
                                My purpose is to assist with any queries related to exception reports within your organization.
                                Please enter your questions in the text box below, or you can choose from the list provided on the left panel.
                                I'm here to provide accurate responses to your inquiries.""")
                ### Add a select box to add scope to choose the model
                llm_selected = st.selectbox("Choose a model :",
                                            options=["OpenAI - Gpt 4.0 Turbo", "OpenAI - Gpt 4.o", "OpenAI - Gpt 4.0",
                                                     "OpenAI - Gpt 3.5 Turbo"])
                if llm_selected == "OpenAI - Gpt 4.0 Turbo":
                    llm_model_name = "gpt-4"
                elif llm_selected == "OpenAI - Gpt 4.o":
                    llm_model_name = "gpt-4"
                elif llm_selected == "OpenAI - Gpt 4.0":
                    llm_model_name = "gpt-4"
                elif llm_selected == "OpenAI - Gpt 3.5 Turbo":
                    llm_model_name = "gpt-3.5-turbo"
                else:
                    llm_model_name = "gpt-4"

                new_data = {"User_Chat_History": str_input}
                chat_df_2 = chat_df_2._append(new_data, ignore_index=True)
                chat_df_2 = chat_df_2.dropna().drop_duplicates()
                chat_df_2 = chat_df_2.sort_index(axis=0, ascending=False)
                chat_df_2.to_csv(his_file_2, index=False)
                chat_reset = st.sidebar.button(":orange[Clear Chat History]", type="secondary",
                                               key="Clear_Chat_History")
                if chat_reset:
                    chat_df_2 = pd.DataFrame(columns=["User_Chat_History"])
                    chat_df_2.to_csv(his_file_2, index=False)

                with st.sidebar:
                    for index, row in chat_df_2.iterrows():
                        if st.button(f"{row['User_Chat_History']}"):
                            str_input = str(row['User_Chat_History'])

                for message in st.session_state.messages1:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"], unsafe_allow_html=True)

                if prompt1 := str_input:
                    st.chat_message("user").markdown(prompt1, unsafe_allow_html=True)
                    st.session_state.messages1.append({"role": "user", "content": prompt1})
                    with st.chat_message("assistant"):
                        result = prompts.letter_chain(str_input)
                        answer = result['result']
                        st.markdown(answer)
                        st.session_state.messages1.append({"role": "assistant", "content": answer})

            elif select_source == 'Trigger Month End':
                st.markdown("<h2>AI Assistant :</h2>", unsafe_allow_html=True)
                st.markdown("""Welcome! I'm your AI assistant. 
                            My purpose is to start the AP month end process and check the status for you.
                            Please click on the below button, I will trigger the process for you.""")

                # Workflow buttons
                # Create a horizontal layout with columns
                col1, col2, col3 = st.columns(3)
                # First horizontal layout
                with col1:
                    # Create a container for the buttons
                    with st.container(border=True, height=600):
                        st.subheader("PeriodClose -3", divider='rainbow')
                        # First Button
                        st.markdown(":grey-background[**Create Accounting**]")
                        if st.session_state.button1 == 1:
                            my_bar31 = st.success('The Process has completed successfully!', icon="✅")
                        else:
                            my_bar31 = st.info("Process yet to be Start", icon="ℹ️")
                        if st.session_state.button1 == 1:
                            if st.button("RUN ▶️", key="Create_Accounting_RERUN", disabled = True):
                                for percent_complete in range(100):
                                    time.sleep(0.05)
                                    my_bar31.progress(percent_complete + 1, text="Operation is in progress. Please wait for sometime...")
                                st.session_state.button1 = 1
                                my_bar31.success('The Process has completed successfully!', icon="✅")
                        else:
                            if st.button("RUN ▶️", key="Create_Accounting_RUN"):
                                st.session_state.button1 = 1
                                UiPath_API_Queue_Load.add_data_to_queue('Create Accounting')
                                for percent_complete in range(100):
                                    queue_item_status, queue_item_progress = UiPath_API_Queue_Load.read_status_in_queue()
                                    if queue_item_status in ('New', 'InProgress'):
                                        my_bar31.progress(percent_complete + 1,
                                                          text="Operation is in progress. Please wait for sometime...")
                                        time.sleep(2)
                                    elif queue_item_status == 'Successful':
                                        my_bar31.success('The Process has completed successfully!', icon="✅")
                                        break
                                    else:
                                        my_bar31.error('Something went wrong. Please check the process', icon="⚠️")
                                        break

                    with st.container(border=True, height=600):
                        st.subheader("PeriodClose +1", divider='rainbow')
                        # First Button
                        st.markdown(":grey-background[**GL Transfer**]")
                        if st.session_state.button2 == 1:
                            my_bar31 = st.success('The Process has completed successfully!', icon="✅")
                        else:
                            my_bar31 = st.info("Process yet to be Start", icon="ℹ️")
                        if st.session_state.button2 == 1:
                            if st.button("RUN ▶️", key="GL_Transfer_RERUN", disabled = True):
                                for percent_complete in range(100):
                                    time.sleep(0.05)
                                    my_bar31.progress(percent_complete + 1, text="Operation is in progress. Please wait for sometime...")
                                st.session_state.button2 = 1
                                my_bar31.success('The Process has completed successfully!', icon="✅")
                        else:
                            if st.button("RUN ▶️", key="GL_Transfer_RUN"):
                                st.session_state.button2 = 1
                                UiPath_API_Queue_Load.add_data_to_queue('GL Transfer')
                                for percent_complete in range(100):
                                    queue_item_status, queue_item_progress = UiPath_API_Queue_Load.read_status_in_queue()
                                    if queue_item_status in ('New', 'InProgress'):
                                        my_bar31.progress(percent_complete + 1,
                                                          text="Operation is in progress. Please wait for sometime...")
                                        time.sleep(2)
                                    elif queue_item_status == 'Successful':
                                        my_bar31.success('The Process has completed successfully!', icon="✅")
                                        break
                                    else:
                                        my_bar31.error('Something went wrong. Please check the process', icon="⚠️")
                                        break

                        # Second Button
                        st.markdown(":grey-background[**Trial Balance Report**]")
                        if st.session_state.button3 == 1:
                            my_bar31 = st.success('The Process has completed successfully!', icon="✅")
                        else:
                            my_bar31 = st.info("Process yet to be Start", icon="ℹ️")
                        if st.session_state.button3 == 1:
                            if st.button("RUN ▶️", key="Trial_Balance_Report_RERUN", disabled = True):
                                for percent_complete in range(100):
                                    time.sleep(0.05)
                                    my_bar31.progress(percent_complete + 1, text="Operation is in progress. Please wait for sometime...")
                                st.session_state.button3 = 1
                                my_bar31.success('The Process has completed successfully!', icon="✅")
                        else:
                            if st.button("RUN ▶️", key="Trial_Balance_Report_RUN"):
                                st.session_state.button3 = 1
                                UiPath_API_Queue_Load.add_data_to_queue('Trial Balance Report')
                                for percent_complete in range(100):
                                    queue_item_status, queue_item_progress = UiPath_API_Queue_Load.read_status_in_queue()
                                    if queue_item_status in ('New', 'InProgress'):
                                        my_bar31.progress(percent_complete + 1,
                                                          text="Operation is in progress. Please wait for sometime...")
                                        time.sleep(2)
                                    elif queue_item_status == 'Successful':
                                        my_bar31.success('The Process has completed successfully!', icon="✅")
                                        break
                                    else:
                                        my_bar31.error('Something went wrong. Please check the process', icon="⚠️")
                                        break

                # Second horizontal layout
                with col2:
                    # Create a container for the buttons
                    with st.container(border=True, height=600):
                        st.subheader("PeriodClose -2", divider='rainbow')
                        # First Button
                        st.markdown(":grey-background[**Accounting Reconciliation**]")
                        if st.session_state.button4 == 1:
                            my_bar31 = st.success('The Process has completed successfully!', icon="✅")
                        else:
                            my_bar31 = st.info("Process yet to be Start", icon="ℹ️")
                        if st.session_state.button4 == 1:
                            if st.button("RUN ▶️", key="Accounting_Reconciliation_RERUN", disabled = True):
                                for percent_complete in range(100):
                                    time.sleep(0.05)
                                    my_bar31.progress(percent_complete + 1, text="Operation is in progress. Please wait for sometime...")
                                st.session_state.button4 = 1
                                my_bar31.success('The Process has completed successfully!', icon="✅")
                        else:
                            if st.button("RUN ▶️", key="Accounting_Reconciliation_RUN"):
                                st.session_state.button4 = 1
                                UiPath_API_Queue_Load.add_data_to_queue('Reconciliation')
                                for percent_complete in range(100):
                                    queue_item_status, queue_item_progress = UiPath_API_Queue_Load.read_status_in_queue()
                                    if queue_item_status in ('New', 'InProgress'):
                                        my_bar31.progress(percent_complete + 1,
                                                          text="Operation is in progress. Please wait for sometime...")
                                        time.sleep(2)
                                    elif queue_item_status == 'Successful':
                                        my_bar31.success('The Process has completed successfully!', icon="✅")
                                        break
                                    else:
                                        my_bar31.error('Something went wrong. Please check the process', icon="⚠️")
                                        break
                        # Second Button
                        st.markdown(":grey-background[**IT Accrual check**]")
                        if st.session_state.button5 == 1:
                            my_bar31 = st.success('The Process has completed successfully!', icon="✅")
                        else:
                            my_bar31 = st.info("Process yet to be Start", icon="ℹ️")
                        if st.session_state.button5 == 1:
                            if st.button("RUN ▶️", key="IT_Accrual_check_RERUN", disabled = True):
                                for percent_complete in range(100):
                                    time.sleep(0.05)
                                    my_bar31.progress(percent_complete + 1,
                                                      text="Operation is in progress. Please wait for sometime...")
                                st.session_state.button5 = 1
                                my_bar31.success('The Process has completed successfully!', icon="✅")
                        else:
                            if st.button("RUN ▶️", key="IT_Accrual_check_RUN"):
                                for percent_complete in range(100):
                                    time.sleep(0.05)
                                    my_bar31.progress(percent_complete + 1,
                                                      text="Operation is in progress. Please wait for sometime...")
                                st.session_state.button5 = 1
                                my_bar31.success('The Process has completed successfully!', icon="✅")

                    with st.container(border=True, height=600):
                        st.subheader("PeriodClose +2", divider='rainbow')
                        # First Button
                        st.markdown(":grey-background[**Inventory Recon**]")
                        if st.session_state.button6 == 1:
                            my_bar31 = st.success('The Process has completed successfully!', icon="✅")
                        else:
                            my_bar31 = st.info("Process yet to be Start", icon="ℹ️")
                        if st.session_state.button6 == 1:
                            if st.button("RUN ▶️", key="Inventory_Recon_RERUN", disabled = True):
                                for percent_complete in range(100):
                                    time.sleep(0.05)
                                    my_bar31.progress(percent_complete + 1, text="Operation is in progress. Please wait for sometime...")
                                st.session_state.button6 = 1
                                my_bar31.success('The Process has completed successfully!', icon="✅")
                        else:
                            if st.button("RUN ▶️", key="Inventory_Recon_RUN"):
                                for percent_complete in range(100):
                                    time.sleep(0.05)
                                    my_bar31.progress(percent_complete + 1, text="Operation is in progress. Please wait for sometime...")
                                st.session_state.button6 = 1
                                my_bar31.success('The Process has completed successfully!', icon="✅")

                        # Second Button
                        st.markdown(":grey-background[**Invoice Aging Check**]")
                        if st.session_state.button7 == 1:
                            my_bar31 = st.success('The Process has completed successfully!', icon="✅")
                        else:
                            my_bar31 = st.info("Process yet to be Start", icon="ℹ️")
                        if st.session_state.button7 == 1:
                            if st.button("RUN ▶️", key="Invoice_Aging_Check_RERUN", disabled = True):
                                for percent_complete in range(100):
                                    time.sleep(0.05)
                                    my_bar31.progress(percent_complete + 1, text="Operation is in progress. Please wait for sometime...")
                                st.session_state.button7 = 1
                                my_bar31.success('The Process has completed successfully!', icon="✅")
                        else:
                            if st.button("RUN ▶️", key="Invoice_Aging_Check_RUN"):
                                st.session_state.button7 = 1
                                UiPath_API_Queue_Load.add_data_to_queue('Invoice Aging')
                                for percent_complete in range(100):
                                    queue_item_status, queue_item_progress = UiPath_API_Queue_Load.read_status_in_queue()
                                    if queue_item_status in ('New', 'InProgress'):
                                        my_bar31.progress(percent_complete + 1,
                                                          text="Operation is in progress. Please wait for sometime...")
                                        time.sleep(2)
                                    elif queue_item_status == 'Successful':
                                        my_bar31.success('The Process has completed successfully!', icon="✅")
                                        break
                                    else:
                                        my_bar31.error('Something went wrong. Please check the process', icon="⚠️")
                                        break

                        # Third Button
                        st.markdown(":grey-background[**Tax and Treasury Analysis**]")
                        if st.session_state.button8 == 1:
                            my_bar31 = st.success('The Process has completed successfully!', icon="✅")
                        else:
                            my_bar31 = st.info("Process yet to be Start", icon="ℹ️")
                        if st.session_state.button8 == 1:
                            if st.button("RUN ▶️", key="Tax_and_Treasury_Analysis_RERUN", disabled = True):
                                for percent_complete in range(100):
                                    time.sleep(0.05)
                                    my_bar31.progress(percent_complete + 1, text="Operation is in progress. Please wait for sometime...")
                                st.session_state.button8 = 1
                                my_bar31.success('The Process has completed successfully!', icon="✅")
                        else:
                            if st.button("RUN ▶️", key="Tax_and_Treasury_Analysis_RUN"):
                                for percent_complete in range(100):
                                    time.sleep(0.05)
                                    my_bar31.progress(percent_complete + 1, text="Operation is in progress. Please wait for sometime...")
                                st.session_state.button8 = 1
                                my_bar31.success('The Process has completed successfully!', icon="✅")

                    # Third horizontal layout
                    with col3:
                        with st.container(border=True, height=600):
                            st.subheader("PeriodClose -1", divider='rainbow')
                            # First Button
                            st.markdown(":grey-background[**Open PO and GL period**]")
                            if st.session_state.button9 == 1:
                                my_bar31 = st.success('The Process has completed successfully!', icon="✅")
                            else:
                                my_bar31 = st.info("Process yet to be Start", icon="ℹ️")
                            if st.session_state.button9 == 1:
                                if st.button("RUN ▶️", key="Open_PO_and_GL_period_RERUN", disabled = True):
                                    for percent_complete in range(100):
                                        time.sleep(0.05)
                                        my_bar31.progress(percent_complete + 1, text="Operation is in progress. Please wait for sometime...")
                                    st.session_state.button9 = 1
                                    my_bar31.success('The Process has completed successfully!', icon="✅")
                            else:
                                if st.button("RUN ▶️", key="Open_PO_and_GL_period_RUN"):
                                    for percent_complete in range(100):
                                        time.sleep(0.05)
                                        my_bar31.progress(percent_complete + 1, text="Operation is in progress. Please wait for sometime...")
                                    st.session_state.button9 = 1
                                    my_bar31.success('The Process has completed successfully!', icon="✅")

                            # Second Button
                            st.markdown(":grey-background[**Unaccounted transaction check**]")
                            if st.session_state.button10 == 1:
                                my_bar31 = st.success('The Process has completed successfully!', icon="✅")
                            else:
                                my_bar31 = st.info("Process yet to be Start", icon="ℹ️")
                            if st.session_state.button10 == 1:
                                if st.button("RUN ▶️", key="Unaccounted_transaction_check_RERUN", disabled = True):
                                    for percent_complete in range(100):
                                        time.sleep(0.05)
                                        my_bar31.progress(percent_complete + 1, text="Operation is in progress. Please wait for sometime...")
                                    st.session_state.button10 = 1
                                    my_bar31.success('The Process has completed successfully!', icon="✅")
                            else:
                                if st.button("RUN ▶️", key="Unaccounted_transaction_check_RUN"):
                                    st.session_state.button10 = 1
                                    UiPath_API_Queue_Load.add_data_to_queue('Unaccounted Transaction Report')
                                    for percent_complete in range(100):
                                        queue_item_status, queue_item_progress = UiPath_API_Queue_Load.read_status_in_queue()
                                        if queue_item_status in ('New', 'InProgress'):
                                            my_bar31.progress(percent_complete + 1,
                                                              text="Operation is in progress. Please wait for sometime...")
                                            time.sleep(2)
                                        elif queue_item_status == 'Successful':
                                            my_bar31.success('The Process has completed successfully!', icon="✅")
                                            break
                                        else:
                                            my_bar31.error('Something went wrong. Please check the process', icon="⚠️")
                                            break

                            # Third Button
                            st.markdown(":grey-background[**Exception Correction**]")
                            if st.session_state.button11 == 1:
                                my_bar31 = st.success('The Process has completed successfully!', icon="✅")
                            else:
                                my_bar31 = st.info("Process yet to be Start", icon="ℹ️")
                            if st.session_state.button11 == 1:
                                if st.button("RUN ▶️", key="Exception_Correction_RERUN", disabled = True):
                                    for percent_complete in range(100):
                                        time.sleep(0.05)
                                        my_bar31.progress(percent_complete + 1, text="Operation is in progress. Please wait for sometime...")
                                    st.session_state.button11 = 1
                                    my_bar31.success('The Process has completed successfully!', icon="✅")
                            else:
                                if st.button("RUN ▶️", key="Exception_Correction_RUN"):
                                    st.session_state.button11 = 1
                                    UiPath_API_Queue_Load.add_data_to_queue('Exception Processing')
                                    for percent_complete in range(100):
                                        queue_item_status, queue_item_progress = UiPath_API_Queue_Load.read_status_in_queue()
                                        if queue_item_status in ('New', 'InProgress'):
                                            my_bar31.progress(percent_complete + 1,
                                                              text="Operation is in progress. Please wait for sometime...")
                                            time.sleep(2)
                                        elif queue_item_status == 'Successful':
                                            my_bar31.success('The Process has completed successfully!', icon="✅")
                                            break
                                        else:
                                            my_bar31.error('Something went wrong. Please check the process', icon="⚠️")
                                            break

                        with st.container(border=True, height=600):
                            st.subheader("PeriodClose +3", divider='rainbow')
                            # First Button
                            st.markdown(":grey-background[**Close Period**]")
                            if st.session_state.button12 == 1:
                                my_bar31 = st.success('The Process has completed successfully!', icon="✅")
                            else:
                                my_bar31 = st.info("Process yet to be Start", icon="ℹ️")
                            if st.session_state.button12 == 1:
                                if st.button("RUN ▶️", key="Close_Period_RERUN", disabled = True):
                                    for percent_complete in range(100):
                                        time.sleep(0.05)
                                        my_bar31.progress(percent_complete + 1, text="Operation is in progress. Please wait for sometime...")
                                    st.session_state.button12 = 1
                                    my_bar31.success('The Process has completed successfully!', icon="✅")
                            else:
                                if st.button("RUN ▶️", key="Close_Period_RUN"):
                                    for percent_complete in range(100):
                                        time.sleep(0.05)
                                        my_bar31.progress(percent_complete + 1, text="Operation is in progress. Please wait for sometime...")
                                    st.session_state.button12 = 1
                                    my_bar31.success('The Process has completed successfully!', icon="✅")

    except Exception as err:
        with st.chat_message("assistant"):
            if "messages" not in st.session_state.keys():
                st.session_state.messages = []
            err_msg = "Something Went Wrong - " + str(err)
            st.markdown(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})

if __name__ == "__main__":
    main()
