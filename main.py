#####################################################################################################
# Python Module for Gen AI Solutions - Coinbase Demo                                                #
# Author: Subhadip Kundu (Jade Global)                                                              #
# --------------------------------------------------------------------------------------------------#
#    Date      |     Author          |                   Comment                                    #
# ------------ + ------------------- + ------------------------------------------------------------ #
# 15-Jun-2024  | Subhadip Kundu      | Created the Initial Code                                     #
#####################################################################################################

import os
import time
from dotenv import load_dotenv
import pandas as pd
from PIL import Image
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
        Keep analysis strictly for business users working in the sales domain to understand nature of output. Limit your response accordingly.
        Few Rules to follow are:
        1. If the result for the query is in tabular format make sure the whole analysis is in same format.
        2. The analysis must be within 80-100 words. 
        3. Do not include supplied data into analysis.
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
    zeroShotAnlyze = zero_shot_analyze_utility(analysis_question, question, "inventory management", metadata)
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

            # Setup the Widgets
            with st.sidebar:
                st.image('assets/JadeGlobal_BW.png', width=150)
                st.markdown("<h1 style='text-align: center;'>AI Finance Assistant</h1>", unsafe_allow_html=True)
                st.write(" \n  \n  \n  \n")

                ### Add a select box to add scope to choose the model
                llm_selected = st.selectbox("Choose a model :", options=["OpenAI - Gpt 4.0 Turbo", "OpenAI - Gpt 4.o", "OpenAI - Gpt 4.0", "OpenAI - Gpt 3.5 Turbo"])
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

            # ### Add button to start the month end process
            # start_process = st.sidebar.button(":white[Start AP Month End]", type="primary", key="AP_Month_End")
            # if start_process:
            #     st.chat_message("user").markdown("Start AP Month End Process", unsafe_allow_html=True)
            #     # st.session_state.messages.append({"role": "user", "content": "Start AP Month End Process"})
            #     st.chat_message("assistant").markdown("Starting the AP Month End Process", unsafe_allow_html=True)
            #     # st.session_state.messages.append({"role": "assistant", "content": "Starting the AP Month End Process"})
            #     post_api_responce = UiPath_API_Queue_Load.add_data_to_queue('Start_Month_End_Process')
            #     st.markdown(post_api_responce)
            #     i = 0
            #     while i <= 80:
            #         get_api_responce = UiPath_API_Queue_Load.read_status_in_queue()
            #         if get_api_responce == 'New':
            #             st.chat_message("assistant").markdown("The process is starting. Please wait for sometime.",unsafe_allow_html=True)
            #             time.sleep(15)
            #             i += 1
            #         elif get_api_responce == 'InProgress':
            #             st.chat_message("assistant").markdown("The process is in progress. Please wait for sometime to get it completed.",unsafe_allow_html=True)
            #             time.sleep(15)
            #             i += 1
            #         elif get_api_responce == 'Successful':
            #             st.chat_message("assistant").markdown("The process has been completed successfully.",unsafe_allow_html=True)
            #             st.session_state.messages.append({"role": "assistant", "content": "The process has been completed successfully."})
            #             break
            #         elif 'Failed' in get_api_responce:
            #             st.markdown("Unable to get the status of the process. Please check the status manually.")
            #             break
            #         else:
            #             st.markdown(f"Following is the status of the process : {get_api_responce}")
            #             break

            ### Add Option menu to select the source
            with st.sidebar:
                select_source = option_menu(menu_title="Menu",
                                            menu_icon="search",
                                            options=["Query your Data", 'Query for Exception Reports', 'AP Month End Process'],
                                            icons=['database', 'filetype-pdf', 'robot'],
                                            default_index=0)

            if select_source == 'Query your Data':
                ### Setup the Home Page
                str_input = st.chat_input("Enter your question:")
                st.markdown("<h2>AI Assistant :</h2>", unsafe_allow_html=True)
                st.markdown("""Welcome! I am Finance Assistant of your company. 
                            I possess the ability to extract information from your company's financial statements like invoice, balance sheet etc. 
                            Please ask me questions and I will try my level best to provide accurate responses.""")

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

            elif select_source == 'Query for Exception Reports':
                str_input = st.chat_input("Enter your question:")
                st.markdown("<h2>AI Assistant :</h2>", unsafe_allow_html=True)
                st.markdown("""Welcome! I'm your AI assistant specialized to extract information from exception report insights. 
                                My purpose is to assist with any queries related to exception reports within your organization.
                                Please enter your questions in the text box below, or you can choose from the list provided on the left panel.
                                I'm here to provide accurate responses to your inquiries.""")

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

                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"], unsafe_allow_html=True)

                if prompt1 := str_input:
                    st.chat_message("user").markdown(prompt1, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "user", "content": prompt1})
                    with st.chat_message("assistant"):
                        result = prompts.letter_chain(str_input)
                        answer = result['result']
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})

            elif select_source == 'AP Month End Process':
                st.markdown("<h2>AI Assistant :</h2>", unsafe_allow_html=True)
                st.markdown("""Welcome! I'm your AI assistant. 
                            My purpose is to start the AP month end process and check the status for you.
                            Please click on the below button, I will trigger the process for you.""")
                start_process = st.button(":white[Start AP Month End Process]", type="primary", key="AP_Month_End")
                if start_process:
                    st.chat_message("user").markdown("Start AP Month End Process", unsafe_allow_html=True)
                    # st.session_state.messages.append({"role": "user", "content": "Start AP Month End Process"})
                    st.chat_message("assistant").markdown("Starting the AP Month End Process", unsafe_allow_html=True)
                    # st.session_state.messages.append({"role": "assistant", "content": "Starting the AP Month End Process"})
                    post_api_responce = UiPath_API_Queue_Load.add_data_to_queue('Start_Month_End_Process')
                    #st.markdown(post_api_responce)
                i = 0
                while i <= 80:
                    get_api_responce = UiPath_API_Queue_Load.read_status_in_queue()
                    if get_api_responce == 'New':
                        st.chat_message("assistant").markdown("The process is starting. Please wait for sometime.", unsafe_allow_html=True)
                        time.sleep(15)
                        i += 1
                    elif get_api_responce == 'InProgress':
                        st.chat_message("assistant").markdown("The process is in progress. Please wait for sometime to get it completed.", unsafe_allow_html=True)
                        time.sleep(15)
                        i += 1
                    elif get_api_responce == 'Successful':
                        st.chat_message("assistant").markdown("The process has been completed successfully.", unsafe_allow_html=True)
                        st.session_state.messages1.append({"role": "assistant", "content": "The process has been completed successfully."})
                        break
                    # elif 'Failed' in get_api_responce:
                    #     st.markdown("Unable to get the status of the process. Please check the status manually.")
                    #     break
                    else:
                        #st.markdown(f"Following is the status of the process : {get_api_responce}")
                        break

    except Exception as err:
        with st.chat_message("assistant"):
            if "messages" not in st.session_state.keys():
                st.session_state.messages = []
            err_msg = "Something Went Wrong - " + str(err)
            st.markdown(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})

if __name__ == "__main__":
    main()
