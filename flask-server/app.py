# pylint: disable=E0401
# pylint: disable=W0611

import os

from flask import Flask, redirect, render_template, request, url_for, jsonify
from flask_cors import CORS

import status  # HTTP Status Codes
from dotenv import load_dotenv

import openai

from langchain import SerpAPIWrapper
from langchain.llms import OpenAI

# from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.chains import (
    ConversationChain,
    ConversationalRetrievalChain,
    LLMChain,
    RetrievalQA,
)
from langchain.agents import (
    OpenAIFunctionsAgent,
    AgentExecutor,
    initialize_agent,
    Tool,
    AgentType,
)
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma

######################################################################
# Initialize
######################################################################
load_dotenv()
app = Flask(__name__)
CORS(app)
os.environ["OPENAI_API_KEY"] = "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"
os.environ[
    "SERPAPI_API_KEY"
] = "1395cc6c7eaf7514e460050cc27d499a252f2ffbe79070bcb0fa9e02f9055524"

# memory
agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
}
agent_memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

# Load Unstructured data
loader = PyPDFLoader("nba.pdf")
documents = loader.load()

# Splitting data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# split it into chunks
splitted_docs = text_splitter.split_documents(documents)

# Create the embedding function
embedding_function = OpenAIEmbeddings()

# Store into vector store Chroma
vector_space = Chroma.from_documents(
    documents=splitted_docs, embedding=embedding_function
)

# Make as retriever
retriever = vector_space.as_retriever()


######################################################################
# API using langchain with agent
######################################################################
@app.route("/langchain/chat/agent/<string:send_message>", methods=["GET"])
def langchain_agent_get_response(send_message):
    """
    Input the send_message
    Output the response
    """
    # 第一次filter
    # filter_check, filter_message = filter(send_message)
    # if not filter_check:
    #     return jsonify(filter_message), status.HTTP_200_OK

    # 搜尋是否有預設答案
    # faq_check, faq_message = search_template_faq(filter_message)
    # if faq_check:
    #     opt_message = optimizer(faq_message)
    #     return jsonify(opt_message), status.HTTP_200_OK

    # 用google搭配db找答案
    agent_check, agent_message = search_db_google(send_message)
    if not agent_check:
        # 把無答案存入faq template??
        return jsonify(agent_message), status.HTTP_200_OK

    # 是否要再filter一次???
    # last_filter_check, last_filter_message = validate_preprocess_and_check_relevance(
    #     agent_message
    # )
    # if not last_filter_check:
    #     return jsonify(last_filter_message), status.HTTP_200_OK

    # 輸出前優化
    # opt_response = optimizer(last_filter_message)

    # 將答案存回FAQ
    # store_response(send_message, opt_response)

    # 輸出
    return jsonify(agent_message), status.HTTP_200_OK


def filter(send_message):
    """
    Input the send_message
    First, use GPT to determine the relation.
    Secondly, use DB to determine the relation.
    Return the relation.
    """
    # Basic Preprocessing
    send_message = send_message.strip().lower()

    # Use GPT to Evaluate Relevance
    filter_template = """
        Please evaluate the following description for relevance to the topic of [Your Topic]. If it is completely related, say 'True'. If it is not related at all, say 'False'. If it might be related but you need more information to determine, say 'More Information Needed'.
        ----------------------
        Description: {send_message}
    """
    filter_messages = [
        SystemMessagePromptTemplate.from_template(filter_template),
        HumanMessagePromptTemplate.from_template("{send_message}"),
    ]
    filter_prompt = ChatPromptTemplate.from_messages(filter_messages)
    filter_llm = OpenAI(model_name="gpt-3.5-turbo", temperature=1)
    filter_chain = LLMChain(
        llm=filter_llm,
        prompt=filter_prompt,
        verbose=True,
    )
    filter_response = filter_chain({"send_message": send_message})

    if filter_response["text"] == "False" or filter_response["text"] == "false":
        filter_error = """
            Sorry, this is not related to [Your Topic].
        """
        return False, filter_error
    if filter_response["text"] == "True" or filter_response["text"] == "true":
        return True, send_message
    filter_suspect = """
        Please provide more details or clarify your question.
    """
    return "More Information Needed", filter_suspect


def search_template_faq(send_message):
    """
    Input the send_message.
    Search the FAQ.
    Return the answer.
    """
    faq_template = """
    Using the below description, please search the database for any matching questions and answers (FAQ) that is related to and can answer the description. Return the relevant results, if available. If cannot find any match, just return 'False'.
    ----------------------
    Description: {send_message}
    """
    faq_messages = [
        SystemMessagePromptTemplate.from_template(faq_template),
        HumanMessagePromptTemplate.from_template("{send_message}"),
    ]
    faq_prompt = ChatPromptTemplate.from_messages(faq_messages)
    faq_llm = OpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.2)
    faq_chain = ConversationalRetrievalChain.from_llm(
        llm=faq_llm,
        retriever=faq_retriever,
        condense_question_prompt=faq_prompt,
        verbose=True,
    )
    faq_response = faq_chain({"send_message": send_message})

    if faq_response["answer"] == "False" or faq_response["answer"] == "false":
        return False, send_message
    return True, faq_response["answer"]


def search_db_google(send_message):
    """
    Input the send_message
    Use agent to determine whether use DB, Google, GPT to get the response.
    Return the response.
    """
    agent_template = """
        You are a very powerful assistant.
        While answering questions, you will also need to take the previous conversation into consideration. The order of searching information for the question is first search in the database, if you cannot find useful information in database, then go search in Google, finally if you still cannot find any useful information in both database or Google, then answer it by your own knowledge. If you don't know the answer, say you don't, don't try to make it up.
        {input}
    """
    agent_prompt = PromptTemplate(
        input_variables=["input"],
        template=agent_template,
    )
    agent_search = SerpAPIWrapper()
    agent_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
    agent_db = RetrievalQA.from_chain_type(
        llm=agent_llm, chain_type="stuff", retriever=retriever
    )
    agent_tools = [
        Tool(
            name="search-in-the-database",
            func=agent_db.run,
            description="useful for when you want to find answer in the database.",
        ),
        Tool(
            name="search-in-Google",
            func=agent_search.run,
            description="useful for when you cannot find answer in the database and need to search.",
        ),
    ]
    agent = initialize_agent(
        tools=agent_tools,
        llm=agent_llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        prompt=agent_prompt,
        agent_kwargs=agent_kwargs,
        memory=agent_memory,
    )
    agent_response = agent.run(input=send_message)

    if agent_response == "False" or agent_response == "false":
        return False, agent_response
    return True, agent_response


def optimizer(send_message):
    """
    Input the send_message
    Use GPT to optimize the send_message.
    Return the opt_message.
    """
    return send_message


def store_response(send_message, opt_response):
    """
    Input the send_message and opt_response
    Store them into FAQ.
    """
    return
