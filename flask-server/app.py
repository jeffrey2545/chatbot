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
from langchain.chains import ConversationChain, ConversationalRetrievalChain, LLMChain, RetrievalQA
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor, initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
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
os.environ["SERPAPI_API_KEY"] = "1395cc6c7eaf7514e460050cc27d499a252f2ffbe79070bcb0fa9e02f9055524"

# memory and message related
messages = [
    {"role": "assistant", "content": "You are a NBA expert."}
]

# TO DO
# 試著改成用memory而不是chat_history，可以研究是否要用ConversationBufferMemory或是有其他更好的套件，同時也研究參數怎麼放
# 記得在改成memory時，147行要刪掉，163行要加上memory，175行要刪掉，172行要修改
chat_history = []
memory_test = ConversationBufferMemory()# Initialize memory buffer
memory_test2 = ConversationBufferMemory(memory_key="chat_history", return_messages=True)# Initialize memory buffer
agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name = "memory")],
}
memory = ConversationBufferMemory(memory_key = "memory", return_messages = True)

# Load Unstructured data
loader = PyPDFLoader("The_Three_Little_Pigs.pdf")# load the document
# loader = UnstructuredHTMLLoader("apple.html")
documents = loader.load()

# TO DO
# 研究是否有更好的spliiter，以及chunk_size和chunk_overlap要設多少比較適合
# Splitting data
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)# split it into chunks
splitted_docs = text_splitter.split_documents(documents)

# TO DO
# 研究是否有更好的embedding function
# Create the embedding function
embedding_function = OpenAIEmbeddings()

# Store into vector store
vector_space = Chroma.from_documents(documents = splitted_docs, embedding = embedding_function)# Embed and store the splits into Chroma

# Make as retriever
retriever = vector_space.as_retriever()

######################################################################
# Testing API
######################################################################
@app.route("/")
def index():
    """
    Test whether the server is working
    """
    return "Hello, World!"

######################################################################
# API using openAI
######################################################################
@app.route("/openai/chat/<string:send_message>", methods=["GET"])
def openai_get_response(send_message):
    """
    Input the sentence
    Call OpenAI API
    Output the return from OpenAI
    """
    user_message = {"role": "user", "content": send_message}
    messages.append(user_message)

    response_message = openai.ChatCompletion.create(
        model = "gpt-4-0613",
        messages = messages,
        temperature = 0.2,
    )

    print(response_message['choices'][0]['message']['content'])
    system_message = {"role": "assistant", "content": response_message['choices'][0]['message']['content']}
    messages.append(system_message)

    return jsonify(response_message['choices'][0]['message']['content']), status.HTTP_200_OK

######################################################################
# API using langchain
######################################################################
@app.route("/langchain/chat/<string:send_message>", methods=["GET"])
def langchain_get_response(send_message):
    """
    Input the sentence
    Call langchain API
    Output the return from langchain
    """
    llm = OpenAI(model_name = "gpt-3.5-turbo", temperature = 0.2)
    conversation = ConversationChain(
        llm = llm,
        verbose = True,
        memory = memory_test
    )
    response_message = conversation.predict(input = send_message)
    print(response_message)
    return jsonify(response_message), status.HTTP_200_OK

######################################################################
# API using langchain with vector space
######################################################################
@app.route("/langchain/chat/vector/<string:send_message>", methods=["GET"])
def langchain_vector_get_response(send_message):
    """
    Input the sentence
    Call langchain API and Chroma
    Output the return from langchain
    """
    # The system template is used to provide context to the LangChain API.
    system_template = """
    Use the following context to answer the user's question.
    If you don't know the answer, say you don't, don't try to make it up.
    -----------
    {question}
    -----------
    {chat_history}
    """

    # The initial messages are used to create a prompt for the LangChain API.
    initial_messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template('{question}')
    ]

    # The prompt is a ChatPromptTemplate object that is created from the initial messages.
    prompt = ChatPromptTemplate.from_messages(initial_messages)

    # The LLM is an OpenAI object that is used to generate the response from the LangChain API.
    llm = OpenAI(model_name = "gpt-3.5-turbo-16k", temperature = 0.2)

    # The chain object is a ConversationalRetrievalChain object that is created from the LLM and retriever objects.
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever = retriever,
        condense_question_prompt = prompt,
        verbose = True,
    )

    # The response message is a dictionary that contains the response from the LangChain API.
    response_message = chain({'question': send_message, 'chat_history': chat_history})

    # The chat history is a list of tuples that stores the chat history.
    chat_history.append((send_message, response_message['answer']))

    # The function returns the response message as JSON.
    return jsonify(response_message['answer']), status.HTTP_200_OK

######################################################################
# API using langchain with agent
######################################################################
@app.route("/langchain/chat/agent/<string:send_message>", methods=["GET"])
def langchain_agent_get_response(send_message):
    filter_template = """
        Determine whether this question is related to NBA, if is related, say true, if not, say false.
        {question}
    """
    filter_messages = [
        SystemMessagePromptTemplate.from_template(filter_template),
        HumanMessagePromptTemplate.from_template('{question}')
    ]
    filter_prompt = ChatPromptTemplate.from_messages(filter_messages)
    filter_llm = OpenAI(model_name = "gpt-3.5-turbo", temperature = 0.2)
    filter_chain = LLMChain(
        llm = filter_llm,
        prompt = filter_prompt
    )
    filter_response = filter_chain({'question': send_message})
    if filter_response['text'] == "False" or filter_response['text'] == "false":
        filter_error = """
            Sorry, this is not related to NBA.
        """
        return jsonify(filter_error), status.HTTP_200_OK

    # agent_prompt = "You are very powerful assistant, and while answerng question, you will first search in the database"
    agent_search = SerpAPIWrapper()
    agent_llm = OpenAI(model_name = "gpt-3.5-turbo", temperature = 0.2)
    agent_db = RetrievalQA.from_chain_type(llm = agent_llm, chain_type = "stuff", retriever = retriever)
    agent_tools = [
        Tool(
            name = "search in the database",
            func = agent_db.run,
            description = "useful for when you want to find answer in the database."
        ),
        Tool(
            name = "search in Google",
            func = agent_search.run,
            description = "useful for when you cannot find answer in the database and need to search."
        ),
    ]
    agent = initialize_agent(
        tools = agent_tools,
        llm = agent_llm,
        agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose = True,
        agent_kwargs = agent_kwargs,
        memory = memory,
    )
    agent_response = agent.run(send_message)
    print(agent_response)
    # agent_openai = OpenAIFunctionsAgent(llm = agent_llm, tools = agent_tools, prompt = agent_prompt)
    # agent_executor = AgentExecutor(agent = agent_openai, tools = agent_tools, verbose = True)

    return jsonify(agent_response), status.HTTP_200_OK
