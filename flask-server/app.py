# pylint: disable=E0401
# pylint: disable=W0611

import os

from flask import Flask, redirect, render_template, request, url_for, jsonify
from flask_cors import CORS

import status  # HTTP Status Codes
from dotenv import load_dotenv

import openai

from langchain.llms import OpenAI
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate
)

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma

######################################################################
# Initialize
######################################################################
load_dotenv()
app = Flask(__name__)
CORS(app)
os.environ["OPENAI_API_KEY"] = ""
messages = [
    {"role": "assistant", "content": "You are a NBA expert."}
]
chat_history = []

# Initialize memory buffer
memory = ConversationBufferMemory()

# load the document and split it into chunks
loader = PyPDFLoader("DevOps-Homework.pdf")
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# load it into Chroma
vector_space = Chroma.from_documents(docs, embedding_function)
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
        model = "gpt-3.5-turbo",
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
        memory = memory
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
    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

    # The retriever is a ChromaVectorSpace object that is used to retrieve the response from the vector space.
    retriever = Chroma.from_documents(docs, embedding_function)

    # The chain object is a ConversationalRetrievalChain object that is created from the LLM and retriever objects.
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        condense_question_prompt=prompt,
        verbose=True
    )

    # The response message is a dictionary that contains the response from the LangChain API.
    response_message = chain({'question': send_message, 'chat_history': chat_history})

    # The chat history is a list of tuples that stores the chat history.
    chat_history.append((send_message, response_message['answer']))

    # The function returns the response message as JSON.
    return jsonify(response_message['answer']), status.HTTP_200_OK
