# pylint: disable=E0401
# pylint: disable=W0611

import os

from flask import Flask, redirect, render_template, request, url_for, jsonify
from flask_cors import CORS

import status  # HTTP Status Codes
from dotenv import load_dotenv

import openai

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate

######################################################################
# Initialize
######################################################################

load_dotenv()
app = Flask(__name__)
CORS(app)
os.environ["OPENAI_API_KEY"] = "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"
messages = [
    {"role": "assistant", "content": "You are a NBA expert."}
]

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
@app.route("/langchain/chat/<string:sendMessage>", methods=["GET"])
def langchain_get_response(send_message):
    """
    Input the sentence
    Call langchain API
    Output the return from langchain
    """
    llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0.2)
    response_message = llm.predict_messages([HumanMessage(content="Translate this sentence from English to French. I love programming.")])

    print(response_message)
    return jsonify(response_message), status.HTTP_200_OK
