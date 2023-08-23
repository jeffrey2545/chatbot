# pylint: disable=E0401
# pylint: disable=W0611
from __future__ import annotations
import os
import status  # HTTP Status Codes
import numpy as np
import re
import fitz
import concurrent.futures

# from langchain.retrievers.knn import (KNNRetriever, create_index)
from flask import Flask, redirect, render_template, request, url_for, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain import SerpAPIWrapper
from langchain.llms import OpenAI
from langchain.chains import (
    ConversationalRetrievalChain,
    LLMChain,
    RetrievalQA,
)
from langchain.agents import (
    initialize_agent,
    Tool,
    AgentType,
)
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.embeddings.tensorflow_hub import TensorflowHubEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from sklearn.neighbors import NearestNeighbors
from typing import Any, List, Optional
from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever, Document


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
embedding_function = OpenAIEmbeddings()

######################################################################
# Dealing pdf to chunks
######################################################################
def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text


def pdf_to_text(path, start_page=1, end_page=None):
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in range(start_page - 1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list


def text_to_chunks(texts, word_length=30, start_page=1):
    text_toks = [t.split(' ') for t in texts]
    chunks = []

    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i : i + word_length]
            if (
                (i + word_length) > len(words)
                and (len(chunk) < word_length)
                and (len(text_toks) != (idx + 1))
            ):
                text_toks[idx + 1] = chunk + text_toks[idx + 1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[Page no. {idx+start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks


######################################################################
# KNNRetriever class
######################################################################
def create_index(contexts: List[str], embeddings: Embeddings) -> np.ndarray:
    """
    Create an index of embeddings for a list of contexts.

    Args:
        contexts: List of contexts to embed.
        embeddings: Embeddings model to use.

    Returns:
        Index of embeddings.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        return np.array(list(executor.map(embeddings.embed_query, contexts)))


class KNNRetriever(BaseRetriever):
    """KNN Retriever."""

    embeddings: Embeddings
    index: Any
    texts: List[str]
    k: int = 4
    relevancy_threshold: Optional[float] = None

    class Config:

        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @classmethod
    def from_texts(
        cls, texts: List[str], embeddings: Embeddings, **kwargs: Any
    ) -> KNNRetriever:
        index = create_index(texts, embeddings)
        return cls(embeddings=embeddings, index=index, texts=texts, **kwargs)

    def _get_relevant_documents(
        self, query: str
    ) -> bool:
        query_embeds = np.array(self.embeddings.embed_query(query))
        # calc L2 norm
        index_embeds = self.index / np.sqrt((self.index**2).sum(1, keepdims=True))
        query_embeds = query_embeds / np.sqrt((query_embeds**2).sum())

        similarities = index_embeds.dot(query_embeds)
        sorted_ix = np.argsort(-similarities)

        # for row in sorted_ix[0 : self.k]:
        #     print("normalized_similarities: ", row, " ", similarities[row])

        for row in sorted_ix[0 : self.k]:
            if (similarities[row] >= self.relevancy_threshold):
                return True
        return False


    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        raise NotImplementedError


######################################################################
# chat history memory
######################################################################
agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
}
agent_memory = ConversationBufferMemory(memory_key="memory", return_messages=True)


######################################################################
# load document into Chroma for search_db_google
######################################################################
# Load Unstructured data
loader = PyPDFLoader("AUD.pdf")
documents = loader.load()

# Split data
splitted_docs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)

# Store into vector store Chroma
vector_space = Chroma.from_documents(
    documents=splitted_docs, embedding=embedding_function
)

# Make as retriever
retriever = vector_space.as_retriever()


######################################################################
# Initialize each knn_retriever
######################################################################
# knn_retriever for black list
black_lists_chunks = text_to_chunks(pdf_to_text("AUD.pdf", start_page=1), start_page=1)
black_lists_knn_retriever = KNNRetriever(
    embeddings=embedding_function,
    index=create_index(black_lists_chunks, embedding_function),
    texts=black_lists_chunks,
    k=5,
    relevancy_threshold=0.8,
)


# knn_retriever for white list
white_lists_chunks = text_to_chunks(pdf_to_text("AUD.pdf", start_page=1), start_page=1)
white_lists_knn_retriever = KNNRetriever(
    embeddings=embedding_function,
    index=create_index(white_lists_chunks, embedding_function),
    texts=white_lists_chunks,
    k=5,
    relevancy_threshold=0.8,
)


# knn_retriever for faq
faq_chunks = text_to_chunks(pdf_to_text("AUD.pdf", start_page=1), start_page=1)
faq_knn_retriever = KNNRetriever(
    embeddings=embedding_function,
    index=create_index(faq_chunks, embedding_function),
    texts=faq_chunks,
    k=5,
    relevancy_threshold=0.8,
)


######################################################################
# API using langchain with agent
######################################################################
@app.route("/langchain/chat/agent/<string:send_message>", methods=["GET"])
def langchain_agent_get_response(send_message):
    """
    Input the send_message
    Output the response
    """

    filter_check, filter_message = filter(send_message)
    if not filter_check:
        return jsonify(filter_message), status.HTTP_200_OK

    # 搜尋是否有預設答案
    # faq_check, faq_message = search_template_faq(filter_message)
    # if faq_check:
    #     opt_message = optimizer(faq_message)
    #     return jsonify(opt_message), status.HTTP_200_OK

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
    Use KNN to calculate the distance between black list and white list
    Return the relation.
    """
    black_lists_result = black_lists_knn_retriever._get_relevant_documents(send_message)

    white_lists_result = white_lists_knn_retriever._get_relevant_documents(send_message)

    if not black_lists_result and white_lists_result:
        return True, send_message
    
    if black_lists_result and not white_lists_result:
        filter_error = """
            Sorry, this is not related to [Your Topic].
        """
        return False, filter_error


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
