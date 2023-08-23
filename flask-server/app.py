# pylint: disable=E0401
# pylint: disable=W0611
# pylint: disable=W0212
# pylint: disable=C0301
from __future__ import annotations
import os
import re
import concurrent.futures
from typing import Any, List, Optional, Tuple
import fitz
import status  # HTTP Status Codes
import numpy as np
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain import SerpAPIWrapper
from langchain.llms import OpenAI
from langchain.chains import (
    ConversationalRetrievalChain,
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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
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
    """
    Preprocess the given text by performing the following operations:

    1. Replace newline characters with spaces.
    2. Replace multiple consecutive whitespace characters with a single space.

    Parameters:
    - text (str): The input text to be preprocessed.

    Returns:
    - str: The preprocessed text.
    """
    text = text.replace("\n", " ")
    text = re.sub("\s+", " ", text)
    return text


def pdf_to_text(path, start_page=1, end_page=None):
    """
    Extract text from a PDF document and return it as a list of strings, where each string represents the text of a page.

    Parameters:
    - path (str): The path to the PDF document.
    - start_page (int, optional): The starting page number from which to extract text. Defaults to 1.
    - end_page (int, optional): The ending page number up to which to extract text. If not specified, extracts up to the last page.

    Returns:
    - list of str: A list containing the extracted text from each page, preprocessed using the preprocess function.

    Note:
    This function requires the `PyMuPDF` library (imported as `fitz`).
    """
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


def text_to_chunks_by_word_length(texts, word_length=30, start_page=1):
    """
    Split a list of texts into chunks of a specified word length. Each chunk is prefixed with its corresponding page number.

    Parameters:
    - texts (list of str): A list of texts, where each text typically represents the content of a page.
    - word_length (int, optional): The desired number of words in each chunk. Defaults to 30.
    - start_page (int, optional): The starting page number for the first text in the list. Defaults to 1.

    Returns:
    - list of str: A list of text chunks, where each chunk is prefixed with its page number.

    Note:
    If the last chunk of a text is shorter than the desired word length and it's not the last text in the list,
    the remaining words are prepended to the next text in the list.
    """
    text_tokens = [t.split(" ") for t in texts]
    chunks = []

    for idx, words in enumerate(text_tokens):
        for i in range(0, len(words), word_length):
            chunk = words[i : i + word_length]
            if (
                (i + word_length) > len(words)
                and (len(chunk) < word_length)
                and (len(text_tokens) != (idx + 1))
            ):
                text_tokens[idx + 1] = chunk + text_tokens[idx + 1]
                continue
            chunk = " ".join(chunk).strip()
            chunk = f"[Page no. {idx+start_page}]" + " " + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks


def text_to_chunks_by_semicolon(texts, start_page=1):
    """
    Split a list of texts into chunks based on semicolons. Each chunk is prefixed with its corresponding page number.

    Parameters:
    - texts (list of str): A list of texts, where each text typically represents the content of a page.
    - start_page (int, optional): The starting page number for the first text in the list. Defaults to 1.

    Returns:
    - list of str: A list of text chunks, where each chunk is prefixed with its page number.

    Note:
    The function splits the text wherever it finds a semicolon.
    """
    chunks = []

    for idx, text in enumerate(texts):
        for chunk in text.split(";"):
            chunk = chunk.strip()
            if chunk:  # Check if the chunk is not empty
                chunk = f"[Page no. {idx+start_page}]" + " " + '"' + chunk + '"'
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
    """
    A K-Nearest Neighbors (KNN) based retriever for finding relevant documents based on embeddings.

    Attributes:
    - embeddings (Embeddings): The embeddings used for representing the texts.
    - index (Any): The index structure holding the embeddings of the texts.
    - texts (List[str]): The list of texts/documents.
    - k (int): The number of nearest neighbors to consider. Defaults to 4.
    - relevancy_threshold (Optional[float]): A threshold for similarity score to consider a document as relevant.

    Note:
    This class inherits from `BaseRetriever` and should implement all its abstract methods.
    """
    embeddings: Embeddings
    index: Any
    texts: List[str]
    k: int = 4
    relevancy_threshold: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_texts(
        cls, texts: List[str], embeddings: Embeddings, **kwargs: Any
    ) -> KNNRetriever:
        """
        Class method to create an instance of KNNRetriever from a list of texts and embeddings.

        Parameters:
        - texts (List[str]): The list of texts/documents.
        - embeddings (Embeddings): The embeddings used for representing the texts.
        - **kwargs: Additional keyword arguments.

        Returns:
        - KNNRetriever: An instance of the KNNRetriever class.
        """
        index = create_index(texts, embeddings)
        return cls(embeddings=embeddings, index=index, texts=texts, **kwargs)

    def _get_relevant_documents(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Determine if there are any relevant documents for the given query based on similarity scores and return the most relevant one.

        Parameters:
        - query (str): The query string.

        Returns:
        - tuple:
            - bool: True if there's at least one relevant document, otherwise False.
            - str or None: The most relevant document if found, otherwise None.
        """
        query_embeds = np.array(self.embeddings.embed_query(query))
        # calc L2 norm
        index_embeds = self.index / np.sqrt((self.index**2).sum(1, keepdims=True))
        query_embeds = query_embeds / np.sqrt((query_embeds**2).sum())

        similarities = index_embeds.dot(query_embeds)
        sorted_ix = np.argsort(-similarities)

        # Check the most relevant chunk based on the highest similarity score
        if similarities[sorted_ix[0]] >= self.relevancy_threshold:
            return True, self.texts[sorted_ix[0]]
        return False, None

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Asynchronous method to get relevant documents for the given query. This method is not implemented.

        Parameters:
        - query (str): The query string.
        - run_manager (AsyncCallbackManagerForRetrieverRun): The callback manager for the retriever run.

        Raises:
        - NotImplementedError: This method is not implemented.
        """
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
loader = PyPDFLoader("docs/AUD_questions.pdf")
documents = loader.load()

# Split data
splitted_docs = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
).split_documents(documents)

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
black_lists_chunks = text_to_chunks_by_word_length(
    pdf_to_text("docs/blacklist.pdf", start_page=1), start_page=1
)
black_lists_knn_retriever = KNNRetriever(
    embeddings=embedding_function,
    index=create_index(black_lists_chunks, embedding_function),
    texts=black_lists_chunks,
    k=5,
    relevancy_threshold=0.8,
)


# knn_retriever for white list
white_lists_chunks = text_to_chunks_by_word_length(
    pdf_to_text("docs/whitelist.pdf", start_page=1), start_page=1
)
white_lists_knn_retriever = KNNRetriever(
    embeddings=embedding_function,
    index=create_index(white_lists_chunks, embedding_function),
    texts=white_lists_chunks,
    k=5,
    relevancy_threshold=0.8,
)


# knn_retriever for faq
faq_chunks = text_to_chunks_by_semicolon(
    pdf_to_text("docs/faq.pdf", start_page=1), start_page=1
)
faq_knn_retriever = KNNRetriever(
    embeddings=embedding_function,
    index=create_index(faq_chunks, embedding_function),
    texts=faq_chunks,
    k=5,
    relevancy_threshold=0.9,
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

    filter_check, filter_message = black_white_list_filter(send_message)
    print("filter_check: ", filter_check)
    print("filter_message: ", filter_message)
    if not filter_check:
        return jsonify(filter_message), status.HTTP_200_OK

    faq_check, faq_message = search_template_faq(filter_message)
    print("faq_check: ", faq_check)
    print("faq_message: ", faq_message)
    if faq_check:
        opt_message = optimizer(faq_message)
        return jsonify(opt_message), status.HTTP_200_OK

    agent_check, agent_message = search_db_google(send_message)
    print("agent_check: ", agent_check)
    print("agent_message: ", agent_message)
    if not agent_check:
        return jsonify(agent_message), status.HTTP_200_OK

    # opt_response = optimizer(last_filter_message)

    # store_response(send_message, opt_response)

    return jsonify(agent_message), status.HTTP_200_OK


def black_white_list_filter(send_message: str) -> Tuple[bool, str]:
    """
    Filters a given message based on its relevance to black and white lists using KNN retrievers.

    The function checks the message against two KNN retrievers: one for black lists and one for white lists.
    - If the message is not in the black list and is in the white list, it is considered valid.
    - If the message is in the black list and not in the white list, an error message is returned.

    Parameters:
    - send_message (str): The message to be filtered.

    Returns:
    - tuple:
        - bool: True if the message is valid, otherwise False.
        - str: The original message if it's valid, otherwise an error message.

    Note:
    This function assumes the existence of `black_lists_knn_retriever` and `white_lists_knn_retriever` in the global scope.
    """
    (
        black_lists_is_relevant,
        black_lists_relevant_document,
    ) = black_lists_knn_retriever._get_relevant_documents(send_message)
    (
        white_lists_is_relevant,
        white_lists_relevant_document,
    ) = white_lists_knn_retriever._get_relevant_documents(send_message)

    if not black_lists_is_relevant and white_lists_is_relevant:
        return True, send_message

    if black_lists_is_relevant and not white_lists_is_relevant:
        filter_error = """
            Sorry, this is not related to [Your Topic].
        """
        return False, filter_error

    # Default return if neither condition is met
    return False, "Sorry, this is not related to [Your Topic]."


def search_template_faq(send_message: str) -> Tuple[bool, Optional[str]]:
    """
    Searches for a relevant FAQ document based on the given message using a KNN retriever.

    The function queries the `faq_knn_retriever` to find a relevant FAQ document. If a relevant document is found,
    it returns `True` and the document. Otherwise, it returns `False` and `None`.

    Parameters:
    - send_message (str): The message or query to be searched for in the FAQ documents.

    Returns:
    - tuple:
        - bool: True if a relevant FAQ document is found, otherwise False.
        - str or None: The relevant FAQ document if found, otherwise None.

    Note:
    This function assumes the existence of `faq_knn_retriever` in the global scope or as a class attribute.
    """
    faq_is_relevant, faq_relevant_document = faq_knn_retriever._get_relevant_documents(
        send_message
    )

    if not faq_is_relevant:
        return False, None
    return True, faq_relevant_document


def search_db_google(send_message: str) -> Tuple[bool, str]:
    """
    Searches for an answer to the given message first in a database, then on Google, and finally using the assistant's own knowledge.

    The function follows a hierarchical approach:
    1. Search in the database.
    2. If no useful information is found in the database, search on Google.
    3. If no useful information is found on Google, answer using the assistant's knowledge.
    4. If the assistant doesn't know the answer, it will return that it doesn't know.

    Parameters:
    - send_message (str): The message or query to be searched for.

    Returns:
    - tuple:
        - bool: True if the agent found a relevant response, otherwise False.
        - str: The agent's response or answer to the query.

    Note:
    This function assumes the existence of various tools and models like `SerpAPIWrapper`, `ChatOpenAI`, `RetrievalQA`, etc., and their appropriate configurations.
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
    agent_llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)
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
