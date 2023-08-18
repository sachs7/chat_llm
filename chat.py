import gradio as gr
import os
import shutil
import re
import json

from datetime import datetime
from dotenv import load_dotenv
from loguru import logger

from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFDirectoryLoader

from translator_langchain import translator

# Load Environment variable(s)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

logger.remove(0)  # To not show the logs in the console
logger.add(f"logs/logs_{timestamp}.log", rotation="23:59", compression="zip")

"""
# Uncomment the following if want to run in the server
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
"""


def read_config(conf_file):
    try:
        with open(conf_file, "r") as f:
            conf = json.load(f)

        return conf
    except Exception as e:
        logger.debug(f"Error while opening the conf_file: {e}")


conf = read_config("config.conf")

logger.info(f"LLM Model Name: {conf['model_name']}")
logger.info(f"Chain Type: {conf['chain_type']}")
logger.info(f"Chunk Size: {conf['chunk_size']}")
logger.info(f"Chunk Overlap: {conf['chunk_overlap']}")
logger.info(f"Number of documents to refer: {conf['number_of_relevant_chunks']}")
logger.info(f"Search Type: {conf['search_type']}")
logger.info(f"DB Persistant Directory: {conf['persist_directory']}")
logger.info(f"Source Documents Directory: {conf['documents_directory']}")

# Model Name
# llm_name = "gpt-3.5-turbo-0301"
llm_name = conf["model_name"]

# Chain Type to use; other options 'map_reduce', "refine", "map_rerank"
# chain_type = "stuff"
chain_type = conf["chain_type"]

# Chunk Size
chunk_size = conf["chunk_size"]

# Chunk Overlap
chunk_overlap = conf["chunk_overlap"]

# Number of relevant chunks
k = conf["number_of_relevant_chunks"]

# Search Type:
# `Maximum marginal relevance` strives to achieve both relevance to the query *and diversity* among the results.
# search_type = "mmr"
search_type = conf["search_type"]

# Directory to store the indexed documents
# persist_directory = "db/chroma/"
persist_directory = conf["persist_directory"]

# Directory containing all the PDFs that we want
# documents_directory = "./input_docs/"
documents_directory = conf["documents_directory"]

# Build prompt
template = """Use the following pieces of context to answer the question at the end.
Answer only if the question is within the context, if anything outside the context, then don't answer.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer as concise as possible and to the given context. Do NOT query the internet for answers.
{context}
Question: {question}
Helpful Answer:"""
QA_PROMPT = PromptTemplate.from_template(template)


# Function to create the directory if it doesn't exist
def create_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except Exception as e:
        logger.debug(f"Error creating a directory: {e}")


# Function to remove the directory if it exist
def remove_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        # Folder and its content removed
        logger.info("Folder and its content removed")
    except Exception as e:
        logger.debug(f"Error removing the folder {folder_path}: {e}")


# Function to extract metadata from the data structure
def extract_metadata(data):
    metadata = set()
    source_documents = data.get("source_documents", [])
    for doc in source_documents:
        metadata.add(doc.metadata["source"])
    return metadata


# Document loader
# def document_loader():
#     # Load PDF
#     loaders = [
#         PyPDFLoader("input_docs/GLCP.pdf")
#     ]
#     docs = []
#     for loader in loaders:
#         docs.extend(loader.load())
#     return docs
def document_loader():
    try:
        loader = PyPDFDirectoryLoader(documents_directory)
        docs = loader.load()
        return docs
    except Exception as e:
        logger.debug(f"Error loading the documents: {e}")


# def data_ingestion_indexing(directory_path, username):
def data_ingestion_indexing():
    # remove old database files if any
    remove_folder(persist_directory)

    # Create the directory if it doesn't exist
    create_directory(persist_directory)

    # loads data from the specified directory path
    # documents = SimpleDirectoryReader(directory_path).load_data()
    # loader = PyPDFLoader("./input_docs/example.pdf")
    # documents = loader.load()
    documents = document_loader()
    logger.info("-" * 50)
    logger.info(f"Length of documents: {len(documents)}")
    try:
        # split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        split_docs = text_splitter.split_documents(documents)
        logger.info(
            f"Number of documents after passing through text_splitter: {len(split_docs)}"
        )
        # define embedding
        embedding = OpenAIEmbeddings()
        # create vector database from data
        vectordb = Chroma.from_documents(
            documents=split_docs,
            embedding=embedding,
            persist_directory=persist_directory,
        )
        logger.info(f"VectorDB collection count: {vectordb._collection.count()}")
        vectordb.persist()
        return vectordb
    except Exception as e:
        logger.debug(f"Error indexing the data: {e}")


def load_index():
    return data_ingestion_indexing()


# Create the index during the start of the script execution
vectordb = load_index()
chat_history = []


# Retrieve the response
def retriev(question):
    try:
        retriever = vectordb.as_retriever(
            search_type=search_type, search_kwargs={"k": k}
        )
        # create a chatbot chain. Memory is managed externally.
        qa = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name=llm_name, temperature=0),
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
            return_generated_question=True,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        )
        result = qa({"question": question, "chat_history": chat_history})

        chat_history.extend([(question, result["answer"])])
        # print(result["answer"])
        logger.info(result)

        # Extract metadata
        metadata_list = []
        metadata_list.extend(extract_metadata(result))

        # Display the extracted metadata
        for idx, metadata in enumerate(metadata_list):
            logger.info(f"Metadata {idx + 1}: {metadata}")

        return result, metadata_list
    except Exception as e:
        logger.debug(f"Error retrieving the results: {e}")


def chat_app(input_text, username, language):
    if username == "":
        raise gr.Error("Please enter a username!")
    elif not re.match("^[a-zA-Z0-9_\s]{3,30}$", username):
        raise gr.Error("Username length should be more than 3 and include letters, numbers and underscores; Example: test_august7")
    if language.lower() == "english":
        response, metadata_list = retriev(input_text)
        append_to_log(username, input_text, response["answer"], metadata_list)
        return response["answer"], metadata_list, get_chat_history(username)
    inter_result = translator(
        input_text=input_text, input_language=language, output_language="English"
    )
    response, metadata_list = retriev(inter_result)
    final_translated_response = translator(
        input_text=response["answer"], input_language=language, output_language=language
    )
    append_to_log(username, input_text, final_translated_response, metadata_list)
    return final_translated_response, metadata_list, get_chat_history(username)


def append_to_log(username, input_text, response, metadata_list):
    with open(f"logs/{username}_log.txt", "a") as file:
        file.write(f"\nUser:\n{input_text}\n")
        file.write("\n")
        file.write(f"Bot:\n{response}\n")
        file.write(f"\nMetadata:\n{metadata_list}\n")
        file.write("\n")
        file.write(f"\nTimestamp: {timestamp}\n")
        file.write("-" * 35)


def new_rating_append(username, new_rating):
    logger.info(new_rating)
    with open(f"logs/{username}_log.txt", "a") as file:
        file.write(f"\nRating: {new_rating}\n")
        file.write(f"\nTimestamp: {timestamp}\n")
        file.write("-" * 35)


def get_chat_history(username):
    log_file = f"logs/{username}_log.txt"
    if os.path.exists(log_file):
        with open(log_file, "r") as file:
            return file.read()
    else:
        return ""


with gr.Blocks() as chat:
    gr.Markdown(
        """
        <div align="center">
            <h1>Chat - A Custom-trained AI Chatbot</h1>
                    Chat with your PDFs!

        </div>
        """
    )
    with gr.Row():
        with gr.Column():
            username = gr.Textbox(label="Username", placeholder="john_august10")
            prompt = gr.Textbox(label="Prompt", placeholder="Enter your text", lines=5)
            language = gr.Dropdown(
                choices=[
                    "Chinese (simplified)",
                    "Chinese (traditional)",
                    "English",
                    "French",
                    "German",
                    "Italian",
                    "Japanese",
                    "Korean",
                    "Polish",
                    "Portuguese",
                    "Spanish",
                ],
                label="Languages",
                value="English",
                info="Select the languages to query in! (Default: English)",
                interactive=True,
            )
            prompt_submit_btn = gr.Button("Prompt Submit")
            with gr.Row():
                with gr.Column():
                    prompt_response = gr.Textbox(label="Prompt Response")
                    new_rating = gr.Radio(
                        choices=["Not Relevant", "Somewhat Relevant", "Perfect"],
                        label="Rating",
                    )
                    rate_btn = gr.Button("Rate Response")
                    with gr.Accordion("Show Metadata...", open=False):
                        metadata_list_view = gr.Textbox(label="Metadata")
        with gr.Row():
            with gr.Column():
                with gr.Accordion("How to use Chat?", open=False):
                    gr.Markdown(
                        """
                        Examples:
                        - "What is this document about?"
                        - "Give me a simple 'hello-world' program in Machine Learning"
                        """
                    )
                with gr.Accordion("Chat History", open=False):
                    with gr.Column():
                        # with gr.Row():
                        user_chat_history = gr.components.Textbox(show_label=False)

    prompt_submit_btn.click(
        fn=chat_app,
        inputs=[prompt, username, language],
        outputs=[prompt_response, metadata_list_view, user_chat_history],
        api_name="chat_app",
    )
    rate_btn.click(
        fn=new_rating_append,
        inputs=[username, new_rating],
        outputs=None,
        api_name="new_rating_append",
    )

chat.launch(server_name="0.0.0.0", server_port=8081)
