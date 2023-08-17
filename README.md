# Chat with your own data!

An app that lets you chat with your documents (PDFs); the app makes use of the OpenAI LLM using LangChain as the framework.

```
.
├── README.md
├── chat.py                             --> Main app
├── config.conf                         --> Configuration file that allows one to change the models etc
├── input_docs                          --> Document directory to store your PDFs
│   └── Python-ML.pdf
├── logs                                --> Log directory; app logs as well as user logs
├── requirements.txt                    
├── restart_service_on_changes.py       --> Restarts the `chat.py` if there are changes in the `input_docs` folder
└── translator_langchain.py             --> Used for supporting the translation of the queries
```

# How it works?

`chat.py` is the main app that connects to the LLM (OpenAI) based on the configuration that you give in the `config.conf` file.
First, we split the document based on the configurations and then create a database index which will be used later to query. The database that we use is a vector database called, 'ChromaDB'. This gets created when the script is executed for the first time. Also, for 'ChromaDB' to work, one needs to have a `Sqlite3 DB`, so make sure to install one.

Create a `.env` file and add your OpenAI API key as:

`OPENAI_API_KEY=sk-XYZ`

The app makes use of 'Gradio' framework; so make sure to run the following command to install the required libraries:

`pip install -r requirements.txt`

Once the required libraries are installed, you can start the application by:

`python3 chat.py`, will output the link which you can paste into your browser to access the app.

The `chat.py` in turn makes use of `translator_langchain.py` to provide a translation feature.

In case, the user wants the indexing to be updated whenever there are changes to the `input_docs` like when a new PDF file is added/removed; the user can make use of `restart_service_on_changes.py` -- this will start the `chat.py` internally, so you need to make sure to close any previous running instance of `chat.py`.

Note: If you are using `restart_service_on_changes.py` --> then expect brief service disruption whenever there are changes to the `input_docs` folder as the script will stop the `chat.py` momentarily to index the new files.
