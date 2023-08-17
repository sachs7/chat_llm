import os

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv
from loguru import logger

# Load Environment variable(s)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


system_template = (
    "You are a helpful assistant that translates {input_language} to {output_language}."
)
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)


human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)


chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)


chat = ChatOpenAI(temperature=0, model="gpt-4-0613")
chain = LLMChain(llm=chat, prompt=chat_prompt)


def translator(input_text, input_language, output_language="English"):
    try:
        input_lang = input_language
        output_lang = output_language
        # print(f"Input Language: {input_lang}")
        logger.info(f"Input Language: {input_lang}")

        english_translation_result = chain.run(
            input_language=input_lang, output_language=output_lang, text=input_text
        )

        return english_translation_result
    except Exception as e:
        logger.error(f"Encountered error while translating: {e}")
