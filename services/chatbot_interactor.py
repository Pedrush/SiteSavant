# invoke llm, feed the relevant context, prompt engineer it, handle the session properly

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from utils.utils import read_markdown_file, read_yaml_file
from config.logging_config import setup_global_logger
import logging

def generate_chat_response(query_results: str, user_query: str, url: str, model_name: str) -> str:
    """
    Generates a chat response based on provided information and a user query using a specified model.

    Constructs a prompt template indicating that the assistant is based on information retrieved from
    a specific URL, processes the prompt through a chat model, and returns the generated response.

    Parameters:
    - query_results (str): The information retrieved from the specified URL, formatted as a string.
    - user_query (str): The user's query that needs to be answered based on the provided information.
    - url (str): The URL from which the information was sourced.
    - model_name (str): The name of the model to use for generating the chat response.

    Returns:
    - str: The generated response to the user's query.
    TODO: url can be str or list of str
    TODO: consider removing the url argument
    """

    # Construct the chat prompt
    prompt = ChatPromptTemplate.from_template(
        "You're SiteSavant, a digital assistant trained to give answers based on specific "
        "information retrieved from websites. If the provided information is insufficient to "
        "answer the query, tell the user that you don't know the answer and suggest what to do next. "
        "Provided Information:\n"
        "{query_results}\n"
        "Based on this information, answer the following query:\n{user_query}"
    )
    # Initialize the model and output parser
    model = ChatOpenAI(model=model_name)
    output_parser = StrOutputParser()

    # Chain the processing steps
    chain = prompt | model | output_parser

    # Invoke the chain with the provided arguments
    result = chain.invoke({"url": url, "query_results": query_results, "user_query": user_query})

    return result

def main():

    logging.basicConfig(level=logging.INFO)
    setup_global_logger() 
    load_dotenv()

    config = read_yaml_file('config/parameters.yml')
    chat_wtih_chatbot_config = config['chatbot_interactor']

    url = chat_wtih_chatbot_config.get('url')
    model_name = chat_wtih_chatbot_config.get('model_name')
    query_results_file_path = chat_wtih_chatbot_config.get('query_results_file_path')
    user_query = chat_wtih_chatbot_config.get('user_query')

    query_results = read_markdown_file(query_results_file_path)
    prompt = ChatPromptTemplate.from_template(
        "You're SiteSavant, a digital assistant trained to give answers based on specific "
        "information retrieved from websites. Below is information sourced from {url}."
        "Don't mention the {url} in your answer."
        "If the provided information is insufficient to answer the query, tell the user that you don't "
        "know the answer and suggest what to do next.\n"
        "Provided Information:\n"
        "{query_results}\n"
        "Based on this information, answer the following query: {user_query}"
    )

    model = ChatOpenAI(model=model_name)
    output_parser = StrOutputParser()

    chain = prompt | model | output_parser

    result = chain.invoke({"url": url, "query_results": query_results, "user_query": user_query})
    print(result)

if __name__ == "__main__":
    main()
    # TODO: run this in a debugger, forget cli for now