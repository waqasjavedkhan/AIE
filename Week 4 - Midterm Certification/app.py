# importing required modules
import os
import getpass
import wandb
from openai import AsyncOpenAI  # importing openai for API usage
import chainlit as cl  # importing chainlit for our app
from chainlit.prompt import Prompt, PromptMessage  # importing prompt tools
from chainlit.playground.providers import ChatOpenAI  # importing ChatOpenAI tools
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

load_dotenv()

# Initializing the OpenAI Embedding Model `text-embedding-3-small`.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Loading and spliting the document.
loader = PyMuPDFLoader("https://d18rn0p25nwr6d.cloudfront.net/CIK-0001045810/1cbe8fe7-e08a-46e3-8dcc-b429fc06c1a4.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# Setting up FAISS-powered vector store, and creating a retriever.
vector_store = FAISS.from_documents(documents, embeddings)
retriever = vector_store.as_retriever()

# Creating prompt using `ChatPromptTemplate`.
template = """Answer the question based only on the following context. If you cannot answer the question with the context, please respond with 'I don't know':

Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Initializing OpenAI Chat Model 'gpt-3.5-turbo' with a temperature of 0.
openai_chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Creating LCEL chain.
retrieval_augmented_qa_chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": prompt | openai_chat_model, "context": itemgetter("context")}
)

# Creating a Retrieval Augmented QA Chain and initializing Weights & Biases project.
retrieval_augmented_qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0.3),
    chain_type="stuff",
    retriever=retriever,
)
@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():
    settings = {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    cl.user_session.set("settings", settings)

@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: cl.Message):
    settings = cl.user_session.get("settings")

    client = AsyncOpenAI()

    print(message.content)

    # Initializing the logging.
    wandb.init(project="Midterm-Certification")
    
    query = message.content
    
    response = retrieval_augmented_qa_chain.run(query)

    # Closing the logging.
    wandb.finish()

    print(response)
    
    msg = cl.Message(content="")

    msg.content = response

    # Send and close the message stream
    await msg.send()