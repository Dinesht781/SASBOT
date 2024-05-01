import streamlit as st
import os
from langchain_groq import ChatGroq
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain import hub
from  langchain.schema import Document
import json
from typing import Iterable
from dotenv import load_dotenv
load_dotenv()


## load the Groq API key
groq_api_key="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"

def load_docs_from_jsonl(file_path)->Iterable[Document]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array
if "vector" not in st.session_state:
    st.session_state.embeddings=OpenAIEmbeddings()
    st.session_state.final_documents=load_docs_from_jsonl("fresh_chunk.jsonl")
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)



st.title("SASBOT using mixtral-8x7b LLM")
llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="mixtral-8x7b-32768")


prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)


document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)


# st.chat_message(name="ai")
import numpy as np
message = st.chat_message("assistant")
message.write("Hello SASTRAite")
input_text = st.text_input("ask something")
if input_text:
    st.write(retrieval_chain.invoke({"input":input_text})['answer'])
# st.chat_input()

