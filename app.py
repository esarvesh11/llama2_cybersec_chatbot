import streamlit as st
from streamlit_chat import message
from langchain.llms import CTransformers
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from glob import glob
import pickle

# Create the title and
st.set_page_config(page_title="Cybersecurity Chatbot")

# create the header and the line underneath it
header_html = "<h1 style='text-align: center; margin-bottom: 1px;'>🤖 The Cybersecurity Chatbot 🤖</h1>"
line_html = "<hr style='border: 2px solid green; margin-top: 1px; margin-bottom: 0px;'>"
st.markdown(header_html, unsafe_allow_html=True)
st.markdown(line_html, unsafe_allow_html=True)

@st.cache_resource()

def load_llm():

    # load the llm with ctransformers
    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q2_K.bin', # model available here: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
                    model_type='llama',
                    config={'max_new_tokens': 256, 'temperature': 0})
    return llm

@st.cache_resource()
def load_vector_store():

    # load the vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(r"C:\Users\esarv\Desktop\offline_chatbo\faiss", embeddings)
    return db

def load_prompt_template():

    # prepare the template we will use when prompting the AI
    template = """Use the provided context to answer the user's question.
    If you don't know the answer, respond with "I do not know".

    Context: {context}
    Question: {question}
    Answer:
    """

    prompt = PromptTemplate(
    template=template,
    input_variables=['context', 'question'])

    return prompt

def create_qa_chain():

    # load the llm, vector store, and the prompt
    llm = load_llm()
    db = load_vector_store()
    prompt = load_prompt_template()

    # create the qa_chain
    retriever = db.as_retriever(search_kwargs={'k': 2})
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type='stuff',
                                        retriever=retriever,
                                        return_source_documents=True,
                                        chain_type_kwargs={'prompt': prompt})
    
    return qa_chain

def generate_response(query, qa_chain):

    # use the qa_chain to answer the given query
    return qa_chain({'query':query})['result']

def get_user_input():

    # get the user query
    input_text = st.text_input('Ask me anything about the Cybersecurity!', "", key='input')
    return input_text


# create the qa_chain
qa_chain = create_qa_chain()

#create llm
# llm = CTransformers(model=r"C:\Users\esarv\Downloads\llama-2-7b-chat.ggmlv3.q2_K.bin", model_type="llama", streaming=True, 
#                     callbacks=[StreamingStdOutCallbackHandler()],
#                     config={'max_new_tokens':4096,'temperature':0.01, 'context_length':4096})


# create lists to store user queries and generated responses
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []


# get the user query
user_input = get_user_input()


if user_input:

    # generate response to the user input
    response = generate_response(query=user_input, qa_chain=qa_chain)

    # add the input and response to session state
    st.session_state.past.append(user_input)
    st.session_state.generated.append(response)


# show queries and responses in the user interface
if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"])):
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        message(st.session_state["generated"][i], key=str(i))

# st.title("CyberSecurity ChatBot")
# def conversation_chat(query):
#     result = chain({"question": query, "chat_history": st.session_state['history']})
#     st.session_state['history'].append((query, result["answer"]))
#     return result["answer"]

# def initialize_session_state():
#     if 'history' not in st.session_state:
#         st.session_state['history'] = []

#     if 'generated' not in st.session_state:
#         st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

#     if 'past' not in st.session_state:
#         st.session_state['past'] = ["Hey! 👋"]

# def display_chat_history():
#     reply_container = st.container()
#     container = st.container()

#     with container:
#         with st.form(key='my_form', clear_on_submit=True):
#             user_input = st.text_input("Question:", placeholder="Ask me about CyberSec", key='input')
#             submit_button = st.form_submit_button(label='Send')

#         if submit_button and user_input:
#             output = conversation_chat(user_input)

#             st.session_state['past'].append(user_input)
#             st.session_state['generated'].append(output)

#     if st.session_state['generated']:
#         with reply_container:
#             for i in range(len(st.session_state['generated'])):
#                 message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
#                 message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

# # Initialize session state
# initialize_session_state()
# # Display chat history
# display_chat_history()