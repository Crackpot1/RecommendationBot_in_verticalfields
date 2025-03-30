# Import necessary libraries and modules
import os
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import cachetools
from langchain.docstore.document import Document
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from voice_component import start_voice_input, text_to_speech,stop_voice_input

# Concrete implementation of BaseChatMessageHistory
class SimpleChatMessageHistory(BaseChatMessageHistory):
    def __init__(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)

    def get_messages(self):
        return self.messages

    def clear(self):
        self.messages.clear()

# Set up the environment
secrets = st.secrets  # Accessing secrets (API keys) stored securely

pinecone_api_key = secrets["pinecone"]["api_key"]
os.environ["PINECONE_API_KEY"] = pinecone_api_key

pinecone_env = secrets["pinecone"]["environment"]
os.environ["PINECONE_ENV"] = pinecone_env

# Initialize Pinecone with API key
pc = Pinecone(api_key=pinecone_api_key)
# Create or connect to a Pinecone index
index_name = "recipe-index"
index = pc.Index(index_name)

embeddings_model = OllamaEmbeddings(model="bge-m3")

# Extract text from PDF files using PyMuPDF with a fallback to pdfminer
def extract_text_from_pdf(pdf_path):
    try:
        document = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error with PyMuPDF: {e}")
        text = extract_text(pdf_path)
        return text

# Asynchronously load and chunk PDF documents
async def load_and_chunk_pdfs(pdf_paths, chunk_size):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        tasks = [
            loop.run_in_executor(executor, functools.partial(extract_text_from_pdf, pdf_path))
            for pdf_path in pdf_paths
        ]
        texts = await asyncio.gather(*tasks)
    
    documents = [Document(page_content=text) for text in texts]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    split_docs = text_splitter.split_documents(documents)
    return split_docs

# Load PDF files from the "Recipe material" directory
pdf_directory = "Recipe material"
pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

# Convert list of documents to a hashable type for caching
def convert_to_hashable(documents):
    return tuple((doc.page_content for doc in documents))

# Caching results to improve performance
@cachetools.cached(cache=cachetools.LRUCache(maxsize=128), key=lambda docs: convert_to_hashable(docs))
def get_cached_vector_store(split_docs):
    return PineconeVectorStore.from_documents(split_docs, embeddings_model, index_name=index_name)

# Define system and QA prompts
system_prompt = (
    "You are an assistant specialized in providing personalized cooking recipe recommendations. "
    "You cater to various user preferences such as dietary restrictions, cuisine types, available ingredients, and cooking time. "
    "Answer the user's questions based on the context provided from the vector database. "
    "If the context does not have relevant information,inform the user that the question is out of scope or the information is not available. Do NOT answer the question out of the vector database."
    "<think></think>\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

model_name = "deepseek-r1:1.5b"

llm = Ollama(
    model="deepseek-r1:1.5b",  # 模型名需与Ollama本地列表一致
    base_url="http://localhost:11434"  # 默认API地址
)


# Set up the retrieval chain
def get_retrieval_chain(vector_store):
    retriever = create_history_aware_retriever(
        llm=llm, retriever=vector_store.as_retriever(search_kwargs={"k": 5}), prompt=contextualize_q_prompt
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain  # Correct parameter
    )

    st.session_state.chat_active = True

    return rag_chain

store = {}

def get_session_history(session_id: str) -> SimpleChatMessageHistory:
    if session_id not in store:
        store[session_id] = SimpleChatMessageHistory()  # Use SimpleChatMessageHistory
    return store[session_id]

def get_answer(query):
    retrieval_chain = get_retrieval_chain(st.session_state.vector_store)
    st.session_state.history.append({"role": "user", "content": query})
    session_id = "session_id"  # Use a specific session ID for tracking
    history = get_session_history(session_id)

    answer = retrieval_chain.invoke({"input": query, "chat_history": history.get_messages()})

    # 获取原始检索结果及分数
    raw_results = st.session_state.vector_store.similarity_search_with_score(query, k=5)
    docs, scores = zip(*raw_results)  # 解包文档和分数

    # 手动过滤低分结果
    if max(scores) < 0.5375:
        return {"answer": "Your question is beyond the scope of my current knowledge. I can provide you with some recipe advice..."}

    st.session_state.history.append({"role": "assistant", "content": answer["answer"]})
    return answer


# Streamlit app setup
st.title("🍕Recipe Recommendation Bot")

if "vector_store" not in st.session_state:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    split_docs = loop.run_until_complete(load_and_chunk_pdfs(pdf_files, chunk_size=1000))  # Adjust chunk size as needed
    st.session_state.vector_store = get_cached_vector_store(split_docs)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

# 初始化会话状态
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = None

# Create two columns for input layout
input_col, voice_col = st.columns([4, 1])

with input_col:
    text_input = st.chat_input("Ask your question or speak using the mic...")

with voice_col:
    # 语音按钮逻辑
    if st.button("🎤", key="voice_input"):
        if not st.session_state.is_recording:
            # 开始录音
            st.session_state.is_recording = True
            st.session_state.recorder = start_voice_input(language="en-US")
        else:
            # 停止录音并获取查询
            query = stop_voice_input(st.session_state.recorder, language="en-US")
            st.session_state.is_recording = False
            del st.session_state.recorder  # 清理录音对象

    # 显示录音中的spinner
    if st.session_state.is_recording:
        st.warning("Recording... Click mic again to stop")

    # 语音输出开关
    voice_enabled = st.toggle(
        "Voice Output",
        value=st.session_state.get("voice_output_enabled", True),
        key="voice_output_toggle"
    )
    st.session_state.voice_output_enabled = voice_enabled

# Process text input
if text_input:
    query = text_input

# Process query
if query:
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Get response
    answer = get_answer(query)
    result = answer["answer"]

    # Display assistant response with voice
    with st.chat_message("assistant"):
        st.markdown(result)

        # 根据开关状态决定是否播放语音
        if st.session_state.voice_output_enabled:
            text_to_speech(text=result, language="en-US")

        st.session_state.messages.append({"role": "assistant", "content": result})

def clear_messages():
    st.session_state.messages = []
    st.session_state.history = []  # Clear the conversation history as well


st.button("Clear", help="Click to clear the chat", on_click=clear_messages)
