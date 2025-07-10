# Import necessary libraries and modules
import os
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
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
from voice_component import start_voice_input, text_to_speech, stop_voice_input
import socket
import json
import threading
from dataclasses import dataclass
from config import setup_environment, get_pinecone_config, get_server_config, get_model_config

# 设置环境变量
setup_environment()

@dataclass
class UserInfo:
    Name: str
    Query: str
    Answer: str = ""

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

# 获取配置
server_config = get_server_config()
host, port = server_config["host"], server_config["port"]


# 全局变量存储向量库和模型
vector_store = None
llm = None
embeddings_model = None

# 初始化Pinecone和模型
def initialize_models():
    global vector_store, llm, embeddings_model
    
    # 从配置文件获取Pinecone配置
    pinecone_config = get_pinecone_config()
    pinecone_api_key = pinecone_config["api_key"]
    pinecone_env = pinecone_config["environment"]
    
    if not pinecone_api_key or pinecone_api_key == "your_pinecone_api_key_here":
        print("警告: 请在config.py中设置正确的PINECONE_API_KEY")
        return False
    
    # Initialize Pinecone with API key
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Create or connect to a Pinecone index
    index_name = "recipe-index"
    index = pc.Index(index_name)
    
    # 从配置文件获取模型配置
    model_config = get_model_config()
    embeddings_model = OllamaEmbeddings(model=model_config["embedding_model"])
    
    # 初始化LLM
    llm = Ollama(
        model=model_config["llm_model"],
        base_url=model_config["ollama_base_url"]
    )
    
    return True

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
def load_pdf_documents():
    pdf_directory = "Recipe material"
    if not os.path.exists(pdf_directory):
        print(f"错误: 目录 '{pdf_directory}' 不存在")
        return None
    
    pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"警告: 在目录 '{pdf_directory}' 中未找到PDF文件")
        return None
    
    return pdf_files

# Convert list of documents to a hashable type for caching
def convert_to_hashable(documents):
    return tuple((doc.page_content for doc in documents))

# Caching results to improve performance
@cachetools.cached(cache=cachetools.LRUCache(maxsize=128), key=lambda docs: convert_to_hashable(docs))
def get_cached_vector_store(split_docs):
    return PineconeVectorStore.from_documents(split_docs, embeddings_model, index_name="recipe-index")

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

# Set up the retrieval chain
def get_retrieval_chain(vector_store):
    retriever = create_history_aware_retriever(
        llm=llm, retriever=vector_store.as_retriever(search_kwargs={"k": 5}), prompt=contextualize_q_prompt
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain
    )

    return rag_chain

# 存储会话历史
store = {}

def get_session_history(session_id: str) -> SimpleChatMessageHistory:
    if session_id not in store:
        store[session_id] = SimpleChatMessageHistory()
    return store[session_id]

# 核心问答函数
def get_answer(query):
    if vector_store is None:
        return {"answer": "错误: 向量库未初始化"}
    
    try:
        retrieval_chain = get_retrieval_chain(vector_store)
        session_id = "session_id"
        history = get_session_history(session_id)

        answer = retrieval_chain.invoke({"input": query, "chat_history": history.get_messages()})

        # 获取原始检索结果及分数
        raw_results = vector_store.similarity_search_with_score(query, k=5)
        docs, scores = zip(*raw_results)

        # 手动过滤低分结果
        if max(scores) < 0.5375:
            return {"answer": "Your question is beyond the scope of my current knowledge. I can provide you with some recipe advice..."}

        return answer
    except Exception as e:
        print(f"获取答案时出错: {e}")
        return {"answer": f"处理您的问题时出现错误: {str(e)}"}

# 将UserInfo实例转为JSON格式并发送给客户端
def send_user_info(client_socket, user_info):
    try:
        json_data = json.dumps(user_info.__dict__)
        json_data += '\n'  # 添加换行符作为分隔符
        client_socket.sendall(json_data.encode())
        print("已发送回客户端")
    except Exception as e:
        print(f"发送数据时出错: {e}")

# 处理客户端连接的函数
def handle_client(client_socket):
    receive_thread = threading.Thread(target=receive_user_info, args=(client_socket,))
    receive_thread.start()

# 从客户端接收JSON数据并解码为UserInfo实例
def receive_user_info(client_socket):
    while True:
        try:
            received_data = client_socket.recv(4096).decode()
            if not received_data:
                break
            
            # 解码JSON数据为UserInfo实例
            user_data = json.loads(received_data)
            user_info = UserInfo(**user_data)
            print("收到Unity信息:", user_info)

            # 调用问答逻辑获取结果
            answer = get_answer(user_info.Query)
            user_info.Answer = answer["answer"] if isinstance(answer, dict) else str(answer)
            
            print("生成的答案:", user_info.Answer)
            
            # 自动发送更新后的UserInfo给客户端
            send_user_info(client_socket, user_info)
            
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
        except Exception as e:
            print(f"接收数据时出错: {e}")
            break
    
    client_socket.close()

# 初始化系统
def initialize_system():
    print("正在初始化系统...")
    
    # 初始化模型
    if not initialize_models():
        print("模型初始化失败")
        return False
    
    # 加载PDF文档
    pdf_files = load_pdf_documents()
    if pdf_files is None:
        print("PDF文档加载失败")
        return False
    
    # 异步加载和分块PDF文档
    global vector_store
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        split_docs = loop.run_until_complete(load_and_chunk_pdfs(pdf_files, chunk_size=1000))
        vector_store = get_cached_vector_store(split_docs)
        print("向量库初始化成功")
        return True
    except Exception as e:
        print(f"向量库初始化失败: {e}")
        return False

# 启动服务器
def start_server():
    # 初始化系统
    if not initialize_system():
        print("系统初始化失败，无法启动服务器")
        return
    
    # 创建TCP socket并绑定IP地址和端口号
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((host, port))
        server_socket.listen(5)
        print(f"正在监听 {host}:{port}")
        
        while True:
            # 等待客户端连接
            client_socket, addr = server_socket.accept()
            print(f"成功连接到客户端 {addr}")

            # 启动一个线程来处理客户端连接
            client_thread = threading.Thread(target=handle_client, args=(client_socket,))
            client_thread.daemon = True
            client_thread.start()
            
    except KeyboardInterrupt:
        print("\n服务器正在关闭...")
    except Exception as e:
        print(f"服务器错误: {e}")
    finally:
        server_socket.close()

if __name__ == "__main__":
    start_server()