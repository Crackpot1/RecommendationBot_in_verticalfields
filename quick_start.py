#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速启动脚本 - 跳过PDF处理，直接使用现有向量库
"""

import os
import socket
import json
import threading
from dataclasses import dataclass
from config import setup_environment, get_pinecone_config, get_server_config, get_model_config
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory

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

# 全局变量
vector_store = None
llm = None
embeddings_model = None
_retrieval_chain = None

def quick_initialize():
    """快速初始化 - 跳过PDF处理"""
    global vector_store, llm, embeddings_model
    
    print("🚀 快速初始化中...")
    
    # 1. 初始化模型 (快速)
    print("1. 初始化模型...")
    model_config = get_model_config()
    
    # 初始化LLM
    llm = Ollama(
        model=model_config["llm_model"],
        base_url=model_config["ollama_base_url"]
    )
    print("   ✅ LLM初始化完成")
    
    # 初始化Embedding模型
    embeddings_model = OllamaEmbeddings(model=model_config["embedding_model"])
    print("   ✅ Embedding模型初始化完成")
    
    # 2. 连接现有向量库 (快速)
    print("2. 连接向量库...")
    try:
        pinecone_config = get_pinecone_config()
        pc = Pinecone(api_key=pinecone_config["api_key"])
        
        # 直接连接到现有的向量库
        vector_store = PineconeVectorStore.from_existing_index(
            index_name="recipe-index",
            embedding=embeddings_model
        )
        print("   ✅ 向量库连接成功")
        
    except Exception as e:
        print(f"   ❌ 向量库连接失败: {e}")
        print("   💡 请确保已经运行过完整初始化，或者运行: python recipechatbot.py")
        return False
    
    print("✅ 快速初始化完成！")
    return True

# Define prompts
system_prompt = (
    "You are an assistant specialized in providing personalized cooking recipe recommendations. "
    "You cater to various user preferences such as dietary restrictions, cuisine types, available ingredients, and cooking time. "
    "Answer the user's questions based on the context provided from the vector database. "
    "If the context does not have relevant information,inform the user that the question is out of scope or the information is not available. Do NOT answer the question out of the vector database."
    "<think></think>\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

def get_retrieval_chain(vector_store):
    global _retrieval_chain
    if _retrieval_chain is None:
        print("正在创建检索链...")
        retriever = create_history_aware_retriever(
            llm=llm, 
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}), 
            prompt=contextualize_q_prompt
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        _retrieval_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=question_answer_chain
        )
        print("检索链创建完成")
    return _retrieval_chain

# 存储会话历史
store = {}

def get_session_history(session_id: str) -> SimpleChatMessageHistory:
    if session_id not in store:
        store[session_id] = SimpleChatMessageHistory()
    return store[session_id]

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

def send_user_info(client_socket, user_info):
    try:
        json_data = json.dumps(user_info.__dict__)
        json_data += '\n'
        client_socket.sendall(json_data.encode())
        print("已发送回客户端")
    except Exception as e:
        print(f"发送数据时出错: {e}")

def handle_client(client_socket):
    receive_thread = threading.Thread(target=receive_user_info, args=(client_socket,))
    receive_thread.start()

def receive_user_info(client_socket):
    while True:
        try:
            received_data = client_socket.recv(4096).decode()
            if not received_data:
                break
            
            user_data = json.loads(received_data)
            user_info = UserInfo(**user_data)
            print("收到Unity信息:", user_info)

            answer = get_answer(user_info.Query)
            user_info.Answer = answer["answer"] if isinstance(answer, dict) else str(answer)
            
            print("生成的答案:", user_info.Answer)
            send_user_info(client_socket, user_info)
            
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
        except Exception as e:
            print(f"接收数据时出错: {e}")
            break
    
    client_socket.close()

def start_server():
    # 快速初始化
    if not quick_initialize():
        print("快速初始化失败，无法启动服务器")
        return
    
    # 创建TCP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((host, port))
        server_socket.listen(5)
        print(f"🚀 服务器启动成功！正在监听 {host}:{port}")
        print("💡 这是快速启动模式，跳过PDF处理步骤")
        
        while True:
            client_socket, addr = server_socket.accept()
            print(f"✅ 客户端连接: {addr}")

            client_thread = threading.Thread(target=handle_client, args=(client_socket,))
            client_thread.daemon = True
            client_thread.start()
            
    except KeyboardInterrupt:
        print("\n🛑 服务器正在关闭...")
    except Exception as e:
        print(f"❌ 服务器错误: {e}")
    finally:
        server_socket.close()

if __name__ == "__main__":
    print("⚡ 快速启动模式")
    print("=" * 50)
    print("此模式将跳过PDF处理，直接使用现有向量库")
    print("如果向量库不存在，请先运行: python recipechatbot.py")
    print("=" * 50)
    
    start_server() 