# 配置文件 - 管理API密钥和环境变量
import os

# Pinecone配置
PINECONE_API_KEY = "pcsk_5Rbhpw_EiKkYBYc6UUFaFepRSdr7CTWGdFnNx1VdLnpMs7JaUQYyjnATKiJKYAE1PTZURb"  # 请替换为你的实际API密钥
PINECONE_ENV = "chatbot"  # 请替换为你的实际环境

# 服务器配置
HOST = "127.0.0.1"
PORT = 25001

# 模型配置
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "deepseek-r1:1.5b"

# 向量库配置
INDEX_NAME = "recipe-index"

# 设置环境变量
def setup_environment():
    """设置环境变量"""
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    os.environ["PINECONE_ENV"] = PINECONE_ENV

# 从环境变量获取配置
def get_pinecone_config():
    """获取Pinecone配置"""
    return {
        "api_key": os.getenv("PINECONE_API_KEY", PINECONE_API_KEY),
        "environment": os.getenv("PINECONE_ENV", PINECONE_ENV)
    }

def get_server_config():
    """获取服务器配置"""
    return {
        "host": HOST,
        "port": PORT
    }

def get_model_config():
    """获取模型配置"""
    return {
        "ollama_base_url": OLLAMA_BASE_URL,
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL
    }