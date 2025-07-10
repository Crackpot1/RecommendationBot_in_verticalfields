#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能分析脚本 - 诊断初始化慢的原因
"""

import time
import os
import asyncio
from pathlib import Path
from config import setup_environment, get_pinecone_config, get_model_config
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from pinecone import Pinecone
import fitz
from pdfminer.high_level import extract_text

def analyze_pdf_files():
    """分析PDF文件大小和数量"""
    print("📊 PDF文件分析")
    print("=" * 50)
    
    pdf_directory = "Recipe material"
    if not os.path.exists(pdf_directory):
        print(f"❌ 目录 '{pdf_directory}' 不存在")
        return
    
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    total_size = 0
    
    print(f"找到 {len(pdf_files)} 个PDF文件:")
    for pdf_file in pdf_files:
        file_path = os.path.join(pdf_directory, pdf_file)
        size = os.path.getsize(file_path)
        total_size += size
        print(f"  - {pdf_file}: {size / 1024 / 1024:.2f} MB")
    
    print(f"\n总大小: {total_size / 1024 / 1024:.2f} MB")
    print(f"平均大小: {total_size / len(pdf_files) / 1024 / 1024:.2f} MB")
    
    return pdf_files, total_size

def test_pdf_extraction_speed(pdf_files):
    """测试PDF文本提取速度"""
    print("\n🔍 PDF文本提取速度测试")
    print("=" * 50)
    
    total_text_length = 0
    total_time = 0
    
    for i, pdf_file in enumerate(pdf_files[:3]):  # 只测试前3个文件
        file_path = os.path.join("Recipe material", pdf_file)
        print(f"测试文件 {i+1}: {pdf_file}")
        
        start_time = time.time()
        try:
            # 使用PyMuPDF
            document = fitz.open(file_path)
            text = ""
            for page_num in range(len(document)):
                page = document.load_page(page_num)
                text += page.get_text()
            document.close()
            
            extraction_time = time.time() - start_time
            total_time += extraction_time
            total_text_length += len(text)
            
            print(f"  提取时间: {extraction_time:.2f}秒")
            print(f"  文本长度: {len(text)} 字符")
            print(f"  速度: {len(text) / extraction_time / 1000:.2f} K字符/秒")
            
        except Exception as e:
            print(f"  ❌ 提取失败: {e}")
    
    if total_time > 0:
        print(f"\n平均提取速度: {total_text_length / total_time / 1000:.2f} K字符/秒")

def test_model_initialization():
    """测试模型初始化速度"""
    print("\n🤖 模型初始化速度测试")
    print("=" * 50)
    
    setup_environment()
    
    # 测试Pinecone连接
    print("测试Pinecone连接...")
    start_time = time.time()
    try:
        pinecone_config = get_pinecone_config()
        pc = Pinecone(api_key=pinecone_config["api_key"])
        index = pc.Index("recipe-index")
        pinecone_time = time.time() - start_time
        print(f"✅ Pinecone连接成功: {pinecone_time:.2f}秒")
    except Exception as e:
        print(f"❌ Pinecone连接失败: {e}")
        return
    
    # 测试Embedding模型
    print("测试Embedding模型...")
    start_time = time.time()
    try:
        model_config = get_model_config()
        embeddings = OllamaEmbeddings(model=model_config["embedding_model"])
        embedding_time = time.time() - start_time
        print(f"✅ Embedding模型初始化: {embedding_time:.2f}秒")
    except Exception as e:
        print(f"❌ Embedding模型初始化失败: {e}")
        return
    
    # 测试LLM模型
    print("测试LLM模型...")
    start_time = time.time()
    try:
        llm = Ollama(
            model=model_config["llm_model"],
            base_url=model_config["ollama_base_url"]
        )
        llm_time = time.time() - start_time
        print(f"✅ LLM模型初始化: {llm_time:.2f}秒")
    except Exception as e:
        print(f"❌ LLM模型初始化失败: {e}")
        return
    
    total_model_time = pinecone_time + embedding_time + llm_time
    print(f"\n模型初始化总时间: {total_model_time:.2f}秒")

def test_network_speed():
    """测试网络连接速度"""
    print("\n🌐 网络连接速度测试")
    print("=" * 50)
    
    import requests
    
    # 测试Pinecone API
    try:
        pinecone_config = get_pinecone_config()
        start_time = time.time()
        response = requests.get(
            "https://api.pinecone.io/",
            headers={"Api-Key": pinecone_config["api_key"]},
            timeout=10
        )
        pinecone_network_time = time.time() - start_time
        print(f"Pinecone API响应时间: {pinecone_network_time:.2f}秒")
    except Exception as e:
        print(f"Pinecone网络测试失败: {e}")
    
    # 测试Ollama API
    try:
        model_config = get_model_config()
        start_time = time.time()
        response = requests.get(f"{model_config['ollama_base_url']}/api/tags", timeout=10)
        ollama_network_time = time.time() - start_time
        print(f"Ollama API响应时间: {ollama_network_time:.2f}秒")
    except Exception as e:
        print(f"Ollama网络测试失败: {e}")

def estimate_initialization_time(pdf_files, total_size):
    """估算初始化时间"""
    print("\n⏱️ 初始化时间估算")
    print("=" * 50)
    
    # 基于文件大小和数量估算
    estimated_extraction_time = len(pdf_files) * 2  # 假设每个文件2秒
    estimated_embedding_time = total_size / 1024 / 1024 * 5  # 假设每MB需要5秒
    estimated_model_time = 3  # 模型初始化约3秒
    
    total_estimated_time = estimated_extraction_time + estimated_embedding_time + estimated_model_time
    
    print(f"PDF提取估算时间: {estimated_extraction_time:.1f}秒")
    print(f"向量化估算时间: {estimated_embedding_time:.1f}秒")
    print(f"模型初始化估算时间: {estimated_model_time:.1f}秒")
    print(f"总估算时间: {total_estimated_time:.1f}秒")
    
    if total_estimated_time > 60:
        print("⚠️ 预计初始化时间较长，建议使用缓存优化")

def provide_optimization_suggestions(pdf_files, total_size):
    """提供优化建议"""
    print("\n💡 优化建议")
    print("=" * 50)
    
    suggestions = []
    
    if len(pdf_files) > 5:
        suggestions.append("📚 PDF文件较多，建议分批处理或使用缓存")
    
    if total_size > 50 * 1024 * 1024:  # 50MB
        suggestions.append("📏 PDF文件较大，建议压缩或分割文件")
    
    if len(pdf_files) > 10:
        suggestions.append("⚡ 考虑使用并行处理提高PDF提取速度")
    
    suggestions.append("💾 使用向量库缓存避免重复计算")
    suggestions.append("🔄 预加载模型减少响应时间")
    
    for suggestion in suggestions:
        print(f"  {suggestion}")

def main():
    print("🚀 开始性能分析...")
    print("=" * 60)
    
    # 分析PDF文件
    pdf_files, total_size = analyze_pdf_files()
    
    # 测试PDF提取速度
    test_pdf_extraction_speed(pdf_files)
    
    # 测试模型初始化
    test_model_initialization()
    
    # 测试网络速度
    test_network_speed()
    
    # 估算初始化时间
    estimate_initialization_time(pdf_files, total_size)
    
    # 提供优化建议
    provide_optimization_suggestions(pdf_files, total_size)
    
    print("\n" + "=" * 60)
    print("✅ 性能分析完成！")
    print("\n💡 建议使用优化版本: python recipechatbot_optimized.py")

if __name__ == "__main__":
    main() 