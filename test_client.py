import socket
import json
from dataclasses import dataclass

@dataclass
class UserInfo:
    Name: str
    Query: str
    Answer: str = ""

def test_recipe_server():
    """测试食谱推荐服务器"""
    host, port = "127.0.0.1", 25001
    
    try:
        # 创建客户端socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))
        print(f"已连接到服务器 {host}:{port}")
        
        # 测试查询
        test_queries = [
            "I want to cook Mexican food, what ingredients do you recommend? Only recommend the name of the dish and the ingredients",
            "I need an easy pasta recipe? Just suggest the name and ingredients",
            "Are there any recipes suitable for beginners? Just recommend the names and ingredients"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- 测试 {i} ---")
            print(f"发送查询: {query}")
            
            # 创建UserInfo对象
            user_info = UserInfo(Name="TestUser", Query=query)
            
            # 发送JSON数据
            json_data = json.dumps(user_info.__dict__) + '\n'
            client_socket.sendall(json_data.encode())
            
            # 接收响应
            response_data = client_socket.recv(4096).decode()
            if response_data:
                try:
                    response_info = json.loads(response_data)
                    print(f"收到答案: {response_info.get('Answer', '无答案')}")
                except json.JSONDecodeError:
                    print(f"收到原始响应: {response_data}")
            else:
                print("未收到响应")
        
        client_socket.close()
        print("\n测试完成")
        
    except ConnectionRefusedError:
        print(f"无法连接到服务器 {host}:{port}，请确保服务器正在运行")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")

if __name__ == "__main__":
    test_recipe_server() 