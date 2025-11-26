import os
import json
import requests
from typing import List, Dict, Optional, Any

class DeepSeekClient:
    """
    DeepSeek大模型API客户端
    提供简单的接口来调用DeepSeek聊天模型，支持多轮对话
    """
    
    def __init__(self, api_key: Optional[str] = None, api_url: str = "https://api.deepseek.com/chat/completions", use_reasoning: bool = False):
        """
        初始化DeepSeek客户端
        
        Args:
            api_key: DeepSeek API密钥，如果为None则尝试从环境变量DEEPSEEK_API_KEY获取
            api_url: DeepSeek API端点URL
            use_reasoning: 是否默认使用思考模式（DeepSeek-V3.2-Exp的思考模式）
        """
        # 优先使用传入的api_key，如果没有则尝试从环境变量获取
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            # 尝试从api_key.json文件读取，使用绝对路径
            try:
                api_key_path = '/Users/chenjiali/workplace/aitrade/config/api_key.json'
                with open(api_key_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 支持多种可能的键名
                    self.api_key = config.get('deepseek_api_key') or config.get('deepseek')
            except (FileNotFoundError, json.JSONDecodeError):
                pass
                
        if not self.api_key:
            raise ValueError("未提供DeepSeek API密钥，请通过参数传入、设置环境变量DEEPSEEK_API_KEY或在api_key.json文件中配置")
            
        self.api_url = api_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.use_reasoning = use_reasoning  # 保存默认的思考模式设置
        # 初始化对话历史，用于多轮对话
        self.conversation_history: List[Dict[str, str]] = []
        self.system_prompt: Optional[str] = None
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]],
                       model: str = "deepseek-chat",
                       use_reasoning: Optional[bool] = None,
                       stream: bool = False,
                       **kwargs) -> Dict[str, Any]:
        """
        调用DeepSeek聊天完成API
        
        Args:
            messages: 消息列表，每个消息包含role和content字段
            model: 要使用的模型名称
            use_reasoning: 是否使用思考模式
            stream: 是否使用流式响应
            **kwargs: 其他可选参数
            
        Returns:
            API响应结果
        """
        # 如果没有指定use_reasoning参数，则使用实例的默认设置
        if use_reasoning is None:
            use_reasoning = self.use_reasoning
            
        # 根据是否启用思考模式选择合适的模型
        if use_reasoning:
            # DeepSeek-V3.2-Exp的思考模式
            model = "deepseek-reasoner"
        # 否则使用默认的非思考模式（deepseek-chat）
        
        # 构建请求体
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        # 添加其他可选参数
        payload.update(kwargs)
        
        try:
            # 发送POST请求
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=60,  # 设置超时时间
                stream=stream  # 启用流式响应
            )
            
            # 检查响应状态
            response.raise_for_status()
            
            # 处理流式响应
            if stream:
                # 对于流式响应，我们会实时处理并yield每个片段
                # 由于函数返回类型仍然是Dict，我们需要收集所有片段
                full_response = {}
                all_content = ""
                
                for chunk in response.iter_lines():
                    if chunk:
                        # 去掉data: 前缀
                        chunk_str = chunk.decode('utf-8')
                        if chunk_str.startswith('data:'):
                            chunk_data = chunk_str[5:].strip()
                            if chunk_data == '[DONE]':
                                break
                            try:
                                chunk_json = json.loads(chunk_data)
                                # 提取content部分
                                if 'choices' in chunk_json and len(chunk_json['choices']) > 0:
                                    delta = chunk_json['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        # 确保content_chunk是字符串，避免None类型导致的拼接错误
                                        content_chunk = delta['content'] or ''
                                        if isinstance(content_chunk, str):
                                            all_content += content_chunk
                                            # 实时打印内容
                                            print(content_chunk, end='', flush=True)
                            except json.JSONDecodeError:
                                pass
                
                # 构建完整响应
                full_response['choices'] = [
                    {
                        'message': {
                            'content': all_content
                        }
                    }
                ]
                print()  # 换行
                return full_response
            else:
                # 非流式响应，返回完整JSON
                return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"API请求出错: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    print(f"错误详情: {json.dumps(error_detail, ensure_ascii=False, indent=2)}")
                except:
                    print(f"响应内容: {e.response.text}")
            raise
    
    def send_message(self, prompt: str, system_prompt: str = "You are a helpful assistant.", use_reasoning: Optional[bool] = None, **kwargs) -> str:
        """
        发送单轮对话消息并获取回复
        
        Args:
            prompt: 用户提问内容
            system_prompt: 系统提示信息
            **kwargs: 其他可选参数
            
        Returns:
            模型回复内容
        """
        # 构建消息列表
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # 调用API，传递use_reasoning参数
        response = self.chat_completion(messages, use_reasoning=use_reasoning, **kwargs)
        
        # 提取回复内容
        if "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["message"]["content"]
        else:
            raise ValueError(f"API响应格式异常: {response}")
    
    def start_conversation(self, system_prompt: str = "You are a helpful assistant.") -> None:
        """
        开始新的多轮对话
        
        Args:
            system_prompt: 系统提示信息
        """
        # 重置对话历史
        self.reset_conversation()
        # 设置系统提示
        self.system_prompt = system_prompt
        # 将系统提示添加到对话历史中
        self.conversation_history.append({"role": "system", "content": system_prompt})
    
    def send_message_round(self, prompt: str, use_reasoning: Optional[bool] = None, **kwargs) -> str:
        """
        发送多轮对话中的一轮消息并获取回复
        自动维护对话历史，上下文在库代码中处理
        
        Args:
            prompt: 用户提问内容
            use_reasoning: 是否使用思考模式
            **kwargs: 其他可选参数
            
        Returns:
            模型回复内容
        """
        # 如果对话历史为空，初始化系统提示
        if not self.conversation_history:
            self.start_conversation()
        
        # 添加用户消息到对话历史
        self.conversation_history.append({"role": "user", "content": prompt})
        
        # 调用API，传递use_reasoning参数
        response = self.chat_completion(self.conversation_history, use_reasoning=use_reasoning, **kwargs)
        
        # 提取回复内容
        if "choices" in response and len(response["choices"]) > 0:
            assistant_response = response["choices"][0]["message"]["content"]
            # 将助手回复添加到对话历史
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            return assistant_response
        else:
            raise ValueError(f"API响应格式异常: {response}")
    
    def reset_conversation(self) -> None:
        """
        重置对话历史
        """
        self.conversation_history = []
        self.system_prompt = None
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        获取当前对话历史
        
        Returns:
            对话历史列表
        """
        return self.conversation_history.copy()
    
    def save_conversation(self, file_path: str) -> None:
        """
        保存对话历史到文件
        
        Args:
            file_path: 保存的文件路径
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({
                "system_prompt": self.system_prompt,
                "conversation_history": self.conversation_history
            }, f, ensure_ascii=False, indent=2)
    
    def load_conversation(self, file_path: str) -> None:
        """
        从文件加载对话历史
        
        Args:
            file_path: 对话历史文件路径
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.system_prompt = data.get("system_prompt")
            self.conversation_history = data.get("conversation_history", [])


def main():
    """
    示例用法
    """
    try:
        # 创建客户端实例
        client = DeepSeekClient()
        
        # 单轮对话示例
        print("=== 单轮对话示例 ===")
        response = client.send_message("你好，介绍一下自己")
        print(f"模型回复: {response}")
        
        # 多轮对话示例
        print("\n=== 多轮对话示例 ===")
        # 开始对话
        client.start_conversation("你是一个专业的股票分析师")
        
        # 第一轮对话
        response1 = client.send_message_round("解释一下什么是市盈率")
        print(f"用户: 解释一下什么是市盈率")
        print(f"模型: {response1}")
        
        # 第二轮对话，上下文自动包含在库中
        response2 = client.send_message_round("它和市净率有什么区别")
        print(f"用户: 它和市净率有什么区别")
        print(f"模型: {response2}")
        
        # 重置对话并开始新对话
        print("\n=== 重置对话后 ===")
        client.reset_conversation()
        client.start_conversation("你是一个代码助手")
        response3 = client.send_message_round("如何在Python中实现多线程")
        print(f"用户: 如何在Python中实现多线程")
        print(f"模型: {response3}")
        
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()