import sys
import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'src'))
from src.utils.deepseek_client import DeepSeekClient

def example_1_config_file_api_key():
    """
    示例1: 从配置文件自动获取API密钥
    """
    print("\n=== 示例1: 从配置文件自动获取API密钥 ===")
    try:
        client = DeepSeekClient()
        print("✓ 客户端初始化成功 (从配置文件获取API密钥)")
        
        print("发送消息并接收流式回复...")
        # 启用流式输出
        response = client.send_message("你好，请简要介绍一下自己。", stream=True)
        
        # 由于在流式模式下，client.send_message已经实时打印了响应
        # 这里我们只需要简单地显示响应已完成
        print("✓ 响应接收完成")
        return True
    except Exception as e:
        print(f"✗ 示例1执行失败: {type(e).__name__}: {str(e)}")
        return False

def example_2_direct_api_key():
    """
    示例2: 直接传入API密钥
    """
    print("\n=== 示例2: 直接传入API密钥 ===")
    try:
        # 注意：在实际应用中，不要硬编码API密钥
        # 这里仅作为示例
        api_key = "sk-9e55f0036ce14232a796f63b71245d76"  # 从api_key.json获取的示例密钥
        client = DeepSeekClient(api_key=api_key)
        print("✓ 客户端初始化成功 (直接传入API密钥)")
        
        print("发送消息并接收流式回复...")
        # 启用流式输出
        response = client.send_message("什么是机器学习？用简单的话解释一下。", stream=True)
        
        print("✓ 响应接收完成")
        return True
    except Exception as e:
        print(f"✗ 示例2执行失败: {type(e).__name__}: {str(e)}")
        return False

def example_3_custom_system_prompt():
    """
    示例3: 使用自定义系统提示
    """
    print("\n=== 示例3: 使用自定义系统提示 ===")
    try:
        client = DeepSeekClient()
        print("✓ 客户端初始化成功")
        
        # 使用自定义系统提示的多轮对话
        messages = [
            {"role": "system", "content": "你是一位专业的厨师助手，专门回答关于烹饪的问题。"},
            {"role": "user", "content": "如何制作简单的意大利番茄面？"}
        ]
        
        print("发送消息并接收流式回复...")
        # 启用流式输出
        response = client.chat_completion(messages, stream=True)
        
        print("✓ 响应接收完成")
        return True
    except Exception as e:
        print(f"✗ 示例3执行失败: {type(e).__name__}: {str(e)}")
        return False

def example_4_multi_turn_conversation():
    """
    示例4: 多轮对话
    """
    print("\n=== 示例4: 多轮对话 ===")
    try:
        client = DeepSeekClient()
        print("✓ 客户端初始化成功")
        
        # 初始化对话历史
        conversation_history = [
            {"role": "user", "content": "请帮我计算25乘以42的结果。"}
        ]
        
        # 第一轮对话
        print("\n第一轮对话 - 发送乘法计算请求...")
        response = client.chat_completion(conversation_history, stream=True)
        
        # 提取完整回复内容用于对话历史
        if isinstance(response, dict):
            response_content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
        else:
            response_content = str(response)
        
        conversation_history.append({"role": "assistant", "content": response_content})
        print("✓ 第一轮对话完成")
        
        # 第二轮对话 - 基于上一轮结果继续提问
        conversation_history.append({"role": "user", "content": "现在请将结果除以15。"})
        print("\n第二轮对话 - 发送除法计算请求...")
        response2 = client.chat_completion(conversation_history, stream=True)
        
        print("✓ 第二轮对话完成")
        return True
    except Exception as e:
        print(f"✗ 示例4执行失败: {type(e).__name__}: {str(e)}")
        return False

def example_5_reasoning_mode():
    """
    示例5: 使用DeepSeek-V3.2-Exp的思考模式
    对于复杂或需要详细推理的问题，思考模式可以提供更深入的分析
    """
    print("\n=== 示例5: 使用DeepSeek-V3.2-Exp的思考模式 ===")
    try:
        # 创建客户端实例
        client = DeepSeekClient()
        print("✓ 客户端初始化成功")
        
        # 使用思考模式发送需要详细推理的问题，启用流式输出
        print("\n发送需要推理的问题，将实时显示思考过程...")
        print("\n【思考模式实时输出开始】")
        response = client.send_message(
            "请详细分析: 如何在不使用额外空间的情况下，判断一个整数是否是回文数？",
            use_reasoning=True,  # 启用思考模式
            stream=True  # 启用流式输出
        )
        print("\n【思考模式实时输出结束】")
        print("✓ 思考模式调用成功")
        return True
    except Exception as e:
        print(f"✗ 示例5执行失败: {type(e).__name__}: {str(e)}")
        return False

def example_6_global_reasoning_mode():
    """
    示例6: 在客户端初始化时设置全局思考模式
    这样所有的请求默认都会使用思考模式
    """
    print("\n=== 示例6: 全局启用DeepSeek-V3.2-Exp思考模式 ===")
    try:
        # 在初始化客户端时启用全局思考模式
        client = DeepSeekClient(use_reasoning=True)
        print("✓ 客户端初始化成功 (全局启用思考模式)")
        
        # 现在默认会使用思考模式，启用流式输出
        print("\n在全局思考模式下发送复杂问题，将实时显示思考过程...")
        print("\n【全局思考模式实时输出开始】")
        response = client.send_message(
            "解释量子计算的基本原理，并举例说明其潜在应用。",
            stream=True
        )
        print("\n【全局思考模式实时输出结束】")
        
        # 即使在全局启用后，仍可以在单个请求中覆盖设置
        print("\n在全局思考模式下，临时使用非思考模式...")
        response_non_reasoning = client.send_message(
            "你好，简单介绍一下自己。",
            use_reasoning=False,  # 覆盖全局设置，使用非思考模式
            stream=True
        )
        
        print("✓ 全局思考模式示例完成")
        return True
    except Exception as e:
        print(f"✗ 示例6执行失败: {type(e).__name__}: {str(e)}")
        return False

def example_7_comparison():
    """
    示例7: 对比思考模式和非思考模式的输出差异
    针对同一个复杂问题，展示两种模式的不同处理方式和实时输出效果
    """
    print("\n=== 示例7: 思考模式与非思考模式对比 ===")
    try:
        client = DeepSeekClient()
        print("✓ 客户端初始化成功")
        
        # 同一个复杂问题
        question = "有三个盒子，分别贴着苹果、橙子和苹果橙子混装的标签。已知所有标签都贴错了，你只能从一个盒子里拿出一个水果，如何通过这个水果确定每个盒子里装的是什么？"
        
        # 使用非思考模式
        print("\n1. 使用非思考模式回答:")
        print("【非思考模式实时输出开始】")
        response_normal = client.send_message(question, use_reasoning=False, stream=True)
        print("【非思考模式实时输出结束】")
        
        # 使用思考模式
        print("\n2. 使用思考模式回答:")
        print("【思考模式实时输出开始】")
        response_reasoning = client.send_message(question, use_reasoning=True, stream=True)
        print("【思考模式实时输出结束】")
        
        print("\n✓ 对比完成，思考模式通常提供更详细的推理过程和实时思考效果")
        return True
    except Exception as e:
        print(f"✗ 示例7执行失败: {type(e).__name__}: {str(e)}")
        return False

def main():
    """
    DeepSeekClient多种使用模式示例
    包含：配置文件API密钥、直接传入API密钥、自定义系统提示、多轮对话、思考模式（带流式输出）
    """
    print("=== DeepSeek API客户端多种使用模式示例 ===")
    print("特别说明: DeepSeek-V3.2-Exp模型支持思考模式(deepseek-reasoner)和非思考模式(deepseek-chat)")
    print("当前示例已启用流式输出，可以实时查看大模型的思考和生成过程")
    
    # 运行所有示例
    examples = [
        example_1_config_file_api_key,
        example_2_direct_api_key,
        example_3_custom_system_prompt,
        example_4_multi_turn_conversation,
        example_5_reasoning_mode,
        example_6_global_reasoning_mode,
        example_7_comparison
    ]
    
    # 可以选择只运行思考模式相关示例，获得最佳演示效果
    # examples = [example_5_reasoning_mode, example_6_global_reasoning_mode, example_7_comparison]
    examples = [example_4_multi_turn_conversation]
    success_count = 0
    for i, example_func in enumerate(examples, 1):
        print(f"\n开始运行示例 {i}...")
        if example_func():
            success_count += 1
    
    # 总结结果
    print("\n" + "="*50)
    print(f"示例执行总结: 成功 {success_count}/{len(examples)} 个示例")
    if success_count == len(examples):
        print("✓ 所有示例执行成功！")
    else:
        print("! 部分示例执行失败，请检查错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main()