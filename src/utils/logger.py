import os
import sys
from datetime import datetime
from typing import Optional

class Logger:
    """
    统一日志模块，处理控制台输出和文件日志记录
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """
        初始化日志模块
        
        Args:
            log_file: 日志文件路径，如果为None则只输出到控制台
        """
        self.log_file = log_file
        
    def log(self, message: str):
        """
        记录日志到控制台和文件
        
        Args:
            message: 日志消息
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        # 输出到控制台
        print(log_message)
        
        # 写入日志文件
        if self.log_file:
            try:
                # 确保日志文件目录存在
                log_dir = os.path.dirname(self.log_file)
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(log_message + '\n')
            except Exception as e:
                # 如果无法写入日志文件，只打印到控制台
                print(f"无法写入日志文件: {e}")
    
    def debug(self, message: str):
        """
        记录调试级别日志
        
        Args:
            message: 日志消息
        """
        self.log(f"[调试] {message}")
    
    def info(self, message: str):
        """
        记录信息级别日志
        
        Args:
            message: 日志消息
        """
        self.log(f"[信息] {message}")
    
    def warning(self, message: str):
        """
        记录警告级别日志
        
        Args:
            message: 日志消息
        """
        self.log(f"[警告] {message}")
    
    def error(self, message: str):
        """
        记录错误级别日志
        
        Args:
            message: 日志消息
        """
        self.log(f"[错误] {message}")
    
    def critical(self, message: str):
        """
        记录严重级别日志
        
        Args:
            message: 日志消息
        """
        self.log(f"[严重] {message}")
    
    def set_log_file(self, log_file: str):
        """
        设置日志文件路径
        
        Args:
            log_file: 新的日志文件路径
        """
        self.log_file = log_file

# 创建全局日志实例，方便其他模块直接导入使用
global_logger = Logger()

# 提供简便的函数接口
def log(message: str):
    """
    全局日志函数
    
    Args:
        message: 日志消息
    """
    global_logger.log(message)

def debug(message: str):
    """
    全局调试日志函数
    
    Args:
        message: 日志消息
    """
    global_logger.debug(message)

def info(message: str):
    """
    全局信息日志函数
    
    Args:
        message: 日志消息
    """
    global_logger.info(message)

def warning(message: str):
    """
    全局警告日志函数
    
    Args:
        message: 日志消息
    """
    global_logger.warning(message)

def error(message: str):
    """
    全局错误日志函数
    
    Args:
        message: 日志消息
    """
    global_logger.error(message)

def critical(message: str):
    """
    全局严重日志函数
    
    Args:
        message: 日志消息
    """
    global_logger.critical(message)

def set_global_log_file(log_file: str):
    """
    设置全局日志文件
    
    Args:
        log_file: 日志文件路径
    """
    global_logger.set_log_file(log_file)