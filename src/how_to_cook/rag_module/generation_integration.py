"""生成集成模块"""
import logging

from langchain_deepseek import ChatDeepSeek


logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
    """生成集成模块 - 负责LLM集成和回答生成"""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        temperature: float,
        max_tokens: int
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None

        logger.info(f"配置大语言模型：{self.model_name}...")
        self.setup_llm()

    def setup_llm(self):
        """配置大语言模型"""

        self.llm = ChatDeepSeek(
            model=self.model_name,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )