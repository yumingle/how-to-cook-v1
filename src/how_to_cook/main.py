"""RAG系统主程序"""
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dotenv import load_dotenv
import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import RAGConfig, DEFAULT_CONFIG
from rag_module import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule
)

load_dotenv()

# 日志配置·
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class RecipeRAGSystem:
    """RAG系统主类"""

    def __init__(self, config: RAGConfig = DEFAULT_CONFIG):
        """初始化RGA系统"""

        self.config = config
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        logger.info("初始化 RAG 系统...")
        self.init_system()


    def init_system(self):
        """初始化各个模块"""

        # 数据准备模块
        logger.info("初始化数据准备模块...")
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")
        self.data_module = DataPreparationModule(self.config.data_path)

        # 索引构建模块
        logger.info("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_path=self.config.index_path
        )

        # 检索优化模块
        vectorstore, chunks = self.build_knowledge_base()
        logger.info("初始化检索优化模块...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

        # 生成集成模块
        logger.info("初始化生成集成模块...")
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise ValueError("请设置 DEEPSEEK_API_KEY")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )



    def build_knowledge_base(self) -> Tuple[Optional[FAISS], Optional[List[Document]]]:
        """构建知识库"""

        return None, None

    def run_interactive(self):
        """交互式问答"""
        pass

def main():
    try:
        rag_system = RecipeRAGSystem()
        rag_system.run_interactive()
    except Exception as e:
        logger.error(f"系统运行错误：{e}")

if __name__ == "__main__":
    main()