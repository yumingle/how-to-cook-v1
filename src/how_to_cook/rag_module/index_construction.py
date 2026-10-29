"""索引构建模块"""
import logging

from  langchain_huggingface import HuggingFaceEmbeddings


logger = logging.getLogger(__name__)

class IndexConstructionModule:
    """索引构建模块 - 负责文档向量化和索引构建"""

    def __init__(self, model_name: str, index_path: str):
        self.model_name = model_name
        self.index_path = index_path
        self.embeddings = None
        self.vectorstores = None

        logger.info(f"配置嵌入模型：{self.model_name}...")
        self.setup_embeddings()

    def setup_embeddings(self):
        """配置嵌入模型"""

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs = {"device": "cpu"},
            encode_kwargs = {"normalize_embeddings": True}
        )