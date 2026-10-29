"""索引优化模块"""
import logging
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class RetrievalOptimizationModule:
    """索引优化模块 - 负责负责混合检索和元数据过滤"""

    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.vector_retriever = None
        self.bm25_retriever = None

        self.setup_retriever()

    def setup_retriever(self):
        """设置检索器"""

        logger.info("初始化 向量检索器 和 BM25检索器...")
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=5
        )

