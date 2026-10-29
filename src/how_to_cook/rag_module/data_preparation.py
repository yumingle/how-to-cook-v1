"""数据准备模块"""
import re
import json
import logging
import hashlib
import uuid
from pathlib import Path
from typing import Dict, List, Any

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

logger = logging.getLogger(__name__)

class DataPreparationModule:
    """数据准备模块 - 数据加载、清洗和预处理"""

    # 菜品名、菜品难度统一配置
    CATEGORY_MAP = {
        "meat_dish": "荤菜",
        "vegetable_dish": "素菜",
        "soup": "汤品",
        "dessert": "甜品",
        "breakfast": "早餐",
        "staple": "主食",
        "aquatic": "水产",
        "condiment": "调料",
        "drink": "饮品"
    }
    CATEGORY_LABELS = list(set(CATEGORY_MAP.values()))
    DIFFICULTY_LABELS = ["非常简单", "简单", "中等", "困难", "非常困难"]


    def __init__(self, data_path: str):
        """初始化数据准备模块

        Args:
            data_path: 外部文档的存储路径
        """
        
        self.data_path = data_path
        self.documents: list[Document] = [] # 父文档（完整食谱）
        self.chunks: list[Document] = [] # 子文档（父文档按标题分割的子块）
        self.parent_child_map: Dict[str, str] = {} # 父子文档映射（子文档ID -> 父文档ID）


    @classmethod
    def get_supported_categories(cls) -> List[str]:
        """对外提供支持的分类标签列表"""
        return cls.CATEGORY_LABELS


    @classmethod
    def get_supported_difficulties(cls) -> List[str]:
        """对外提供支持的难度标签列表"""
        return cls.DIFFICULTY_LABELS


    def load_documents(self) -> List[Document]:
        """加载文档数据

        Returns:
            加载的文档列表
        """

        logger.info(f"正在从 {self.data_path} 加载文档数据...")

        documents = []
        data_path = Path(self.data_path)
        data_root = Path(self.data_path).resolve()

        for md_file in data_path.rglob("*.md"):
            try:
                # 读取父文档（一个完整的md文件）
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # 为父文档分配唯一确定性ID
                # 为什么不直接用UUID：为了支持增量更新、可复现
                try:
                    relative_path = Path(md_file).resolve().relative_to(data_root).as_posix()
                except ValueError:
                    relative_path = Path(md_file).resolve().as_posix()
                parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()

                # 创建Document对象
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(md_file),
                        "parent_id": parent_id,
                        "doc_type": "parent"
                    }
                )

                # 添加元数据
                self._enhance_metadata(doc)

                documents.append(doc)

            except Exception as e:
                logger.warning(f"读取 {md_file} 失败：{e}")

            self.documents = documents
            logger.info(f"成功加载 {len(documents)} 个文档...")
            return documents

    def _enhance_metadata(self, doc: Document):
        """添加文档元数据

        Args:
            doc: 需要添加元数据的文档
        """

        file_path = Path(doc.metadata.get("source", ""))
        path_parts = file_path.parts

        # 提取菜品分类
        doc.metadata["category"] = "其他"
        for key, value in self.CATEGORY_MAP.items():
            if key in path_parts:
                doc.metadata["category"] = value
                break

        # 提取菜品名称
        doc.metadata["dish_name"] = file_path.stem

        # 提取难度等级
        content = doc.page_content
        
        star_match = re.search(r"★+", content)

        if star_match:
            difficulty_map = {5: "非常困难", 4: "困难", 3: "中等", 2: "简单", 1: "非常简单"}
            star_count = len(star_match.group())
            doc.metadata["difficulty"] = difficulty_map.get(star_count, "未知")
        else:
            doc.metadata["difficulty"] = "未知"

    def chunk_documents(self) -> List[Document]:
        """Markdown结构感知分块

        Returns:
            分块后的子文档列表
        """

        logger.info(f"md 文档分块...")

        if not self.documents:
            raise ValueError("文档未加载...")

        # 文档分割
        chunks = self._markdown_header_split()

        # 添加元数据
        for i, chunk in enumerate(chunks):
            if "child_id" not in chunk.metadata:
                chunk.metadata["child_id"] = str(uuid.uuid4())
            chunk.metadata["batch_index"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        self.chunks = chunks
        logger.info(f"文档分割结束，共生成 {len(chunks)} 个文档块...")
        return chunks



    def _markdown_header_split(self) -> List[Document]:
        """使用Markdown标题分割器进行结构化分割

        Returns:
            分割后的子文档列表
        """

        # 定义要分割的标题层级
        headers_to_split_on = [
            ("#", "主标题"),      # 菜品名称
            ("##", "二级标题"),   # 必备原料、计算、操作等
            ("###", "三级标题")   # 简易版本、复杂版本等
        ]

        # 创建Markdown分割器
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False  # 保留标题，便于理解上下文
        )

        chunks = []

        for doc in self.documents:
            try:
                md_chunks = markdown_splitter.split_text(doc)
                logger.debug(f"文档 {doc.metadata.get('dish_name')} 分割成 {len(md_chunks)} 个chunk...")

                parent_id = doc.metadata["parent_id"]

                
                for i, md_chunk in enumerate(md_chunks):
                    # 为子文档添加元数据
                    child_id = str(uuid.uuid4())

                    md_chunk.metadata.update(doc.metadata)
                    md_chunk.metadata.update({
                        "child_id": child_id,
                        "doc_type": "child",
                        "chunk_index": i
                    })

                    # 建立父子文档映射关系
                    self.parent_child_map[child_id] = parent_id

                chunks.extend(md_chunks)

            except Exception as e:
                logger.warning(f"文档 {doc.metadata.get('source')} 分割失败：{e}")
                chunks.append(doc)

            return chunks

    def filter_documents_by_category(self, category: str) -> List[Document]:
        """按分类过滤文档

        Args:
            category: 菜品类别
        
        Returns:
            某一菜品类别下的文档列表
        """

        return [doc for doc in self.documents if doc.metadata.get("catogery") == category]


    def filter_documents_by_difficulty(self, difficulty: str) -> List[Document]:
        """按难度过滤文档
        
        Args:
            difficulty: 难度等级
        
        Returns:
            某一难度等级下的文档列表
        """

        return [doc for doc in self.documents if doc.metadata.get("difficulty") == difficulty]


    def get_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息

        Returns:
            统计数据字典
        """

        if not self.documents:
            return {}

        categories = {}
        difficulties = {}

        for doc in self.documents:
            category = doc.metadata.get("category")
            categories[category] = categories.get(category, 0) + 1

            difficulty = doc.metadata.get("difficulty")
            difficulties[difficulty] = doc.metadata.get(difficulty, 0) + 1

        avg_chunk_size = sum(chunk.metadata.get("chunk_size") for chunk in self.chunks) / len(self.chunks)

        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "categories": categories,
            "difficulties": difficulties,
            "avg_chunk_size": avg_chunk_size
        }


    def export_metadata(self, output_path: str):
        """导出元数据到JSON文件

        Args:
            output_path: JSON保存路径
        """

        metadata_list = []
        for doc in self.documents:
            metadata_list.append({
                "source": doc.metadata.get("source"),
                "dish_name": doc.metadata.get("dish_name"),
                "category": doc.metadata.get("category"),
                "difficulty": doc.metadata.get("difficulty"),
                "content_length": len(doc.page_content)
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=4)

        logger.info(f"文档元数据已保存到 {output_path} ...")

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """根据子块获取对应的父文档（智能去重）

        Args:
            child_chunks: 子文档列表
        
        Returns:
            去重后，子文档列表对应的父文档列表
        """

        parent_relevance = {}
        parent_docs_map = {}

        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")

            parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1
            
            if parent_id not in parent_docs_map:
                for doc in self.documents:
                    if doc.metadata.get("parent_id") == parent_id:
                        parent_docs_map[parent_id] = doc
                        break
                

        sorted_parent_ids = sorted(parent_relevance.keys,
                                   key=lambda x: parent_relevance[x],
                                   reverse=True
                                )

        parent_docs = []
        for parent_id in sorted_parent_ids:
            parent_docs.append(parent_docs_map[parent_id])

        parent_info = []
        for doc in parent_docs:
            dish_name = doc.metadata.get("dish_name")
            parent_id = doc.metadata.get("parent_id")
            relevance_count = parent_relevance[parent_id]
            parent_info.append(f"{dish_name}(有{relevance_count}个相关块)")

        logger.info(f"从 {len(child_chunks)} 中找到 {len(parent_docs)} 个去重的父文档：{", ".join(parent_info)}")

        return parent_docs

