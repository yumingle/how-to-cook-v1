"""RAGp配置"""
from dataclasses import dataclass, asdict
from typing import Dict, Any
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = str(PROJECT_ROOT / "data" / "cook")
INDEX_PATH = str(PROJECT_ROOT / "vector_index")

@dataclass
class RAGConfig:
    """RAG配置类"""

    # 路径配置
    data_path: str = DATA_PATH
    index_path: str = INDEX_PATH

    # 模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = "deepseek-v4-flash"

    # 检索配置
    top_k: int = 3

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "RAGConfig":
        """从字典创建配置对象"""
        return RAGConfig(**config)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


DEFAULT_CONFIG = RAGConfig()