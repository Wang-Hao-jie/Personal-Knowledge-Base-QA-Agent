import os


class Settings:
    @staticmethod
    def get_api_key() -> str:
        # 优先读取通义千问（DashScope）密钥，同时兼容 OPENAI_API_KEY
        return os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY", ""))

    @staticmethod
    def get_openai_base_url() -> str:
        return os.getenv(
            "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    @staticmethod
    def get_embedding_model() -> str:
        return os.getenv("EMBEDDING_MODEL", "text-embedding-v4")

    @staticmethod
    def get_chat_model() -> str:
        return os.getenv("CHAT_MODEL", "qwen3.6-plus")

    @staticmethod
    def get_vector_db_dir() -> str:
        return os.getenv("VECTOR_DB_DIR", "data/chroma_db")

    @staticmethod
    def get_chunk_size() -> int:
        return int(os.getenv("CHUNK_SIZE", "500"))

    @staticmethod
    def get_chunk_overlap() -> int:
        return int(os.getenv("CHUNK_OVERLAP", "100"))

    @staticmethod
    def get_retrieval_k() -> int:
        return int(os.getenv("RETRIEVAL_K", "4"))
