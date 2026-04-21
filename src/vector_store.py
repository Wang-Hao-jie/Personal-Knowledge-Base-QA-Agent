from openai import OpenAI
from langchain_chroma import Chroma

from src.config import Settings


class DashScopeCompatibleEmbeddings:
    """Use OpenAI-compatible embeddings API with DashScope endpoint."""

    def __init__(self):
        api_key = Settings.get_api_key()
        if not api_key:
            raise ValueError(
                "未检测到 API Key。请设置 DASHSCOPE_API_KEY（或 OPENAI_API_KEY）。"
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url=Settings.get_openai_base_url(),
        )
        self.model = Settings.get_embedding_model()

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            content = text if isinstance(text, str) else str(text)
            resp = self.client.embeddings.create(
                model=self.model,
                input=content,
            )
            embeddings.append(resp.data[0].embedding)
        return embeddings

    def embed_query(self, text):
        content = text if isinstance(text, str) else str(text)
        resp = self.client.embeddings.create(
            model=self.model,
            input=content,
        )
        return resp.data[0].embedding


def build_embeddings():
    return DashScopeCompatibleEmbeddings()


def build_vector_store(chunks):
    embeddings = build_embeddings()
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=Settings.get_vector_db_dir(),
    )
    return db


def load_retriever():
    embeddings = build_embeddings()
    db = Chroma(
        embedding_function=embeddings,
        persist_directory=Settings.get_vector_db_dir(),
    )
    return db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": Settings.get_retrieval_k()},
    )
