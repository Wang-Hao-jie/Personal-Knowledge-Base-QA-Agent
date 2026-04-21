from openai import OpenAI
from langchain_core.prompts import PromptTemplate

from src.config import Settings
from src.vector_store import load_retriever


PROMPT_TEMPLATE = """你是一个专业的文档助手。请仅基于以下上下文回答用户的问题：
----------------
{context}
----------------
问题：{question}
要求：
1. 答案必须来自上下文；
2. 若上下文无相关信息，请回答“未找到相关内容”；
3. 回答尽量简洁、准确。
"""


def build_qwen_client() -> OpenAI:
    api_key = Settings.get_api_key()
    if not api_key:
        raise ValueError(
            "未检测到 API Key。请设置 DASHSCOPE_API_KEY（或 OPENAI_API_KEY）。"
        )
    return OpenAI(
        api_key=api_key,
        base_url=Settings.get_openai_base_url(),
    )


def create_qa_chain():
    client = build_qwen_client()
    retriever = load_retriever()
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    return {
        "client": client,
        "retriever": retriever,
        "prompt": prompt,
    }


def ask_question(chain_parts, question: str):
    retriever = chain_parts["retriever"]
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    answer = _create_completion_text(chain_parts, context, question)
    return {"result": answer, "source_documents": docs}


def _create_completion_text(chain_parts, context: str, question: str) -> str:
    client = chain_parts["client"]
    prompt = chain_parts["prompt"]
    final_prompt = prompt.format(context=context, question=question)
    messages = [{"role": "user", "content": final_prompt}]
    completion = client.chat.completions.create(
        model=Settings.get_chat_model(),
        messages=messages,
        extra_body={"enable_thinking": True},
        stream=False,
    )
    return completion.choices[0].message.content or "未找到相关内容"


def stream_question(chain_parts, question: str):
    client = chain_parts["client"]
    prompt = chain_parts["prompt"]
    retriever = chain_parts["retriever"]

    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = prompt.format(context=context, question=question)
    messages = [{"role": "user", "content": final_prompt}]

    completion = client.chat.completions.create(
        model=Settings.get_chat_model(),
        messages=messages,
        extra_body={"enable_thinking": True},
        stream=True,
    )

    reasoning_text = ""
    answer_text = ""
    for chunk in completion:
        delta = chunk.choices[0].delta
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            reasoning_text += delta.reasoning_content
            yield {
                "reasoning": reasoning_text,
                "answer": answer_text,
                "source_documents": docs,
            }
        if hasattr(delta, "content") and delta.content:
            answer_text += delta.content
            yield {
                "reasoning": reasoning_text,
                "answer": answer_text,
                "source_documents": docs,
            }


def stream_chat(messages):
    client = build_qwen_client()
    completion = client.chat.completions.create(
        model=Settings.get_chat_model(),
        messages=messages,
        extra_body={"enable_thinking": True},
        stream=True,
    )

    reasoning_text = ""
    answer_text = ""
    for chunk in completion:
        delta = chunk.choices[0].delta
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            reasoning_text += delta.reasoning_content
            yield {"reasoning": reasoning_text, "answer": answer_text}
        if hasattr(delta, "content") and delta.content:
            answer_text += delta.content
            yield {"reasoning": reasoning_text, "answer": answer_text}
