from embedding import model
from openai import OpenAI

client = OpenAI(api_key="sk-51401f3e32ed4ed7a888690b99eeea91", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
def generate_answer(query, context):

    prompt = f"""
    请根据以下内容回答问题：

    内容:
    {context}

    问题:
    {query}

    如果内容中没有答案，请说明未找到。
    """

    response = client.chat.completions.create(
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def rag_query(query, vector_store):

    query_embedding = model.encode([query])

    results = vector_store.search(query_embedding)

    context = "\n".join(results)

    answer = generate_answer(query, context)

    return answer