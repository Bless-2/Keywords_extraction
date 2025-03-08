from openai import OpenAI

client = OpenAI(
    api_key="你的 DeepSeek API Key",
    base_url="https://api.deepseek.com"
)


def extract_keywords(text):
    messages = [
        {
            "role": "system",  # 改为 "system" 而非 "developer"
            "content": (
                "你是一位专业的自然语言处理专家。"
                "请根据用户提供的文本，抽取出其中最能反映文本主题的关键词，"
                "并使用逗号进行分隔。"
            )
        },
        {
            "role": "user",
            "content": f"这是我想要提取关键词的文本：\n{text}\n"
        }
    ]

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages
    )

    reasoning_content = response.choices[0].message.reasoning_content
    content = response.choices[0].message.content
    return reasoning_content, content


if __name__ == "__main__":
    text = """
    随着信息技术的不断发展，自然语言处理（NLP）已经成为人工智能领域中重要的研究方向。
    它的应用范围涵盖了机器翻译、文本摘要、情感分析等众多方向。
    """

    reasoning, keywords = extract_keywords(text)
    print("推理过程 (reasoning_content):")
    print(reasoning)
    print("\n抽取到的关键词 (content):")
    print(keywords)
