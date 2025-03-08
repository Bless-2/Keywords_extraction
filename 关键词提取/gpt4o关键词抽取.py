from openai import OpenAI

client = OpenAI(api_key="你的 OpenAI API Key")


def extract_keywords(text):
    prompt = f"""
    你是一位专业的自然语言处理专家。请从以下文本中抽取最能反映主要内容或主题的关键词：
    {text}
    输出要求：关键词以逗号分隔。
    """

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "developer",
                "content": "你是一位专业的自然语言处理专家。请根据用户提供的文本进行关键词抽取。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # 用 .content 来获取字符串而不是通过 ["content"] 的字典索引
    keywords = completion.choices[0].message.content.strip()
    return keywords


if __name__ == "__main__":
    text = """
    随着信息技术的不断发展，自然语言处理（NLP）已经成为人工智能领域中重要的研究方向。
    其应用范围涵盖了机器翻译、文本摘要、情感分析等众多方向。通过对文本数据进行分词、
    词性标注以及依存句法分析等处理手段，可以更好地理解和挖掘语言中的知识与信息。
    """

    result = extract_keywords(text)
    print("抽取的关键词：", result)
