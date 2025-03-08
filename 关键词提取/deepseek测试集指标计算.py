import json
import time
from openai import OpenAI
from nltk.stem import PorterStemmer

# 初始化 DeepSeek API 客户端
client = OpenAI(
    api_key="你的 DeepSeek API Key",
    base_url="https://api.deepseek.com"
)

# 初始化 Porter 词干化器
ps = PorterStemmer()


def extract_keywords_with_deepseek(text):
    """
    使用 DeepSeek `deepseek-reasoner` 提取关键词。
    """
    messages = [
        {
            "role": "system",
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

    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages
        )

        # DeepSeek 返回的数据格式
        content = response.choices[0].message.content.strip()
        return content
    except Exception as e:
        print(f"DeepSeek API 调用失败: {e}")
        return ""


def stem_words(word_list):
    """
    将关键词短语进行词干化（Stemming）
    """
    return {" ".join([ps.stem(word) for word in phrase.split()]) for phrase in word_list}


def parse_keywords(keyword_str):
    """
    解析 DeepSeek 返回的逗号分隔的关键词字符串，并进行词干化
    """
    raw_keywords = [k.strip() for k in keyword_str.split(",") if k.strip()]
    return list(stem_words(raw_keywords))


def fuzzy_match(predicted, ground_truth):
    """
    子串匹配：若 predicted 和 ground_truth 之间有部分重叠，则算作匹配
    """
    return any(gt in predicted or predicted in gt for gt in ground_truth)


def evaluate_prediction(predicted_list, ground_truth_list):
    """
    计算 Precision / Recall / F1-score
    - 词干化
    - 允许子串匹配（"data center market" vs "data center" 也算匹配）
    """
    pred_set = set(predicted_list)
    gt_set = set(ground_truth_list)

    tp = sum(1 for pred in pred_set if fuzzy_match(pred, gt_set))
    fp = len(pred_set) - tp
    fn = len(gt_set) - tp

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return precision, recall, f1


def main():
    total_p, total_r, total_f1 = 0.0, 0.0, 0.0
    count = 0

    with open("test.json", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            doc_id = record["document_id"]
            tokens_2d = record["tokens"]

            # 拼接 tokens 为文本
            tokens_flat = [token for line in tokens_2d for token in line]
            text = " ".join(tokens_flat)

            # 词干化 keyphrases
            ground_truth = stem_words(record["keyphrases"])

            # 使用 DeepSeek 提取关键词
            predicted_str = extract_keywords_with_deepseek(text)
            predicted_list = parse_keywords(predicted_str)

            # 评估
            p, r, f1 = evaluate_prediction(predicted_list, ground_truth)
            count += 1

            print(f"[Doc ID={doc_id}]")
            print(f"  Predicted: {predicted_list}")
            print(f"  True Keyphrases: {ground_truth}")
            print(f"  Precision={p:.2f}, Recall={r:.2f}, F1={f1:.2f}\n")

            total_p += p
            total_r += r
            total_f1 += f1

            # 防止 API 速率限制，每次调用后休息 1 秒
            time.sleep(1)

    # **计算并打印总平均 Precision, Recall, F1**
    if count > 0:
        avg_p = total_p / count
        avg_r = total_r / count
        avg_f1 = total_f1 / count
        print("\n=== 结果汇总 ===")
        print(f"文档总数: {count}")
        print(f"平均 Precision={avg_p:.2f}, 平均 Recall={avg_r:.2f}, 平均 F1={avg_f1:.2f}")
    else:
        print("未读取到有效的 JSON 数据。")


if __name__ == "__main__":
    main()
