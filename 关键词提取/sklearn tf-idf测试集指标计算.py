import json
import time
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# 下载 nltk 需要的资源
nltk.download('stopwords')
nltk.download('punkt')

# 初始化 Porter 词干化器
ps = PorterStemmer()

# 设置停用词
stoplist = stopwords.words('english') + list(string.punctuation)


def extract_keywords_with_sklearn(text, top_n=10):
    """
    使用 sklearn 的 TfidfVectorizer 提取关键词
    """
    try:
        # 初始化 TF-IDF 向量器
        vectorizer = TfidfVectorizer(stop_words=stoplist, max_features=5000)

        # 计算 TF-IDF 矩阵
        X = vectorizer.fit_transform([text])

        # 获取词汇表和对应的 TF-IDF 分数
        feature_array = vectorizer.get_feature_names_out()
        tfidf_scores = X.toarray()[0]

        # 选取最高分数的前 top_n 个关键词
        tfidf_dict = dict(zip(feature_array, tfidf_scores))
        sorted_keywords = sorted(tfidf_dict, key=tfidf_dict.get, reverse=True)[:top_n]

        return sorted_keywords
    except Exception as e:
        print(f"sklearn TF-IDF 关键词提取失败: {e}")
        return []


def stem_words(word_list):
    """
    词干化（Stemming）以保证一致性
    """
    return {" ".join([ps.stem(word) for word in phrase.split()]) for phrase in word_list}


def fuzzy_match(predicted, ground_truth):
    """
    子串匹配：如果 predicted 和 ground_truth 之间有部分重叠，则算作匹配
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

            # 使用 sklearn TF-IDF 提取关键词
            predicted_list = extract_keywords_with_sklearn(text)
            predicted_list = stem_words(predicted_list)

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

            # 适当延迟，避免文件读取过快
            time.sleep(0.1)

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
