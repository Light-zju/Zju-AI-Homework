import json
import requests
import sys


import csv
import json
import re
from openai import OpenAI
import time

# 配置大模型API参数
client = OpenAI(
    api_key="dab88b90d1466275d34b5af41eab74d4aff5768d",
    base_url="https://aistudio.baidu.com/llm/lmapi/v3"
)


def parse_response(content):
    """
    解析大模型返回的响应内容
    处理包含代码块的JSON响应
    """
    # 检查是否包含代码块
    # print(content)
    code_block_pattern = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
    # print(code_block_pattern)
    match = code_block_pattern.search(content)

    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"JSON解析失败: {json_str}")
            return None
    else:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print(f"直接JSON解析失败: {content}")
            return None


def call_large_model(symptoms):
    """
    调用ERNIE X1大模型进行证型治法生成
    返回结构化JSON结果
    """
    system_prompt = """
    你是一位经验丰富的中医专家，请根据患者症状描述判断：
    1. 证型：需符合慢性淋巴细胞白血病中医辨证标准
    2. 治法：需与证型对应且符合中医治疗原则
    请用JSON格式输出：{"证型":"", "治法":""}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"患者症状：{symptoms}"}
    ]

    try:

        response = client.chat.completions.create(
            model="ernie-4.5-0.3b",
            messages=messages,
            max_completion_tokens=512,
            temperature=0.3,
            top_p=0.7
        )

        # 解析响应内容
        content = response.choices[0].message.content
        result = parse_response(content)

        if result:
            # print(result)
            return result
        else:
            return {"证型": "未知", "治法": "待定"}

    except Exception as e:
        print(f"API调用异常: {str(e)}")
        return {"证型": "未知", "治法": "待定"}


# 读取result.csv文件中的症状数据
train_data = './datasets/68f201a04e0f8ad44a62069b-momodel/train_data.csv'
symptoms_data = []
ZX = []  # 证型
ZF = []  # 治法

with open(train_data, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        symptoms_data.append(row[1])  # 提取症状列
        ZX.append(row[2])  # 提取症状列
        ZF.append(row[3])  # 提取症状列



# 生成新的结果数据
predict_zx = []
predict_zf = []

def predict(symptom):
    model_result = call_large_model(symptom)
    return model_result['证型'], model_result['治法']


for symptom in symptoms_data:
    zx, zf = predict(symptom)
    predict_zx.append(zx), predict_zf.append(zf)
    time.sleep(0.2)  # 调用间隔，避免过快请求导致API限制


predict_zx


predict_zf


def get_embedding(text):
    """
    获取文本的向量表示
    """
    url = "https://qianfan.baidubce.com/v2/embeddings"
    payload = json.dumps({
        "model": "bge-large-zh",
        "input": [text]
    })
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer bce-v3/ALTAK-5tfX41HMR5wReJjGJL1pP/ec6d6701e0a9b84dd4951ce977f2a9bc5c624d53'
    }

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data['data'][0]['embedding']
    except Exception as e:
        print(json.dumps({
            "code": 1,
            "errorMsg": f"Embedding API error: {str(e)}",
            "score": 0.0,
            "data": [{"score": 0}]
        }, ensure_ascii=False), flush=True)


get_embedding('how old are you')


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_similarity(text1, text2):
    """
    计算两个文本的相似度
    """
    if not text1.strip() or not text2.strip():
        return 0.0
    try:
        emb1 = get_embedding(text1)
        emb2 = get_embedding(text2)
        sim = cosine_similarity(
            np.array(emb1).reshape(1, -1),
            np.array(emb2).reshape(1, -1)
        )
        return float(sim[0][0])
    except Exception as e:
        print(json.dumps({
            "code": 1,
            "errorMsg": f"Similarity calculation error: {str(e)}",
            "score": 0.0,
            "data": [{"score": 0}]
        }, ensure_ascii=False), flush=True)


def score(predict_zx, predict_zf):
    ZX_score = []
    ZF_score = []
    for i in range(len(predict_zx)):

        zheng_xing_sub = predict_zx[i].strip()
        zheng_xing_gro = ZX[i].strip()
        zhi_fa_sub = predict_zf[i].strip()
        zhi_fa_gro = ZF[i].strip()

        sim_x = calculate_similarity(zheng_xing_sub, zheng_xing_gro)
        sim_f = calculate_similarity(zhi_fa_sub, zhi_fa_gro)

        ZX_score.append(sim_x)
        ZF_score.append(sim_f)

    zx_mean = np.mean(ZX_score) if ZX_score else 0.0
    zf_mean = np.mean(ZF_score) if ZF_score else 0.0

    final_score = ((zx_mean + zf_mean) / 2) * 100  # 百分制
    return final_score


final_score = score(predict_zx, predict_zf)
print(final_score)

