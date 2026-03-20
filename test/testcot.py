# ============================================================
#  GPT-5 深度思考模式调用示例（SDK 版）
#  功能：通过 OpenAI Python SDK 调用 Responses API
# ============================================================

from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()  

# ---------- 初始化客户端 ----------
client = OpenAI(
    api_key=os.getenv("api_key"),
    base_url="https://www.dmxapi.cn/v1"
)

# ---------- 发送请求 ----------
response = client.responses.create(
    model="gpt-5-nano",                        # 使用的模型
    input="一加到十最复杂的方法",  # 用户输入的问题
    reasoning={
        "effort": "low",                  # 思考等级
        "summary": "auto"                 # 思考摘要：auto 自动返回思考过程
    }
)

# ---------- 输出结果 ----------
# 遍历 output 列表，分别提取「思考内容」和「响应结果」
for item in response.output:
    if item.type == "reasoning":
        print("【思考内容】")
        for content in item.summary:
            print(content.text)
        print()
    elif item.type == "message":
        print("【响应结果】")
        for content in item.content:
            print(content.text)