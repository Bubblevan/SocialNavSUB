import requests
import os
from openai import OpenAI
from dotenv import load_dotenv

# 加载.env中的api_key
load_dotenv()  

"""
╔═══════════════════════════════════════════════════════════════╗
║               DMXAPI 调用Gemini模型（OpenAI原生接口）          ║
╚═══════════════════════════════════════════════════════════════╝
📝 核心修改：
1. 改用OpenAI原生chat/completions接口（适配Gemini类模型）
2. 参数从input改为messages（数组格式，符合OpenAI规范）
3. 流式响应处理增加边界判断，解决IndexError问题
"""

# ═══════════════════════════════════════════════════════════════
# 🔑 步骤1: 初始化 DMXAPI 客户端（不变）
# ═══════════════════════════════════════════════════════════════
client = OpenAI(
    api_key=os.getenv("api_key"),  # 确保.env里有api_key=你的DMXAPI密钥
    base_url="https://www.dmxapi.cn/v1",
)

# ═══════════════════════════════════════════════════════════════
# 💬 步骤2: 调用OpenAI原生chat/completions接口（适配Gemini）
# ═══════════════════════════════════════════════════════════════
try:
    # 注意：这里用client.chat.completions.create（原生接口），而非responses.create
    response = client.chat.completions.create(
        model="gemini-2.5-flash-lite",  # 若仍报错，换成gpt-4.1-nano（DMXAPI必支持）
        messages=[  # 核心：改用messages数组，而非input参数
            {"role": "user", "content": "你好!"}  # user是用户角色，content是提问内容
        ],
        stream=True,  # 开启流式输出
        temperature=0.7,  # 可选：控制随机性
        max_tokens=2048,  # 可选：最大输出长度（替代原max_output_tokens）
    )

    # ═══════════════════════════════════════════════════════════════
    # 📤 步骤3: 处理原生接口的流式响应（增加边界判断，解决IndexError）
    # ═══════════════════════════════════════════════════════════════
    print("🤖 模型响应：", end="", flush=True)
    for chunk in response:
        # 核心修复：多层判断，避免IndexError
        # 1. 判断choices是否存在且非空；2. 判断delta是否存在；3. 判断content是否存在
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                print(delta.content, end="", flush=True)

except KeyboardInterrupt:
    print("\n\n⚠️ 用户中断了请求")
except Exception as e:
    print(f"\n\n❌ 发生错误: {type(e).__name__} - {e}")
    # 若报错，建议先切换模型为gpt-4.1-nano测试
    print("\n💡 提示：若仍报错，可将model改为gpt-4.1-nano（DMXAPI必支持）")
finally:
    print()  # 最后换行，保证格式整洁