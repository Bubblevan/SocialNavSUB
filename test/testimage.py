"""
DMXAPI 图片识别示例脚本（修复版）
================================
修复点：
1. 修正 Authorization 授权头格式（添加 Bearer 前缀）
2. 适配 DMXAPI/OpenAI 规范的多模态请求格式（messages + image_url 结构）
3. 修正参数命名（messages 替代 input）
4. 优化异常处理和响应解析逻辑
"""

import base64
import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()  

# ========================================
# 配置参数
# ========================================
API_KEY = os.getenv("api_key")  # 你的 DMXAPI 密钥
IMAGE_PATH = "D:/MyLab/SocialNavSUB/data/front_view/33_Spot_45_228/228.jpg"  # 本地图片路径
API_ENDPOINT = "https://www.dmxapi.cn/v1/chat/completions"  # DMXAPI 标准端点

# ========================================
# 图片编码函数（保留，仅优化注释）
# ========================================
def encode_image_to_base64(filepath):
    """将图片编码为 Base64 数据 URI 格式（适配 OpenAI 接口）"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"图片文件不存在：{filepath}")
    
    ext = os.path.splitext(filepath)[1].lower()
    mime_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    mime_type = mime_map.get(ext, 'image/jpeg')  # 默认用 jpeg 兼容更多场景
    
    with open(filepath, "rb") as f:
        base64_data = base64.b64encode(f.read()).decode('utf-8')
    return f"data:{mime_type};base64,{base64_data}"

# ========================================
# 主程序入口
# ========================================
if __name__ == "__main__":
    print("=" * 60)
    print("DMXAPI 图片识别示例程序（修复版）")
    print("=" * 60)
    
    # ----------------------------------------
    # 步骤 1: 图片编码
    # ----------------------------------------
    try:
        print("[步骤 1/3] 正在编码图片...")
        print(f"图片路径: {IMAGE_PATH}")
        image_data = encode_image_to_base64(IMAGE_PATH)
        print(f"✓ 编码成功，数据长度: {len(image_data)} 字符")
        
        # ----------------------------------------
        # 步骤 2: 构造 API 请求（核心修复）
        # ----------------------------------------
        print("[步骤 2/3] 正在构造 API 请求...")
        
        # 修复1：授权头添加 Bearer 前缀
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"  # 关键修复点
        }
        
        # 修复2：适配 OpenAI 规范的多模态 messages 结构
        payload = {
            "model": "gemini-2.5-flash-lite",
            "messages": [  # 关键修复：用 messages 替代 input
                {
                    "role": "user",
                    "content": [
                        # 文本指令
                        {"type": "text", "text": "详细描述这张图片的内容"},
                        # 图片数据（OpenAI 规范格式）
                        {"type": "image_url", "image_url": {"url": image_data}}
                    ]
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2048
        }
        print("✓ 请求构造完成")
        
        # ----------------------------------------
        # 步骤 3: 发送请求并处理响应
        # ----------------------------------------
        print("\n[步骤 3/3] 正在发送请求到 DMXAPI...")
        response = requests.post(
            API_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=30  # 添加超时保护
        )
        
        # 检查 HTTP 状态码
        response.raise_for_status()
        print("✓ 请求发送成功")
        
        # 解析响应
        result = response.json()
        print("\n" + "=" * 60)
        print("完整 API 响应:")
        print("=" * 60)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 提取核心结果
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            print("\n" + "=" * 60)
            print("📷 图片识别结果")
            print("=" * 60)
            print(content)
            print("=" * 60)
    
    except FileNotFoundError as e:
        print(f"\n❌ 错误：{e}")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ 网络请求失败: {str(e)}")
        # 打印服务器返回的错误详情（关键调试信息）
        if hasattr(e, 'response') and e.response is not None:
            print(f"服务器错误详情: {e.response.text[:500]}")
        print("\n请检查：")
        print("  1. API 密钥是否正确（.env 文件中的 api_key）")
        print("  2. 图片路径是否存在且可访问")
        print("  3. 网络是否能正常访问 https://www.dmxapi.cn")
    except json.JSONDecodeError as e:
        print(f"\n❌ JSON 解析失败: {str(e)}")
        print(f"响应内容: {response.text[:200]}")
    except Exception as e:
        print(f"\n❌ 未知错误: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
    finally:
        print("=" * 60)
        print("程序执行完毕")
        print("=" * 60)