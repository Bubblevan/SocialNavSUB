import os
import torch
# import decord
# from decord import VideoReader, cpu
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------- 核心配置 --------------------------
# 本地模型路径
MODEL_PATH = r"D:\MyLab\SocialNavSUB\checkpoints\llava-next-video-7b"
# 输入文件路径（支持图片：jpg/png/bmp；视频：mp4/avi/mov）
INPUT_PATH = r"D:\MyLab\SocialNavSUB\data\front_view\33_Spot_45_228\229.jpg"  # 替换为图片/视频路径
# 推理设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 视频帧采样数（图片会自动忽略此参数，固定为1帧）
NUM_FRAMES = 8

# -------------------------- 工具函数 --------------------------
# def extract_video_frames(video_path, num_frames=8):
#     """提取视频帧"""
#     vr = VideoReader(video_path, ctx=cpu(0))
#     total_frames = len(vr)
#     frame_indices = [i * total_frames // num_frames for i in range(num_frames)]
#     frames = [Image.fromarray(vr[i].asnumpy()) for i in frame_indices]
#     return frames

def load_image(image_path):
    """加载单张图片并转为模型可接受的格式"""
    image = Image.open(image_path).convert("RGB")  # 确保RGB格式
    return [image]  # 封装为列表（和视频帧格式统一）

def get_input_frames(input_path):
    """自动识别输入类型，返回统一格式的帧列表"""
    # 判断文件类型
    ext = os.path.splitext(input_path)[-1].lower()
    # 图片格式
    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"图片文件不存在：{input_path}")
        print(f"检测到图片输入：{input_path}")
        return load_image(input_path)
    # 视频格式
    elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"视频文件不存在：{input_path}")
        print(f"检测到视频输入：{input_path}")
        # return extract_video_frames(input_path, NUM_FRAMES)
        return load_image(input_path)
    else:
        raise ValueError(f"不支持的文件格式：{ext}，仅支持图片(jpg/png/bmp)和视频(mp4/avi/mov)")

# -------------------------- 加载模型和Tokenizer --------------------------
print("开始加载模型和Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    padding_side="right"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
).to(DEVICE).eval()

print("模型和Tokenizer加载完成！")

# -------------------------- 多模态推理 --------------------------
try:
    # 1. 获取统一格式的帧/图片列表
    input_frames = get_input_frames(INPUT_PATH)
    print(f"成功加载 {len(input_frames)} 帧输入")
    
    # 2. 构建Prompt（通用格式，图片/视频都适用）
    prompt = "请详细描述这张图片/视频的内容。"  # 通用提问，可根据需求修改
    input_prompt = f"<image>\n{prompt}"
    
    # 3. 预处理输入
    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)
    
    # 4. 推理
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            images=input_frames,  # 统一传入帧列表（图片是1帧，视频是多帧）
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 5. 输出结果
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(input_prompt, "").strip()
    
    print("\n===== 模型回答 =====")
    print(response)

except Exception as e:
    print(f"推理出错：{str(e)}")
finally:
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
