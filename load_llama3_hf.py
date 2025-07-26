from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, LlavaNextConfig, AutoConfig
import torch
import requests
from PIL import Image

# config = LlavaNextConfig.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
# config.text_config = AutoConfig.from_pretrained("unsloth/llama-3-8b-Instruct")
config = LlavaNextConfig.from_pretrained("/data/tangwenyue/Models/llava-v1.6-mistral-7b-hf")
config.text_config = AutoConfig.from_pretrained("/data/tangwenyue/Models/llama-3-8b-Instruct")

# 通过 safe_open 从多个 safetensors 文件中加载模型权重，并将文件权重存储到字典 sd 中
from safetensors import safe_open
sd = {}
for i in range(1, 5):
    # models/model-0000{i}-of-00004.safetensors
    with safe_open(f"models/model-0000{i}-of-00004.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            sd[key] = f.get_tensor(key)

model = LlavaNextForConditionalGeneration(config)

# 重命名并整理权重
keys = list(sd.keys())
for key in keys:
    if 'mm_projector' not in key and 'vision_tower' not in key:
        sd['language_model.' + key] = sd[key] # 语言模块重命名加上 'l_m' 前缀
        del sd[key]
keys = list(sd.keys())
for key in keys:
    if 'vision_tower' in key: # 去掉 'vision_tower' 权重 'model.vision_tower.' 的前缀
        sd[key.replace('model.vision_tower.', '')] = sd[key]
        del sd[key]
# 将权重赋值给新的键，并删除原始的权重项
sd['multi_modal_projector.linear_1.weight'] = sd['model.mm_projector.0.weight']
sd['multi_modal_projector.linear_2.weight'] = sd['model.mm_projector.2.weight']
sd['multi_modal_projector.linear_1.bias']   = sd['model.mm_projector.0.bias']
sd['multi_modal_projector.linear_2.bias']   = sd['model.mm_projector.2.bias']
del sd['model.mm_projector.0.weight']
del sd['model.mm_projector.2.weight']
del sd['model.mm_projector.0.bias']
del sd['model.mm_projector.2.bias']
sd['image_newline'] = sd['language_model.model.image_newline']
del sd['language_model.model.image_newline']

# 将处理后的权重字典加载到模型中
model.load_state_dict(sd)
model.save_pretrained('models/llava-llama-3-8b')
# save language model for training 单独保存 language model
model.language_model.save_pretrained('models/llava-llama-3-8b-llm')
