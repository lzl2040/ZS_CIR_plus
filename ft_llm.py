import os
import sys
from typing import List

import fire    # 自动生成命令行界面的 Python 库，它可以将 Python 代码中的类和函数自动转换为命令行工具
import torch
import torch.nn as nn
import bitsandbytes as bnb    # 用于加速深度学习模型训练的 Python 库，提供了一些高效的低精度计算和量化技术
from datasets import load_dataset, load_from_disk    # HF 的库，用于高效加载、处理、共享和管理各种 NLP 和 ML 的数据集
import transformers
from transformers import Trainer
import torch.distributed as dist    # 用于支持分布式训练和多进程通信

from transformers import set_seed
from transformers import LlamaTokenizer, LlamaConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import BitsAndBytesConfig

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    # get_peft_model_state_dict,
    # set_peft_model_state_dict,
)

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass

# from transformers.utils import logging
from transformers.trainer_callback import TrainerCallback
from swanlab.integration.transformers import SwanLabCallback
import swanlab

import logging

logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.ERROR)
# logging.getLogger("transformers.trainer").setLevel(logging.ERROR)

NIL_DATASET = True

llama3_template = '''<|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

'''
llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'


class ForceTqdmUpdateCallback(TrainerCallback):
    # 在训练过程中定制回调函数，在每个训练步骤结束时执行一些自定义操作
    def on_step_end(self, args, state, control, **kwargs):
        # pdsh can't update tqdm, except warning
        if state.is_world_process_zero: # 检查当前进程是否是主进程（通常在分布式训练中，主进程负责更新进度条或者进行日志记录
            if state.global_step % 5 == 0 or state.global_step < 20: # 每 5 步或前 20 步，记录一下
                logger.warning('') # 可以记录更有意义的信息
@dataclass
class DataCollatorForSeq2SeqForNeg:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    # 主要用于处理输入数据，使其适应模型训练或推理
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        # 如果 features[0] 字典包含 "labels"，就提取出每个元素的 labels，否则为 None
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            # 计算最大标签长度，并进行填充
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                # 将 max_label_length 调整为 self.pad_to_multiple_of 的倍数
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            # 设置填充的位置，通常为 left 或 right
            padding_side = self.tokenizer.padding_side
            for feature in features:
                # 计算需要填充的部分，然后用 self.label_pad_token_id 填充列表
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        _features = self.tokenizer.pad(
            {'input_ids': [feature['input_ids'] for feature in features]},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        ) # 对 'input_ids' 进行填充
        _features['attention_mask'] = self.tokenizer.pad(
            {'input_ids': [feature['attention_mask'] for feature in features]},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )['input_ids'] # 填充 'attention_mask'，'input_ids'通常表示填充后的各种数据
        _features['labels'] = self.tokenizer.pad(
            {'input_ids': [feature['labels'] for feature in features]},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )['input_ids'] # 填充 'labels'，'input_ids'通常表示填充后的各种数据
        features = _features

        # prepare decoder_input_ids，对解码器输入 ID
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """
    # 计算两个向量之间的余弦相似度
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

from transformers.trainer_utils import has_length
from transformers.file_utils import is_datasets_available
from transformers.trainer_pt_utils import (
    LengthGroupedSampler,
)
from torch.utils.data import RandomSampler, SequentialSampler, Dataset

class SentembTrainer(Trainer):
    force_tqdm_update = True
    fix_attention_mask = False

    import warnings
    warnings.filterwarnings("ignore", message="Trainer.tokenizer is now deprecated")

    # def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
    #     if self.train_dataset is None or not has_length(self.train_dataset):
    #         return None # 检查训练数据集的有效性
    def _get_train_sampler(self, train_dataset: Optional[Dataset] = None) -> Optional[torch.utils.data.Sampler]:
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None or not has_length(train_dataset):
            return None # 检查训练数据集的有效性
        if self.force_tqdm_update: # 强制更新 tqdm 进度条
            self.add_callback(ForceTqdmUpdateCallback) # 回调

        # Build the sampler.
        if self.args.group_by_length: # 是否根据按长度分组选择采样器
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            # 设置模型输入名称
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler( # 按样本长度分组的采样器
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )
        return RandomSampler(self.train_dataset)

    # 用于计算模型的损失，加入负样本处理、对比学习损失计算等逻辑
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # import pdb; pdb.set_trace()
        # print("num_items_in_batch =", num_items_in_batch)
        if self.is_nli and self.use_neg_sentence: # NLI 自然语言推断任务，并使用了负样本
            input_ids, labels, neg = inputs["input_ids"], inputs["labels"], inputs['attention_mask']
            pad_token_id = self.tokenizer.pad_token_id
            if self.fix_attention_mask:
                labels[labels < 0 ] = pad_token_id
                neg[neg < 0] = pad_token_id
            else:
                labels[labels < 0 ] = 0
                neg[neg < 0] = 0

            # padding tensor length 根据最大长度，对输入进行填充
            mw = max(input_ids.size(1), labels.size(1), neg.size(1))
            pad_size = mw - labels.size(1)
            if pad_size > 0:
                label_pads = torch.zeros(labels.size(0), pad_size).cuda().long()
                label_pads.fill_(pad_token_id)
                labels = torch.cat([label_pads, labels], dim=1)
            pad_size = mw - input_ids.size(1)
            if pad_size > 0:
                input_pads = torch.zeros(input_ids.size(0), pad_size).cuda().long()
                input_pads.fill_(pad_token_id)
                input_ids = torch.cat([input_pads,
                                       input_ids], dim=1)
            pad_size = mw - neg.size(1)
            if pad_size > 0:
                neg_pads = torch.zeros(neg.size(0), pad_size).cuda().long()
                neg_pads.fill_(pad_token_id)
                neg = torch.cat([neg_pads,
                                 neg], dim=1)

            # 将 input_ids, labels, neg 拼接在一起
            inputs['input_ids'] = torch.cat([input_ids, labels, neg], dim=0)
            if self.fix_attention_mask:
                inputs['attention_mask'] = (inputs['input_ids'] != pad_token_id).long()
            else:
                inputs['attention_mask'] = (inputs['input_ids'] > 0).long()
            del inputs['labels']
        elif self.is_nli: # NIL 任务，但没有使用负样本
            input_ids, labels = inputs["input_ids"], inputs["labels"]
            labels[labels < 0 ] = 0
            # padding tensor length
            if input_ids.size(1) > labels.size(1):
                pad_size = input_ids.size(1) - labels.size(1)
                labels = torch.cat([torch.zeros(labels.size(0), pad_size).cuda().long(), labels], dim=1)
            else:
                pad_size = labels.size(1) - input_ids.size(1)
                input_ids = torch.cat([torch.zeros(input_ids.size(0), pad_size).cuda().long(), input_ids], dim=1)
            inputs['input_ids'] = torch.cat([input_ids, labels], dim=0)
            inputs['attention_mask'] = (inputs['input_ids'] > 0).long()
            del inputs['labels']
        else: # 更新 'input_ids'
            inputs['input_ids'] = torch.cat([inputs['input_ids'], inputs['input_ids']], dim=0)
            inputs['attention_mask'] = torch.cat([inputs['attention_mask'], inputs['attention_mask']], dim=0)
            del inputs['labels']

        # import pdb; pdb.set_trace()
        # 选择最后一个时间步的隐藏状态，取最后一个时间步的隐藏状态
        pooler_output = model(output_hidden_states=True, return_dict=True, **inputs).hidden_states[-1][:, -1, :]
        # print(pooler_output)

        if self.use_neg_sentence: # 是否使用负句子
            batch_size = pooler_output.size(0)//3 # 模型的输出包括三部分
            pooler_output = torch.stack([pooler_output[:batch_size],
                                         pooler_output[batch_size:2*batch_size],
                                         pooler_output[2*batch_size:]], dim=1)
            z1, z2, z3 = pooler_output[:,0], pooler_output[:,1], pooler_output[:,2]
        else: # 没有使用负句子，则模型的输出包括两部分
            batch_size = pooler_output.size(0)//2
            pooler_output = torch.stack([pooler_output[:batch_size], pooler_output[batch_size:]], dim=1)
            z1, z2 = pooler_output[:,0], pooler_output[:,1]

        loss_fct = nn.CrossEntropyLoss() # 定义交叉熵函数

        # 定义分布式训练
        if dist.is_initialized():
            if self.use_neg_sentence: # 收集负样本
                z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())] # 收集所有计算节点上的 z3
                dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous()) # 将每个计算节点上的 z3 张量收集到 zs_list
                z3_list[dist.get_rank()] = z3 # 将当前节点的 z3 放到对应位置
                z3 = torch.cat(z3_list, 0) # 将所有计算节点上的 z3 张量拼接成一个完整的张量

            # Dummy vectors for allgather
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            # Allgather，将每个节点的张量聚合成一个更大的张量，但不会保留收集数据的梯度信息，即不可参与反向传播
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())
            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)

        # 计算余弦相似度，检查是否有 sim 属性
        if not hasattr(model, "sim"):
            self.sim = Similarity(temp=0.05)
        cos_sim = self.sim(z1.unsqueeze(1).float(), z2.unsqueeze(0).float())

        # 计算负样本相似度
        if self.use_neg_sentence:
            z1_z3_cos = self.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

        labels = torch.arange(cos_sim.size(0)).long().to(inputs['input_ids'].device)

        # 生成标签
        if self.use_neg_sentence:
            z3_weight = 0
            weights = torch.tensor(
                [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
            ).to(input_ids.device) # 对 z3 相关的部分施加额外的权重
            cos_sim = cos_sim + weights
        # print("num_items_in_batch = ", num_items_in_batch)
        loss = loss_fct(cos_sim, labels) # # 计算交叉熵损失函数
        # print("loss = ", loss)
        return (loss, pooler_output) if return_outputs else loss

def generate_sentemb_prompt(data_point, tokenizer, cutoff_len, template, prefix='input'):
    sp = f's{prefix}'
    if sp not in data_point: # 检查 sp 是否存在于 data_point
        input = tokenizer( # 将文本转换为模型输入格式的 tokenizer 对象
            data_point[prefix], # 需要处理的原始文本数据
            truncation=True,
            max_length=cutoff_len, # 截断文本的最大长度，限制文本的最大 token 长度
            padding=False,
            return_tensors=None,
            add_special_tokens=False, # 添加特殊标记，如 [CLS], [SEP]
        )
        input = tokenizer.decode(input['input_ids']) # 将 token 转换回原始的文本字符串
        data_point[sp] = input # 存储到 data_point[sp]
    else:
        input = data_point[sp]

    # 将_替换为 ，删除 *sep+*，*cls*
    template = template.replace('_', ' ').replace('*sep+*', '')\
                                         .replace('*cls*', '').replace('\\n', '\n')
    # print(template.replace('*sent 0*', input).strip())
    return template.replace('*sent 0*', input).strip() # 找到 *sent 0* 这个占位符，然后将其替换为处理后的文本 input

def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "data/nli_for_simcse.csv",
        output_dir: str = "./lora-alpaca",
        # training hyperparams
        batch_size: int = 256,
        micro_batch_size: int = 64,
        num_epochs: int = 1,
        learning_rate: float = 5e-4, # 5e-4
        cutoff_len: int = 32,
        # lora hyperparams
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
        ],
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve,
        is_sentemb: bool = False,
        mask_embedding_sentence_template: str = None,
        run_name: str = None,
        use_neg_sentence: bool = False,
        load_kbit: int = 4,
        save_steps: int = 100,
        seed: int = 42,
        # Different from Sentemb ft_llm
        deepspeed: str = None,
        logging_steps: int = 10,
        grad_checkpoint: bool = False,
        fix_attention_mask: bool = False,
        set_pad_to_unk: bool = False,
        bf16: bool = False,
        not_eol: bool = False,
        org_attn: bool = False,
):
    # set NCCL_DEBUG
    os.makedirs(output_dir, exist_ok=True) # 确保输出目录存在
    global NIL_DATASET
    NIL_DATASET = True

    group_by_length = False
    train_on_inputs = False
    cutoff_len = 32 # 将文本截断或填充到最大长度 32

    assert load_kbit in [4, 8, 16] # 量化比特数为 4, 8, 16

    run_name = output_dir

    # 在更新模型参数之前，累积多少个小批次的梯度
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "cuda"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    #if ddp and False:
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

        # 分布式训练代码再看一下
        torch.distributed.init_process_group("nccl")
        rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        device_id = rank % torch.cuda.device_count()
        device = torch.device(device_id)
        torch.cuda.set_device(device)

    set_seed(seed)

    config = None

    dtype = torch.float16 if load_kbit == 16 else torch.float32
    if bf16:
        dtype = torch.bfloat16

    print("base_model = ", base_model)
    if 'Phi-3' not in base_model: # 检查 Phi-3 模型是否在 base_model 中
        from accelerate import Accelerator # 该库用于加速训练和推理，可以帮助更高效地管理多设备训练任务
        accelerator = Accelerator() # 创建 Accelerator 实例
        #device = accelerator.device
        with accelerator.main_process_first(): # 确保在分布式环境中，只有主进程会执行该部分代码
            base_llm_model = base_model.split('/')[-1] + '-llm'
            base_llm_model = os.path.join('models', base_llm_model)
            base_llm_model = base_llm_model.strip('-')  # 去除末尾多余的 - 字符，strip('-') 只会删除字符串开头和结尾的指定字符
            if not os.path.exists(base_llm_model):
                from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
                LlavaNextForConditionalGeneration.from_pretrained(
                    base_model,
                    device_map='cpu',
                ).language_model.save_pretrained(base_llm_model)

        if load_kbit == 4: # 加载 4 位量化的模型
            assert load_kbit == 4
            MODEL_CLS = AutoModelForCausalLM
            model = MODEL_CLS.from_pretrained(
                base_llm_model,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                ),
                torch_dtype=torch.bfloat16 if bf16 else torch.float16,
                device_map=device_map,
                attn_implementation='eager' if org_attn else None,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_llm_model,
                load_in_8bit=load_kbit == 8,
                load_in_4bit=load_kbit == 4,
                torch_dtype=torch.bfloat16 if bf16 else torch.float16,
                device_map=device_map,
                attn_implementation='eager' if org_attn else None,
            )
    elif load_kbit == 4: # 'Phi-3' in base_model, 加载 4-bit 量化的模型
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            ),
            config=config,
            torch_dtype=torch.bfloat16 if bf16 else torch.float16,
            _attn_implementation='eager' if 'phi3' in base_model else None,
            trust_remote_code=True,
            device_map=device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_kbit == 8 ,
            torch_dtype=dtype,
            device_map=device_map,
        )


    if 'llama-3' in base_model:
        tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        tokenizer.padding = True
    elif 'llava' in base_model:
        from transformers import LlavaNextProcessor
        # if "llava-v1.6-mistral-7b-hf" in base_model: # "llava-hf/llava-v1.6-mistral-7b-hf"
        #     # bug in new vision of tokenizer
        #     tokenizer = LlavaNextProcessor.from_pretrained(base_model, revision='a1d521368f8d353afa4da2ed2bb1bf646ef1ff5f').tokenizer
        # else:
        #     tokenizer = LlavaNextProcessor.from_pretrained(base_model).tokenizer
        tokenizer = LlavaNextProcessor.from_pretrained(base_model, revision='a1d521368f8d353afa4da2ed2bb1bf646ef1ff5f').tokenizer
    elif 'Phi-3' in base_model:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        tokenizer = processor.tokenizer
        tokenizer.padding_side = "left"
        tokenizer.padding = True
    else:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)

        if tokenizer.bos_token_id == 0:
            # fix bos token id
            tokenizer.bos_token_id = 1
            tokenizer.eos_token = '</s>'

        tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
        tokenizer.padding_side = "left"  # Allow batched inference

    if set_pad_to_unk: # 是否是填充符号的 ID 设置为 unk 未知的 ID
        tokenizer.pad_token_id = tokenizer.unk_token_id

    if 'llama-3' in base_model:
        mask_embedding_sentence_template = llama3_template.format(mask_embedding_sentence_template)

    if not_eol:
        mask_embedding_sentence_template = '*sent_0*'
    print(mask_embedding_sentence_template)

    def tokenize(prompt, add_eos_token=True, label_prompt=None, neg_prompt=None):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        if label_prompt:
            label_result = tokenizer(
                label_prompt,
                padding=False,
                return_tensors=None,
            )
            result["labels"] = label_result["input_ids"]
            if neg_prompt:
                neg_result = tokenizer(
                    neg_prompt,
                    padding=False,
                    return_tensors=None,
                )
                result["attention_mask"] = neg_result["input_ids"]
        else:
            result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        if NIL_DATASET:
            data_point['input'] = data_point['sent0']
            data_point['output'] = data_point['sent1']
            if use_neg_sentence:
                data_point['neg'] = data_point['hard_neg']

        full_prompt = generate_sentemb_prompt(data_point, tokenizer, cutoff_len,
                                              mask_embedding_sentence_template,
                                              prefix='input')
        if NIL_DATASET:
            pos_full_prompt = generate_sentemb_prompt(data_point, tokenizer, cutoff_len,
                                                      mask_embedding_sentence_template,
                                                      prefix='output')
            if use_neg_sentence:
                neg_full_prompt = generate_sentemb_prompt(data_point, tokenizer, cutoff_len,
                                                          mask_embedding_sentence_template,
                                                          prefix="neg")
        # 用于生成并处理数据点的完整文本提示 prompt，然后将其分词并转换为模型训练所需的输入格式
        tokenized_full_prompt = tokenize(full_prompt, False,
                                         label_prompt=None if not NIL_DATASET else pos_full_prompt,
                                         neg_prompt=neg_full_prompt if NIL_DATASET and use_neg_sentence else None)
        if not train_on_inputs and not NIL_DATASET:
            user_prompt = generate_sentemb_prompt({**data_point, "output": ""}, tokenizer, cutoff_len,
                                                  mask_embedding_sentence_template,
                                                  prefix='input')
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        # print("tokenized_full_prompt = ", tokenized_full_prompt)
        return tokenized_full_prompt

    if grad_checkpoint:
        model.enable_input_require_grads()

    if load_kbit == 4:
        if 'Phi-3' in base_model:
            target_modules = [
                [f'model.layers.{i}.mlp.gate_up_proj',
                 f'model.layers.{i}.mlp.down_proj',
                 f'model.layers.{i}.self_attn.o_proj',
                 f'model.layers.{i}.self_attn.qkv_proj' ] for i in range(32)
            ]
            target_modules = sum(target_modules, [])
            print(target_modules)
        else:
            model = prepare_model_for_kbit_training(model)
            # 查找 Linear 层并设置 LoRA 配置
            def find_all_linear_names(model):
                cls = bnb.nn.Linear4bit # 查找所有属于 Linear4bit 类型的模块
                lora_module_names = set()
                for name, module in model.named_modules():
                    if isinstance(module, cls):
                        names = name.split('.')
                        lora_module_names.add(names[0] if len(names) == 1 else names[-1]) # 将其添加到 lora_module_names 集合中
                # 排除 lm_head 层，其在 LoRA 配置中通常不参与微调
                if 'lm_head' in lora_module_names: # needed for 16-bit
                    lora_module_names.remove('lm_head')
                return list(lora_module_names)
            target_modules = find_all_linear_names(model)
            print(target_modules)

        # 配置 LoRA
        config = LoraConfig(
            r=lora_r, # 低秩矩阵的秩
            lora_alpha=lora_alpha, # LoRA 的缩放因子
            target_modules=target_modules, # 包含 LoRA 微调的目标模块名称
            lora_dropout=lora_dropout, # LoRA 层的 dropout 比例
            bias="none",
            task_type="CAUSAL_LM", # 设置任务类型为因果语言建模
        )
        # 将 LoRA 配置应用到模型中，通常用于对模型进行参数高效微调 PEFT
        model = get_peft_model(model, config)

    else:
        if load_kbit == 8:
            model = prepare_model_for_kbit_training(model)
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    # 加载数据集 CSV 和 JSON
    if 'csv' in data_path:
        data = load_dataset("csv", data_files=data_path)
    elif os.path.isdir(data_path):
        data = load_from_disk(data_path)
    else:
        data = load_dataset("json", data_files=data_path)

    # 打印可训练参数
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt, num_proc=25)
    DC_FUN = DataCollatorForSeq2SeqForNeg if NIL_DATASET and use_neg_sentence else transformers.DataCollatorForSeq2Seq
    data_collator = DC_FUN(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        #tokenizer, return_tensors="pt", padding=True
    )

    swanlab_callback = SwanLabCallback(
        project="Phi3-finetune",
        experiment_name="Phi3-qlora-embedding",
        # config=config,
        config={
            "r": lora_r,  # 低秩矩阵的秩
            "lora_alpha": lora_alpha,  # LoRA 的缩放因子
            "lora_dropout": lora_dropout,  # LoRA 层的 dropout 比例
            "task_type": "CAUSAL_LM",
        },
    )

    trainer = SentembTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True if not bf16 else False,
            bf16=bf16,
            logging_steps=logging_steps,
            eval_strategy="no",  # 修复：evaluation_strategy -> eval_strategy
            save_strategy="steps",
            eval_steps=None,
            save_steps=save_steps,
            output_dir=output_dir,
            save_total_limit=100,
            load_best_model_at_end=False,
            #ddp_find_unused_parameters=False if ddp else None,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            run_name=run_name,
            report_to="swanlab",  # None, "none"
            deepspeed=deepspeed,
            gradient_checkpointing=grad_checkpoint,
        ),
        data_collator=data_collator,
        callbacks=[swanlab_callback],
    )
    # trainer.tokenizer = tokenizer
    trainer.processing_class = tokenizer
    trainer.is_nli = NIL_DATASET
    trainer.use_neg_sentence = use_neg_sentence
    trainer.fix_attention_mask = fix_attention_mask
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()

    model.save_pretrained(output_dir)

import warnings
warnings.filterwarnings("ignore", message="Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.")

if __name__ == "__main__":

    fire.Fire(train)
