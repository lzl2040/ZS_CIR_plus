import multiprocessing
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from statistics import mean
from typing import List, Tuple

import torch
from tqdm import tqdm

import cv2
import json
import shutil
from statistics import mean, geometric_mean, harmonic_mean
import os
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoModel
from transformers import LlavaNextConfig, LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers.models.llava_next.modeling_llava_next import LlavaNextMultiModalProjector
from peft import PeftModel

from sft_utils import extract_index_features
from sft_datasets import collate_fn, FashionIQDataset, CIRRDataset, CIRCODataset

from accelerate import Accelerator
accelerator = Accelerator()

feature_root = "/mnt/input_zuo/ZS-CIR/plus_version/saves/phi3_index_features"
# feature_root = "phi3_index_features"
os.makedirs(feature_root, exist_ok=True)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

DEBUG = False

llama3_template = '''<|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

'''
llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'

def init_model_and_transform(base_model, lora_path, bf16):
    dtype = torch.bfloat16 if bf16 else torch.float16

    if 'llava' in base_model:
        MODEL_CLASS = LlavaNextForConditionalGeneration
    else:
        MODEL_CLASS = AutoModelForCausalLM

    if base_model == 'e5-v':
        transform = LlavaNextProcessor.from_pretrained("royokong/e5-v")
        # transform = LlavaNextProcessor.from_pretrained("/data/tangwenyue/Models/e5-v")
    elif base_model == 'phi3_vision':
        transform = AutoProcessor.from_pretrained("microsoft/Phi-3-vision-128k-instruct", trust_remote_code=True)
        # transform = AutoProcessor.from_pretrained("/data/tangwenyue/Models/Phi-3-vision-128k-instruct", trust_remote_code=True)
    else:
        transform = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        # transform = LlavaNextProcessor.from_pretrained("/data/tangwenyue/Models/llava-v1.6-mistral-7b-hf")

    if base_model == 'llava_llama3':
        tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct")
        # tokenizer = AutoTokenizer.from_pretrained("/data/tangwenyue/Models/Llama-3-8B-Instruct")
        transform.tokenizer = tokenizer
        transform.tokenizer.add_tokens('<image>')
        transform.tokenizer.pad_token_id = transform.tokenizer.eos_token_id

    transform.tokenizer.padding_side = "left"
    transform.tokenizer.padding = True

    if base_model == 'llava_llama3':
        # model_name = "/data/tangwenyue/code_URM/Code_twy/CIR/models/llava-llama-3-8b"
        model_name = "/data/tangwenyue/Code/ZS-CIR/ZS-CIR-twy/models/llava-llama-3-8b"
    elif base_model == 'llava_v1.6':
        model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
        # model_name = "/data/tangwenyue/Models/llava-v1.6-mistral-7b-hf"
    elif base_model == 'e5-v':
        model_name = "royokong/e5-v"
        # model_name = "/data/tangwenyue/Models/e5-v"
    elif base_model == 'phi3_vision':
        model_name = "microsoft/Phi-3-vision-128k-instruct"

    if lora_path is not None:
        merge_path = 'merged-' + lora_path.replace('/', '-').replace('.', '')
        print(merge_path)
        with accelerator.main_process_first():
            if not os.path.exists(merge_path):
                # model = MODEL_CLASS.from_pretrained(model_name, device_map='cpu')
                # model.language_model = PeftModel.from_pretrained(model.language_model, lora_path).merge_and_unload()
                # model.save_pretrained(merge_path)
                if base_model == 'phi3_vision':
                    model = MODEL_CLASS.from_pretrained(model_name, device_map="cuda",
                                                        trust_remote_code=True,
                                                        torch_dtype=dtype, _attn_implementation='eager')
                    model = PeftModel.from_pretrained(model, lora_path).merge_and_unload()
                    model.save_pretrained(merge_path, safe_serialization=False)
                else:
                    model = MODEL_CLASS.from_pretrained(model_name, device_map='cpu')
                    model.language_model = PeftModel.from_pretrained(model.language_model, lora_path).merge_and_unload()
                    model.save_pretrained(merge_path)

        model_name = merge_path

    print("model_name = ", model_name)
    if base_model == 'phi3_vision':
        model = MODEL_CLASS.from_pretrained(model_name, device_map="cuda", trust_remote_code=True,
                                                     torch_dtype=dtype, _attn_implementation='eager')
    else:
        model = MODEL_CLASS.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True)
    if MODEL_TYPE == 'llava_llama3':
        model.config.image_token_index = 128256
    elif MODEL_TYPE == 'llava_v1.6':
        model.config.image_token_index = 32000

    return model, transform

# 将不同数据集 data 的评估指标 metrics 输出到文件中
def log_to_file(data, metrics, checkpoint_name, file_path, fiq_data_type=None):
    os.makedirs(file_path, exist_ok=True)
    if data == 'fashioniq':
        assert len(metrics) == 2
        r_at_10, r_at_50 = metrics
        output = f"{data} {fiq_data_type}: R@10: {r_at_10:.4f} R@50: {r_at_50:.4f}"
    elif data == 'cirr':
        assert len(metrics) == 7
        r_at_1, r_at_5, r_at_10, r_at_50, rs_at_1, rs_at_2, rs_at_3 = metrics
        output = f"{data}:  R@1: {r_at_1:.4f} R@5: {r_at_5:.4f} R@10: {r_at_10:.4f} R@50: {r_at_50:.4f} Rs@1: {rs_at_1:.4f} Rs@2: {rs_at_2:.4f} Rs@3: {rs_at_3:.4f}"
    elif data == 'circo':
        assert len(metrics) == 4
        map_at_5, map_at_10, map_at_25, map_at_50 = metrics
        output = f"{data}: mAP@5: {map_at_5:.4f} mAP@10: {map_at_10:.4f} mAP@25: {map_at_25:.4f} mAP@50: {map_at_50:.4f}"
    elif data == 'genecis':
        assert len(metrics) == 3
        r_at_1, r_at_2, r_at_3 = metrics
        output = f"{data}:  R@1: {r_at_1:.4f} R@2: {r_at_2:.4f} R@3: {r_at_3:.4f}"

    if checkpoint_name is not None:
        save_path = os.path.join(file_path, checkpoint_name)
        with open(save_path, 'a') as f:
            print(output, file=f)

    return output

# 计算检索召回率
@torch.no_grad()
def generate_fiq_val_predictions(model, transform, device, relative_val_dataset, classic_val_dataset,
                                 batch_size, img_prompt, text_img_prompt, phi3, feature_path, shared_concept):
    print('img_prompt = ', img_prompt)
    print('text_img_prompt = ', text_img_prompt)

    bsz = batch_size

    if os.path.exists(feature_path):
        index_data = torch.load(feature_path)
        index_features, index_names = index_data['features'], index_data['names']
    else:
        index_features, index_names = extract_index_features(classic_val_dataset, model, transform, img_prompt, phi3)
        index_data = {'features': index_features, 'names': index_names}
        torch.save(index_data, feature_path)

    relative_val_dataloader = DataLoader(dataset=relative_val_dataset, batch_size=bsz, num_workers=4,
                                         collate_fn=collate_fn, shuffle=False)
    relative_val_dataloader = accelerator.prepare(relative_val_dataloader)

    predicted_features_list = []
    target_names_list = []
    for batch in tqdm(relative_val_dataloader):
        reference_images = batch['reference_image']
        relative_captions = batch['relative_captions']
        target_names = batch['target_name']

        flattened_captions: list = np.array(relative_captions).T.flatten().tolist()
        input_captions = [
            f"{flattened_captions[i].strip('.?, ')} and {flattened_captions[i + 1].strip('.?, ')}" for
            i in range(0, len(flattened_captions), 2)]

        if shared_concept:
            shared_concepts = batch['shared_concept']
            input_texts = [text_img_prompt.replace('<sent>', c).replace('<concept>', s) for c, s in zip(input_captions, shared_concepts)]
        else:
            input_texts = [text_img_prompt.replace('<sent>', c) for c in input_captions]

        # inputs = transform(input_texts, reference_images, return_tensors="pt", padding=True).to(device)
        # with torch.no_grad():
        #     embs = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
        #     embs = F.normalize(embs, dim=-1)
        if phi3:
            with torch.no_grad():
                _embs = []
                for i in range(len(input_texts)):
                    inputs = transform(input_texts[i], [reference_images[i], ], return_tensors="pt", padding=True).to(device)
                    _embs.append(model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :])
                embs = torch.cat(_embs, dim=0)
                embs = F.normalize(embs, dim=-1)
        else:
            inputs = transform(input_texts, reference_images, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                embs = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
                embs = F.normalize(embs, dim=-1)
        embs = accelerator.gather(embs)
        predicted_features_list.append(embs.cpu().float())
        target_names_list.extend(target_names)

    predicted_features = torch.vstack(predicted_features_list)

    assert predicted_features.isnan().sum().item() == 0, 'nan in predicted_features'
    assert index_features.isnan().sum().item() == 0, 'nan in index_features'

    return predicted_features, target_names_list, index_features, index_names

@torch.no_grad()
def fiq_compute_val_metrics(model, transform, device, relative_val_dataset, classic_val_dataset,
                            batch_size, img_prompt, text_img_prompt, phi3, feature_path, shared_concept):
    # Generate the predicted_features, target_names_list, index_features, index_names_list
    predicted_features, target_names, index_features, index_names = generate_fiq_val_predictions(model, transform, device, relative_val_dataset, classic_val_dataset,
                                                                                                 batch_size, img_prompt, text_img_prompt, phi3, feature_path, shared_concept)
    # Move the features to the device
    index_features = index_features.to(device)
    predicted_features = predicted_features.to(device)

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Check if the target names are in the top 10 and top 50
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    r_at_10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    r_at_50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    metrics = [r_at_10, r_at_50]
    return metrics


@torch.no_grad()
def generate_cirr_val_predictions(model, transform, device, relative_val_dataset, classic_val_dataset,
                                  batch_size, img_prompt, text_img_prompt, phi3, feature_path, shared_concept):
    print(img_prompt)
    print(text_img_prompt)

    bsz = batch_size

    if os.path.exists(feature_path):
        index_data = torch.load(feature_path)
        index_features, index_names = index_data['features'], index_data['names']
    else:
        index_features, index_names = extract_index_features(classic_val_dataset, model, transform, img_prompt, phi3)
        index_data = {'features': index_features, 'names': index_names}
        torch.save(index_data, feature_path)

    relative_val_dataloader = DataLoader(dataset=relative_val_dataset, batch_size=bsz, num_workers=4,
                                         collate_fn=collate_fn, shuffle=False)
    relative_val_dataloader = accelerator.prepare(relative_val_dataloader)
    print("len(relative_val_dataset) = ", len(relative_val_dataset))

    predicted_features_list = []
    reference_names_list = []
    target_names_list = []
    group_members_list = []
    for batch in tqdm(relative_val_dataloader):
        target_names = batch['target_name']
        reference_names = batch['reference_name']

        reference_images = batch['reference_image']
        relative_captions = batch['relative_caption']
        shared_concepts = batch['shared_concept']

        group_members = batch['group_members']
        group_members = np.array(group_members).tolist()

        if shared_concept:
            input_texts = [text_img_prompt.replace('<sent>', c).replace('<concept>', s) for c, s in zip(relative_captions, shared_concepts)]
        else:
            input_texts = [text_img_prompt.replace('<sent>', c) for c in relative_captions]

        try:
            if phi3:
                with torch.no_grad():
                    _embs = []
                    for i in range(len(input_texts)):
                        inputs = transform(input_texts[i], [reference_images[i], ], return_tensors="pt",
                                           padding=True).to(device)
                        _embs.append(
                            model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :])
                    embs = torch.cat(_embs, dim=0)
                    embs = F.normalize(embs, dim=-1)
            else:
                inputs = transform(input_texts, reference_images, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    embs = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
                    embs = F.normalize(embs, dim=-1)
            # inputs = transform(input_texts, reference_images, return_tensors="pt", padding=True).to(device)
            # with torch.no_grad():
            #     embs = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
            #     embs = F.normalize(embs, dim=-1)
        except Exception as e:
            print(f"[Warning] image names {reference_names} failed to process ：{e}")

        embs = accelerator.gather(embs)
        predicted_features_list.append(embs.cpu().float())
        reference_names_list.extend(reference_names)
        target_names_list.extend(target_names)
        group_members_list.extend(group_members)

    predicted_features = torch.vstack(predicted_features_list)

    assert predicted_features.isnan().sum().item() == 0, 'nan in predicted_features'
    assert index_features.isnan().sum().item() == 0, 'nan in index_features'

    return predicted_features, reference_names_list, target_names_list, group_members_list, index_features, index_names


@torch.no_grad()
def cirr_compute_val_metrics(model, transform, device, relative_val_dataset, classic_val_dataset,
                             batch_size, img_prompt, text_img_prompt, phi3, feature_path, shared_concept):
    # Generate the predicted_features, target_names_list, index_features, index_names_list
    predicted_features, reference_names, target_names, group_members, index_features, index_names = generate_cirr_val_predictions(
                                                                                                model, transform, device, relative_val_dataset, classic_val_dataset,
                                                                                                batch_size, img_prompt, text_img_prompt, phi3, feature_path, shared_concept)

    # Move the features to the device
    index_features = index_features.to(device)
    predicted_features = predicted_features.to(device)

    # Compute the distance and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))

    # Compute the subset predictions and ground-truth labels
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    r_at_1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    r_at_5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    r_at_10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    r_at_50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    rs_at_1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    rs_at_2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    rs_at_3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    metrics = [r_at_1, r_at_5, r_at_10, r_at_50, rs_at_1, rs_at_2, rs_at_3]

    return metrics


@torch.no_grad()
def generate_circo_val_predictions(model, transform, device, relative_val_dataset, classic_val_dataset,
                                  batch_size, img_prompt, text_img_prompt, phi3, feature_path, shared_concept):
    print(img_prompt)
    print(text_img_prompt)

    bsz = batch_size

    if os.path.exists(feature_path):
        index_data = torch.load(feature_path)
        index_features, index_names = index_data['features'], index_data['names']
    else:
        index_features, index_names = extract_index_features(classic_val_dataset, model, transform, img_prompt, phi3)
        index_data = {'features': index_features, 'names': index_names}
        torch.save(index_data, feature_path)

    relative_val_dataloader = DataLoader(dataset=relative_val_dataset, batch_size=bsz, num_workers=4,
                                         collate_fn=collate_fn, shuffle=False)
    relative_val_dataloader = accelerator.prepare(relative_val_dataloader)
    print("len(relative_val_dataset) = ", len(relative_val_dataset))

    predicted_features_list = []
    target_names_list = []
    gts_img_ids_list = []

    # Compute the features
    for batch in tqdm(relative_val_dataloader):
        reference_images = batch['reference_image']
        target_names = batch['target_name']
        relative_captions = batch['relative_caption']
        shared_concepts = batch['shared_concept']
        # shared_concepts = batch['new_shared_concept']

        gt_img_ids = batch['gt_img_ids']
        gt_img_ids = np.array(gt_img_ids).tolist()
        # gt_img_ids = np.array(gt_img_ids).T.tolist()

        if shared_concept:
            input_texts = [text_img_prompt.replace('<sent>', c).replace('<concept>', s) for c, s in zip(relative_captions, shared_concepts)]
        else:
            input_texts = [text_img_prompt.replace('<sent>', c) for c in relative_captions]

        if phi3:
            with torch.no_grad():
                _embs = []
                for i in range(len(input_texts)):
                    inputs = transform(input_texts[i], [reference_images[i], ], return_tensors="pt", padding=True).to(device)
                    _embs.append(model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :])
                embs = torch.cat(_embs, dim=0)
                embs = F.normalize(embs, dim=-1)
        else:
            inputs = transform(input_texts, reference_images, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                embs = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
                embs = F.normalize(embs, dim=-1)
        embs = accelerator.gather(embs)
        predicted_features_list.append(embs.cpu().float())
        target_names_list.extend(target_names)
        gts_img_ids_list.extend(gt_img_ids)

    predicted_features = torch.vstack(predicted_features_list)

    assert predicted_features.isnan().sum().item() == 0, 'nan in predicted_features'
    assert index_features.isnan().sum().item() == 0, 'nan in index_features'

    return predicted_features, target_names_list, gts_img_ids_list, index_features, index_names


@torch.no_grad()
def circo_compute_val_metrics(model, transform, device, relative_val_dataset, classic_val_dataset,
                             batch_size, img_prompt, text_img_prompt, phi3, feature_path, shared_concept):
    # Generate the predicted_features, target_names_list, index_features, index_names_list
    predicted_features, target_names, gts_img_ids, index_features, index_names = generate_circo_val_predictions(
                                                                                                model, transform, device, relative_val_dataset, classic_val_dataset,
                                                                                                batch_size, img_prompt, text_img_prompt, phi3, feature_path, shared_concept)
    # Move the features to the device
    index_features = index_features.to(device)
    predicted_features = predicted_features.to(device)

    # Normalize the features
    index_features = F.normalize(index_features.float())

    ap_at5 = []
    ap_at10 = []
    ap_at25 = []
    ap_at50 = []

    recall_at5 = []
    recall_at10 = []
    recall_at25 = []
    recall_at50 = []

    for predicted_feature, target_name, gt_img_ids in tqdm(zip(predicted_features, target_names, gts_img_ids)):
        gt_img_ids = np.array(gt_img_ids)[
            np.array(gt_img_ids) != '']  # remove trailing empty strings added for collate_fn
        similarity = predicted_feature @ index_features.T
        sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
        sorted_index_names = np.array(index_names)[sorted_indices]
        map_labels = torch.tensor(np.isin(sorted_index_names, gt_img_ids), dtype=torch.uint8)
        precisions = torch.cumsum(map_labels, dim=0) * map_labels  # Consider only positions corresponding to GTs
        precisions = precisions / torch.arange(1, map_labels.shape[0] + 1)  # Compute precision for each position

        ap_at5.append(float(torch.sum(precisions[:5]) / min(len(gt_img_ids), 5)))
        ap_at10.append(float(torch.sum(precisions[:10]) / min(len(gt_img_ids), 10)))
        ap_at25.append(float(torch.sum(precisions[:25]) / min(len(gt_img_ids), 25)))
        ap_at50.append(float(torch.sum(precisions[:50]) / min(len(gt_img_ids), 50)))

        assert target_name == gt_img_ids[0], f"Target name not in GTs {target_name} {gt_img_ids}"
        single_gt_labels = torch.tensor(sorted_index_names == target_name)
        recall_at5.append(float(torch.sum(single_gt_labels[:5])))
        recall_at10.append(float(torch.sum(single_gt_labels[:10])))
        recall_at25.append(float(torch.sum(single_gt_labels[:25])))
        recall_at50.append(float(torch.sum(single_gt_labels[:50])))

    map_at5 = np.mean(ap_at5) * 100
    map_at10 = np.mean(ap_at10) * 100
    map_at25 = np.mean(ap_at25) * 100
    map_at50 = np.mean(ap_at50) * 100
    recall_at5 = np.mean(recall_at5) * 100
    recall_at10 = np.mean(recall_at10) * 100
    recall_at25 = np.mean(recall_at25) * 100
    recall_at50 = np.mean(recall_at50) * 100

    metrics = [map_at5, map_at10, map_at25, map_at50]

    return metrics


def main(
        base_model: str = "llava_llama3",
        lora_path: str = None,
        file_path: str = None,
        batch_size: int = 1,
        bf16: bool = False,
        fp32: bool = False,
        data: str = None,
        debug: bool = False,
        llava: bool = False,
        phi3: bool = False,
        llava_llama3: bool = False,
        llava_mistral: bool = False,
        name: str = None,
        prompt_type: str="cot",
        feature_path: str = None,
        shared_concept: bool=False,
):
    global DEBUG, MODEL_TYPE
    DEBUG = debug
    if lora_path != None:
        ckt_id = lora_path.split("/")[-1].split("-")[1]
        train_prompt_type = lora_path.split("/")[-2].split("-")[-1]
    else:
        ckt_id = -1
        train_prompt_type = "none"
    print(f"Use checkpoint {ckt_id} to eval, use train prompt type {train_prompt_type}")

    if phi3:
        MODEL_TYPE = 'phi3_vision'
    elif llava_llama3:
        MODEL_TYPE = 'llava_llama3'
    elif llava_mistral:
        MODEL_TYPE = 'llava_v1.6'
    # assert MODEL_TYPE in ['llava_v1.6', 'llava_llama3', 'phi3']

    # set NCCL_DEBUG (NVIDIA Collective Communication Library)
    if os.environ.get("NCCL_DEBUG", None) is None:
        os.environ["NCCL_DEBUG"] = "ERROR"
    device = accelerator.device

    model, transform = init_model_and_transform(base_model, lora_path, bf16)
    model.to(device)

    def process_prompt(img_prompt, text_img_prompt):
        if llava_llama3:
            img_prompt = img_prompt.replace('[INST] ', '').replace(' [/INST]', '')
            text_img_prompt = text_img_prompt.replace('[INST] ', '').replace(' [/INST]', '')
            # llama3_template
            img_prompt = llama3_template.format(img_prompt)
            text_img_prompt = llama3_template.format(text_img_prompt)
        elif phi3:
            img_prompt = img_prompt.replace('[INST] ', '').replace(' [/INST]', '').replace('<image>', '<|image_1|>')
            text_img_prompt = text_img_prompt.replace('[INST] ', '').replace(' [/INST]', '').replace('<image>', '<|image_1|>')
            # phi3_template
            img_prompt = '<|user|>\n{} <|end|>\n<|assistant|>\n'.format(img_prompt)
            text_img_prompt = '<|user|>\n{} <|end|>\n<|assistant|>\n'.format(text_img_prompt)
        else:
            return img_prompt, text_img_prompt

        return img_prompt, text_img_prompt

    # datasets = ['circo']
    # datasets = ['fashioniq dress', 'fashioniq shirt', 'fashioniq toptee', 'cirr']
    datasets = ['fashioniq dress', 'fashioniq shirt', 'fashioniq toptee']
    # datasets = ['cirr']
    if data:
        datasets = data.split(',')

    all_results = []
    for data in datasets:
        if 'fashioniq' in data:
            data, fiq_data_type = data.split(' ')
            assert fiq_data_type in ['dress', 'shirt', 'toptee']
            fiq_data_name = fiq_data_type
            if fiq_data_type == 'toptee':
                fiq_data_name = 'shirt'

            if shared_concept:
                if prompt_type == 'org':
                    # Original prompt with shared concept (summarize word)
                    img_prompt = f"[INST] <image> Describe this {fiq_data_name} in one word based on its style: [/INST]"
                    text_img_prompt = f"[INST] <image> change the style of this {fiq_data_name} to <sent>,  " \
                                      f"and the shared concept is \"<concept>\", and describe this modified {fiq_data_name} in one word based on its style: [/INST]"
                elif prompt_type == 'cot':
                    # Pretended CoT prompt with shared concept (summarize word)
                    img_prompt = f"[INST] <image>\n After thinking step by step, describe this {fiq_data_name} in one word based on its style: [/INST]"
                    text_img_prompt = f"[INST] <image>\n After thinking step by step, change the style of this {fiq_data_name} to <sent>, " \
                                      f"and the shared concept is \"<concept>\", and describe this modified {fiq_data_name} in one word based on its style: [/INST]"
                elif prompt_type == 'ke':
                    # Knowledge enhanced prompt with shared concept (summarize word)
                    img_prompt = f"[INST] <image>\n The essence of this {fiq_data_name} is often captured by its main objects and actions, while additional details provide context. " \
                                 f"With this in mind, describe this {fiq_data_name} in one word based on its style: [/INST]"
                    text_img_prompt = f"[INST] <image>\n The essence of this {fiq_data_name} is often captured by its main objects and actions, while additional details provide context. " \
                                      f"With this in mind, change the style of this {fiq_data_name} to \"<sent>\", and the shared concept is \"<concept>\", and describe this modified {fiq_data_name} in one word based on its style: [/INST]"
            else:
                if prompt_type == 'org':
                    # Original prompt (summarize word)
                    img_prompt = f"[INST] <image> Describe this {fiq_data_name} in one word based on its style: [/INST]"
                    text_img_prompt = f"[INST] <image> change the style of this {fiq_data_name} to <sent>\n " \
                                      f"Describe this modified {fiq_data_name} in one word based on its style: [/INST]"
                elif prompt_type == 'cot':
                    # Pretended CoT prompt (summarize word)
                    img_prompt = f"[INST] <image>\n After thinking step by step, describe this {fiq_data_name} in one word based on its style: [/INST]"
                    text_img_prompt = f"[INST] <image>\n After thinking step by step, change the style of this {fiq_data_name} to <sent>\n" \
                                      f" Describe this modified {fiq_data_name} in one word based on its style: [/INST]"
                elif prompt_type == 'ke':
                    # Knowledge enhanced prompt (summarize word)
                    img_prompt = f"[INST] <image>\n The essence of this {fiq_data_name} is often captured by its main objects and actions, while additional details provide context. " \
                                 f"With this in mind, describe this {fiq_data_name} in one word based on its style: [/INST]"
                    text_img_prompt = f"[INST] <image>\n The essence of this {fiq_data_name} is often captured by its main objects and actions, while additional details provide context. " \
                                      f"With this in mind, change the style of this {fiq_data_name} to \"<sent>\", and describe this modified {fiq_data_name} in one word based on its style: [/INST]"

            img_prompt, text_img_prompt = process_prompt(img_prompt, text_img_prompt)

            relative_val_dataset = FashionIQDataset('val', [fiq_data_type], 'relative')
            classic_val_dataset = FashionIQDataset('val', [fiq_data_type], 'classic')

            feature_path = f'{feature_root}/train_{train_prompt_type}_fiq_{fiq_data_type}_index_data_{ckt_id}_eval_{prompt_type}.pt'
            metrics = fiq_compute_val_metrics(model, transform, device, relative_val_dataset, classic_val_dataset, batch_size, img_prompt, text_img_prompt, phi3, feature_path, shared_concept)
            print(metrics)

        # Validate on CIRR dataset
        elif data == 'cirr':
            if shared_concept:
                if prompt_type == 'org':
                    # Original prompt with shared concept (summarize word)
                    img_prompt = "[INST] <image> Describe this image in one word: [/INST]"
                    text_img_prompt = "[INST] <image>Modify this image with \"<sent>\", and the shared concept is \"<concept>\", and describe modified image in one word: [/INST]"
                elif prompt_type == 'cot':
                    # Pretended CoT prompt with shared concept (summarize word)
                    img_prompt = "[INST] <image>\n After thinking step by step, describe this image in one word: [/INST]"
                    text_img_prompt = "[INST] <image>\n After thinking step by step, modify this image with \"<sent>\", and the shared concept is \"<concept>\", and describe the modified image in one word: [/INST]"
                elif prompt_type == 'ke':
                    # Knowledge enhanced prompt with shared concept (summarize word)
                    img_prompt = "[INST] <image>\n The essence of image is often captured by its main objects and actions, while additional details provide context. " \
                                 "With this in mind, describe this image in one word: [/INST]"
                    text_img_prompt = "[INST] <image>\n The essence of this image is often captured by its main objects and actions, while additional details provide context. " \
                                      "With this in mind, modify this image with \"<sent>\", and the shared concept is \"<concept>\", and describe the modified image in one word: [/INST]"
            else:
                if prompt_type == 'org':
                    # Original prompt (summarize word)
                    img_prompt = "[INST] <image> Describe this image in one word: [/INST]"
                    text_img_prompt = "[INST] <image>Modify this image with \"<sent>\", describe modified image in one word: [/INST]"
                elif prompt_type == 'cot':
                    # Pretended CoT prompt (summarize word)
                    img_prompt = "[INST] <image>\n After thinking step by step, describe this image in one word: [/INST]"
                    text_img_prompt = "[INST] <image>\n After thinking step by step, modify this image with \"<sent>\", describe the modified image in one word: [/INST]"
                elif prompt_type == 'ke':
                    # Knowledge enhanced prompt (summarize word)
                    img_prompt = "[INST] <image>\n The essence of image is often captured by its main objects and actions, while additional details provide context. " \
                                 "With this in mind, describe this image in one word: [/INST]"
                    text_img_prompt = "[INST] <image>\n The essence of this image is often captured by its main objects and actions, while additional details provide context. " \
                                      "With this in mind, modify this image with \"<sent>\", and describe the modified image in one word: [/INST]"

            feature_path = f"{feature_root}/train_{train_prompt_type}_cirr_index_data_{ckt_id}_eval_{prompt_type}.pt"
            img_prompt, text_img_prompt = process_prompt(img_prompt, text_img_prompt)

            relative_val_dataset = CIRRDataset('val', 'relative')
            classic_val_dataset = CIRRDataset('val', 'classic')

            metrics = cirr_compute_val_metrics(model, transform, device, relative_val_dataset, classic_val_dataset, batch_size, img_prompt, text_img_prompt, phi3, feature_path, shared_concept)
            print(metrics)

        # Validate on CIRCO dataset
        elif data == 'circo':
            if shared_concept:
                if prompt_type == 'org':
                    # Original prompt (summarize word) with shared concepts
                    img_prompt = "[INST] <image>\n Describe this image in one word: [/INST]"
                    text_img_prompt = "[INST] <image>Modify this image with \"<sent>\", and the shared concept is \"<concept>\", describe modified image in one word: [/INST]"
                elif prompt_type == 'cot':
                    # Pretended CoT prompt (summarize word) with shared concepts
                    img_prompt = "[INST] <image>\n After thinking step by step, describe this image in one word: [/INST]"
                    text_img_prompt = "[INST] <image>\n After thinking step by step, modify this image with \"<sent>\", and the shared concept is \"<concept>\", describe modified image in one word: [/INST]"
                elif prompt_type == 'ke':
                    # Knowledge enhanced prompt (summarize word)
                    img_prompt = "[INST] <image>\n The essence of image is often captured by its main objects and actions, while additional details provide context. " \
                                 "With this in mind, describe this image in one word: [/INST]"
                    text_img_prompt = "[INST] <image>\n The essence of this image is often captured by its main objects and actions, while additional details provide context. " \
                                      "With this in mind, modify this image with \"<sent>\", and the shared concept is \"<concept>\", and describe the modified image in one word: [/INST]"
            else:
                if prompt_type == 'org':
                    # Original prompt (summarize word)
                    img_prompt = "[INST] <image>\n Describe this image in one word: [/INST]"
                    text_img_prompt = "[INST] <image>Modify this image with \"<sent>\", describe modified image in one word: [/INST]"
                elif prompt_type == 'cot':
                    # Pretended CoT prompt (summarize word)
                    img_prompt = "[INST] <image>\n After thinking step by step, describe this image in one word: [/INST]"
                    text_img_prompt = "[INST] <image>\n After thinking step by step, modify this image with \"<sent>\", describe the modified image in one word: [/INST]"
                elif prompt_type == 'ke':
                    # Knowledge enhanced prompt (summarize word)
                    img_prompt = "[INST] <image>\n The essence of image is often captured by its main objects and actions, while additional details provide context. " \
                                 "With this in mind, describe this image in one word: [/INST]"
                    text_img_prompt = "[INST] <image>\n The essence of this image is often captured by its main objects and actions, while additional details provide context. " \
                                      "With this in mind, modify this image with \"<sent>\", and describe the modified image in one word: [/INST]"

            img_prompt, text_img_prompt = process_prompt(img_prompt, text_img_prompt)

            relative_val_dataset = CIRCODataset('val', 'relative')
            classic_val_dataset = CIRCODataset('val', 'classic')

            feature_path = f'{feature_root}/train_{train_prompt_type}_circo_index_data_{ckt_id}_eval_{prompt_type}.pt'
            # feature_path = f'{feature_path}/circo_index_data_org.pt'
            metrics = circo_compute_val_metrics(model, transform, device, relative_val_dataset, classic_val_dataset,
                                               batch_size, img_prompt, text_img_prompt, phi3, feature_path, shared_concept)
            print(metrics)

        if accelerator.is_main_process:
            if lora_path is not None:
                train_model_name = lora_path.split("/")[-2]
                ckt_name = lora_path.split("/")[-1]
                # checkpoint_name = lora_path.replace('/', '_') + '.txt'
                checkpoint_name = f"{train_model_name}_{ckt_name}_eval_{prompt_type}.txt"
            elif name is not None:
                checkpoint_name = name if name.endswith('.txt') else name + '.txt'
            else:
                checkpoint_name = None

            if data == 'cirr':
                all_results.append(log_to_file(data, metrics, checkpoint_name, file_path))
            elif data == 'fashioniq':
                all_results.append(log_to_file(data, metrics, checkpoint_name, file_path, fiq_data_type=fiq_data_type))
            elif data == 'circo':
                all_results.append(log_to_file(data, metrics, checkpoint_name, file_path))

    if accelerator.is_main_process:
        print('\n'.join(all_results))

if __name__ == '__main__':
    from fire import Fire

    Fire(main)
