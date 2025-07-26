import copy
import random
import multiprocessing
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from accelerate import Accelerator
accelerator = Accelerator()

from sft_datasets import collate_fn, CIRRDataset, FashionIQDataset, CIRCODataset
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_index_features(dataset: Union[CIRRDataset, FashionIQDataset], model, transform, img_prompt, phi3) -> \
        Tuple[torch.tensor, List[str]]:
    classic_val_loader = DataLoader(dataset=dataset, batch_size=1, num_workers=2,
                                     pin_memory=True, collate_fn=collate_fn)
    classic_val_loader = accelerator.prepare(classic_val_loader)

    index_features = []
    index_names = []
    if isinstance(dataset, CIRRDataset):
        print(f"extracting CIRR {dataset.split} index features")
    elif isinstance(dataset, FashionIQDataset):
        print(f"extracting FashionIQ {dataset.dress_types} - {dataset.split} index features")
    for batch in tqdm(classic_val_loader):
        images = batch['image']
        names = batch['image_name']

        # input_texts = [img_prompt]*len(images)
        # inputs = transform(input_texts, images, return_tensors="pt", padding=True).to(device)
        # with torch.no_grad():
        #     embs = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
        #     embs = F.normalize(embs, dim=-1)
        #     assert embs.isnan().sum() == 0, 'nan in emb after norm'
        try:
            input_texts = [img_prompt] * len(images)
            if phi3:
                assert len(input_texts) == 1
                input_texts = input_texts[0]
            inputs = transform(input_texts, images, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                embs = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
                embs = F.normalize(embs, dim=-1)
                assert embs.isnan().sum() == 0, 'nan in emb after norm'
        except Exception as e:
            print(f"[Warning] image names {names} failed to process ：{e}")

        embs = accelerator.gather(embs)
        index_features.append(embs.cpu().float())
        index_names.extend(names)
    index_features = torch.vstack(index_features)

    return index_features, index_names


# NV-MM-Embed Model HuggingFace
def extract_index_features_nvmm(dataset: Union[CIRRDataset, FashionIQDataset], model, img_prompt) -> \
        Tuple[torch.tensor, List[str]]:
    classic_val_loader = DataLoader(dataset=dataset, batch_size=1, num_workers=2,
                                     pin_memory=True, collate_fn=collate_fn)
    classic_val_loader = accelerator.prepare(classic_val_loader)

    index_features = []
    index_names = []
    if isinstance(dataset, CIRRDataset):
        print(f"extracting CIRR {dataset.split} index features")
    elif isinstance(dataset, FashionIQDataset):
        print(f"extracting FashionIQ {dataset.dress_types} - {dataset.split} index features")
    for batch in tqdm(classic_val_loader):
        images = batch['image']
        names = batch['image_name']

        inputs = [{'img': img} for img in images]
        assert isinstance(inputs, list), "inputs should be a list of dictionary"

        with torch.no_grad():
            outputs = model.encode(inputs, is_query=True, instruction=img_prompt)
            embs = outputs.hidden_states
            embs = F.normalize(embs, dim=-1)
            assert embs.isnan().sum() == 0, 'nan in emb after norm'
        embs = accelerator.gather(embs)
        index_features.append(embs.cpu().float())
        index_names.extend(names)
    index_features = torch.vstack(index_features)

    return index_features, index_names


# Qwen2_5_VL_HuggingFace
def extract_index_features_qwen(dataset: Union[CIRRDataset, FashionIQDataset, CIRCODataset], model, transform, img_prompt) -> \
        Tuple[torch.tensor, List[str]]:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    
    classic_val_loader = DataLoader(dataset=dataset, batch_size=1, num_workers=2,
                                     pin_memory=True, collate_fn=collate_fn)
    classic_val_loader = accelerator.prepare(classic_val_loader)

    index_features = []
    index_names = []
    if isinstance(dataset, CIRRDataset):
        print(f"extracting CIRR {dataset.split} index features")
    elif isinstance(dataset, FashionIQDataset):
        print(f"extracting FashionIQ {dataset.dress_types} - {dataset.split} index features")
    elif isinstance(dataset, CIRCODataset):
        print(f"extracting CIRCO {dataset.split} index features")
    for batch in tqdm(classic_val_loader):
        images = batch['image']
        names = batch['image_name']

        messages_batch = []
        for img in images:
            prompt = copy.deepcopy(img_prompt)
            # prompt[1]["content"][0]["image"] = img
            prompt[0]["content"][0]["image"] = img
            messages_batch.append(prompt)

        input_texts = transform.apply_chat_template(messages_batch, tokenize=False, add_generation_prompt=True)
        input_images, input_videos = process_vision_info(messages_batch)
        inputs = transform(text=input_texts, images=input_images, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            embs = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
            embs = F.normalize(embs, dim=-1)
            assert embs.isnan().sum() == 0, 'nan in emb after norm'
        embs = accelerator.gather(embs)
        index_features.append(embs.cpu().float())
        index_names.extend(names)
    index_features = torch.vstack(index_features)

    return index_features, index_names


def update_train_running_results(train_running_results: dict, loss: torch.tensor, images_in_batch: int):
    train_running_results['accumulated_train_loss'] += loss.to('cpu',
                                                               non_blocking=True).detach().item() * images_in_batch
    train_running_results["images_in_epoch"] += images_in_batch


def set_train_bar_description(train_bar, epoch: int, num_epochs: int, train_running_results: dict):
    train_bar.set_description(
        desc=f"[{epoch}/{num_epochs}] "
             f"train loss: {train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch']:.3f} "
    )


def update_train_running_results_dict(train_running_results: dict, loss_dict: dict, images_in_batch: int):
    for key in loss_dict.keys():
        if key not in train_running_results:
            train_running_results[key] = 0
        train_running_results[key] += loss_dict[key].to('cpu', non_blocking=True).detach().item() * images_in_batch

    train_running_results["images_in_epoch"] += images_in_batch


def set_train_bar_description_dict(train_bar, epoch: int, num_epochs: int, train_running_results: dict):
    images_in_epoch = train_running_results['images_in_epoch']
    bar_content = ''
    for key in train_running_results:
        if key != 'images_in_epoch':
            bar_content += f'{key}: {train_running_results[key] / images_in_epoch:.3f}, '
    train_bar.set_description(
        desc=f"[{epoch}/{num_epochs}] "
             f"{bar_content}"
    )


def save_model(name: str, cur_epoch: int, model_to_save: nn.Module, training_path: Path):
    """
    Save the weights of the model during training
    :param name: name of the file
    :param cur_epoch: current epoch
    :param model_to_save: pytorch model to be saved
    :param training_path: path associated with the training run
    """
    models_path = training_path / "saved_models"
    models_path.mkdir(exist_ok=True, parents=True)
    model_name = model_to_save.__class__.__name__
    torch.save({
        'epoch': cur_epoch,
        model_name: model_to_save.state_dict(),
    }, str(models_path / f'{name}.pt'))


