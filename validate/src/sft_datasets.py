"""Provides data for training and testing."""
import base64
import io
import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Literal

import PIL
import PIL.Image
import torch
from torch.utils.data import Dataset

import os
from os import listdir
from os.path import isfile
from os.path import join

def collate_fn(batch):
    '''
    function which discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    '''
    batch = list(filter(lambda x: x is not None, batch))

    collate_batch = {}
    for key in batch[0].keys():
        collate_batch[key] = [b[key] for b in batch]
    return collate_batch

dataset_path = Path('/mnt/input_zuo/ZS-CIR')
# dataset_path = Path('/home/v-zuoleili/Data')

# FashionIQ Dataset
class FashionIQDataset(Dataset):
    """
    FashionIQ dataset class for PyTorch.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield :a dict with keys ['image', 'image_name']
        - In 'relative' mode the dataset yield dict with keys:
            - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_captions'] when
             split in ['train', 'val']
            - ['reference_image', 'reference_name', 'relative_captions'] when split == test
    """
    def __init__(self, split: Literal['train', 'val', 'test'], dress_types: List[str],
                 mode: Literal['relative', 'classic'], preprocess: callable = None, no_duplicates: Optional[bool] = False):
        """
        :param dataset_path: path to the FashionIQ dataset
        :param split: dataset split, should be in ['train, 'val', 'test']
        :param dress_types: list of fashionIQ categories, each category should be in ['dress', 'shirt', 'toptee']
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield a dict with keys ['image', 'image_name']
            - In 'relative' mode the dataset yield dict with keys:
                - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_captions']
                 when split in ['train', 'val']
                - ['reference_image', 'reference_name', 'relative_captions'] when split == test
        :param preprocess: function which preprocesses the image
        :param no_duplicates: if True, the dataset will not yield duplicate images in relative mode, does not affect classic mode
        """
        self.mode = mode
        self.dress_types = dress_types
        self.split = split
        self.no_duplicates = no_duplicates

        # Validate the inputs
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")

        self.preprocess = preprocess

        # get triplets made by (reference_image, target_image, a pair of relative captions)
        self.triplets: List[dict] = []
        for dress_type in dress_types:
            with open(dataset_path / 'FashionIQ' / 'captions' / f'cap.{dress_type}.{split}.json') as f:
                self.triplets.extend(json.load(f))

        # Remove duplicates from
        if self.no_duplicates:
            seen = set()
            new_triplets = []
            for triplet in self.triplets:
                if triplet['candidate'] not in seen:
                    seen.add(triplet['candidate'])
                    new_triplets.append(triplet)
            self.triplets = new_triplets

        # get the image names
        self.image_names: list = []
        for dress_type in dress_types:
            with open(dataset_path / 'FashionIQ' / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
                self.image_names.extend(json.load(f))

        print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")

    def __getitem__(self, index) -> dict:
        try:
            if self.mode == 'relative':
                relative_captions = self.triplets[index]['captions']
                reference_name = self.triplets[index]['candidate']

                if self.split in ['train', 'val']:
                    reference_image_path = dataset_path / 'FashionIQ' / 'images' / f"{reference_name}.jpg"
                    reference_image = PIL.Image.open(reference_image_path)
                    if self.preprocess is not None:
                        reference_image = self.preprocess(reference_image)
                    target_name = self.triplets[index]['target']
                    target_image_path = dataset_path / 'FashionIQ' / 'images' / f"{target_name}.jpg"
                    target_image = PIL.Image.open(target_image_path)
                    if self.preprocess is not None:
                        target_image = self.preprocess(target_image)
                    # shared_concept = self.triplets[index]['shared_concept']

                    return {
                        'reference_image': reference_image,
                        'reference_name': reference_name,
                        'target_image': target_image,
                        'target_name': target_name,
                        'relative_captions': relative_captions,
                        # 'shared_concept': shared_concept,
                    }

                elif self.split == 'test':
                    reference_image_path = dataset_path / 'FashionIQ' / 'images' / f"{reference_name}.jpg"
                    reference_image = PIL.Image.open(reference_image_path)
                    if self.preprocess is not None:
                        reference_image = self.preprocess(reference_image)

                    return {
                        'reference_image': reference_image,
                        'reference_name': reference_name,
                        'relative_captions': relative_captions
                    }

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image_path = dataset_path / 'FashionIQ' / 'images' / f"{image_name}.jpg"
                image = PIL.Image.open(image_path)
                if self.preprocess is not None:
                    image = self.preprocess(image)

                return {
                    'image': image,
                    'image_name': image_name
                }

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


# CIRR Dataset
class CIRRDataset(Dataset):
    """
    CIRR dataset class for PyTorch dataloader.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield a dict with keys ['image', 'image_name']
        - In 'relative' mode the dataset yield dict with keys:
            - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_caption', 'group_members']
             when split in ['train', 'val']
            - ['reference_image', 'reference_name' 'relative_caption', 'group_members', 'pair_id'] when split == test
    """
    def __init__(self, split: Literal['train', 'val', 'test'],
                 mode: Literal['relative', 'classic'], preprocess: callable = None, no_duplicates: Optional[bool] = False):
        """
        :param dataset_path: path to the CIRR dataset
        :param split: dataset split, should be in ['train', 'val', 'test']
        :param mode: dataset mode, should be in ['relative', 'classic']:
                - In 'classic' mode the dataset yield a dict with keys ['image', 'image_name']
                - In 'relative' mode the dataset yield dict with keys:
                    - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_caption',
                    'group_members'] when split in ['train', 'val']
                    - ['reference_image', 'reference_name' 'relative_caption', 'group_members', 'pair_id'] when split == test
        :param preprocess: function which preprocesses the image
        :param no_duplicates: if True, the dataset will not yield duplicate images in relative mode, does not affect classic mode
        """
        self.preprocess = preprocess
        self.mode = mode
        self.split = split
        self.no_duplicates = no_duplicates

        if split == "test":
            split = "test1"
            self.split = "test1"

        # Validate inputs
        if split not in ['test1', 'train', 'val']:
            raise ValueError("split should be in ['test1', 'train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        # get triplets made by (reference_image, target_image, relative caption)
        with open(dataset_path / 'CIRR' / 'cirr' / 'captions' / f'cap.rc2.{split}.json') as f:
            self.triplets = json.load(f)

        # Remove duplicates from triplets
        if self.no_duplicates:
            seen = set()
            new_triplets = []
            for triplet in self.triplets:
                if triplet['reference'] not in seen:
                    seen.add(triplet['reference'])
                    new_triplets.append(triplet)
            self.triplets = new_triplets

        # get a mapping from image name to relative path
        with open(dataset_path / 'CIRR' / 'cirr' / 'image_splits' / f'split.rc2.{split}.json') as f:
            self.name_to_relpath = json.load(f)

        print(f"CIRR {split} dataset in {mode} mode initialized")

    def __getitem__(self, index) -> dict:
        try:
            if self.mode == 'relative':
                group_members = self.triplets[index]['img_set']['members']
                reference_name = self.triplets[index]['reference']
                relative_caption = self.triplets[index]['caption']

                if self.split in ['train', 'val']:
                    reference_image_path = dataset_path / 'CIRR' / self.name_to_relpath[reference_name]
                    # reference_image_path = reference_image_path.with_suffix('.jpg')
                    reference_image = PIL.Image.open(reference_image_path)
                    if self.preprocess is not None:
                        reference_image = self.preprocess(reference_image)
                    target_hard_name = self.triplets[index]['target_hard']
                    target_image_path = dataset_path / 'CIRR' / self.name_to_relpath[target_hard_name]
                    # target_image_path = target_image_path.with_suffix('.jpg')
                    target_image = PIL.Image.open(target_image_path)
                    if self.preprocess is not None:
                        target_image = self.preprocess(target_image)
                    # shared_concept = self.triplets[index]['shared_concept']

                    return {
                        'reference_image': reference_image,
                        'reference_name': reference_name,
                        'target_image': target_image,
                        'target_name': target_hard_name,
                        'relative_caption': relative_caption,
                        'group_members': group_members,
                        # 'shared_concept': shared_concept
                    }

                elif self.split == 'test1':
                    pair_id = self.triplets[index]['pairid']
                    reference_image_path = dataset_path / 'CIRR' / self.name_to_relpath[reference_name]
                    reference_image = PIL.Image.open(reference_image_path)
                    if self.preprocess is not None:
                        reference_image = self.preprocess(reference_image)
                    return {
                        'reference_image': reference_image,
                        'reference_name': reference_name,
                        'relative_caption': relative_caption,
                        'group_members': group_members,
                        'pair_id': pair_id
                    }

            elif self.mode == 'classic':
                image_name = list(self.name_to_relpath.keys())[index]
                image_path = dataset_path / 'CIRR' / self.name_to_relpath[image_name]
                # image_path = image_path.with_suffix('.jpg')
                image = PIL.Image.open(image_path)
                if self.preprocess is not None:
                    image = self.preprocess(image)

                return {
                    'image': image,
                    'image_name': image_name
                }

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.name_to_relpath)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


# New code for CIRCO Dataset
class CIRCODataset(Dataset):
    """
    CIRCO dataset class for PyTorch.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield a dict with keys ['image', 'image_name']
        - In 'relative' mode the dataset yield dict with keys:
            - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_captions', 'shared_concept',
             'gt_img_ids', 'query_id'] when split == 'val'
            - ['reference_image', 'reference_name', 'relative_captions', 'shared_concept', 'query_id'] when split == test
    """

    def __init__(self, split: Literal['val', 'test'], mode: Literal['relative', 'classic'], preprocess: callable = None):
        """
        Args:
            dataset_path (Union[str, Path]): path to CIRCO dataset
            split (str): dataset split, should be in ['test', 'val']
            mode (str): dataset mode, should be in ['relative', 'classic']
            preprocess (callable): function which preprocesses the image
        """
        # Set dataset paths and configurations
        self.mode = mode
        self.split = split
        self.preprocess = preprocess

        # Ensure input arguments are valid
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'val']:
            raise ValueError("split should be in ['test', 'val']")

        # Load COCO images information
        with open(dataset_path / 'CIRCO' / 'COCO2017_unlabeled' / "annotations" / "image_info_unlabeled2017.json", "r") as f:
            imgs_info = json.load(f)

        self.img_paths = [dataset_path / 'CIRCO' / 'COCO2017_unlabeled' / "unlabeled2017" / img_info["file_name"] for img_info in imgs_info["images"]]
        self.img_ids = [img_info["id"] for img_info in imgs_info["images"]]
        self.img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(self.img_ids)}

        # get CIRCO annotations
        with open(dataset_path / 'CIRCO' / 'annotations' / f'{split}.json', "r") as f:
            self.annotations: List[dict] = json.load(f)

        # Get maximum number of ground truth images (for padding when loading the images)
        self.max_num_gts = 23  # Maximum number of ground truth images

        print(f"CIRCODataset {split} dataset in {mode} mode initialized")

    def get_target_img_ids(self, index) -> Dict[str, int]:
        """
        Returns the id of the target image and ground truth images for a given query

        Args:
            index (int): id of the query

        Returns:
             Dict[str, int]: dictionary containing target image id and a list of ground truth image ids
        """
        return {
            'target_img_id': self.annotations[index]['target_img_id'],
            'gt_img_ids': self.annotations[index]['gt_img_ids']
        }

    def __getitem__(self, index) -> dict:
        """
        Returns a specific item from the dataset based on the index.

        In 'classic' mode, the dataset yields a dictionary with the following keys: [img, img_id]
        In 'relative' mode, the dataset yields dictionaries with the following keys:
            - [reference_img, reference_img_id, target_img, target_img_id, relative_caption, shared_concept, gt_img_ids,
            query_id]
            if split == val
            - [reference_img, reference_img_id, relative_caption, shared_concept, query_id]  if split == test
        """
        if self.mode == 'relative':
            # Get the query id
            query_id = str(self.annotations[index]['id'])

            # Get relative caption and shared concept
            relative_caption = self.annotations[index]['relative_caption']
            shared_concept = self.annotations[index]['shared_concept']
            # shared_concept = self.annotations[index]['new_shared_concept']

            # Get the reference image
            reference_img_id = str(self.annotations[index]['reference_img_id'])
            reference_img_path = self.img_paths[self.img_ids_indexes_map[reference_img_id]]
            reference_img = PIL.Image.open(reference_img_path)
            if self.preprocess is not None:
                reference_img = self.preprocess(reference_img)

            if self.split == 'val':
                # Get the target image and ground truth images
                target_img_id = str(self.annotations[index]['target_img_id'])
                gt_img_ids = [str(x) for x in self.annotations[index]['gt_img_ids']]
                # Pad ground truth image IDs with zeros for collate_fn
                gt_img_ids += [''] * (self.max_num_gts - len(gt_img_ids))

                target_img_path = self.img_paths[self.img_ids_indexes_map[target_img_id]]
                target_img = PIL.Image.open(target_img_path)
                if self.preprocess is not None:
                    target_img = self.preprocess(target_img)
                # new_shared_concept = self.annotations[index]['new_shared_concept']

                return {
                    'reference_image': reference_img,
                    'reference_name': reference_img_id,
                    'target_image': target_img,
                    'target_name': target_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'gt_img_ids': gt_img_ids,
                    'query_id': query_id,
                    # 'new_shared_concept': new_shared_concept,
                }

            elif self.split == 'test':
                return {
                    'reference_image': reference_img,
                    'reference_name': reference_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'query_id': query_id,
                }

        elif self.mode == 'classic':
            # Get image ID and image path
            img_id = str(self.img_ids[index])
            img_path = self.img_paths[index]
            img = PIL.Image.open(img_path)
            # Preprocess image and return
            if self.preprocess is not None:
                img = self.preprocess(img)
            return {
                'image': img,
                'image_name': img_id
            }

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        if self.mode == 'relative':
            return len(self.annotations)
        elif self.mode == 'classic':
            return len(self.img_ids)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


# GeneCIS Datasets
class COCODataset(Dataset):
    def __init__(self, preprocess: callable):
        # Set dataset paths and configurations
        self.preprocess = preprocess

    def load_sample(self, sample):
        val_img_id = sample['val_image_id']
        image_path = dataset_path / 'GeneCIS' / 'COCO_val2017' / f"{val_img_id}.jpg"
        if self.preprocess is not None:
            orig_img = PIL.Image.open(image_path)
            img = self.preprocess(orig_img)
            return img
        else:
            return image_path

class COCOValSubset(COCODataset):
    def __init__(self, val_split_path, data_split, tokenizer=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        with open(val_split_path) as f:
            val_samples = json.load(f)

        self.val_samples = val_samples
        self.tokenizer = tokenizer
        self.data_split = data_split

    def __getitem__(self, index):
        """
        Follow same return signature as CIRRSubset
        """
        sample = self.val_samples[index]
        orig_reference = sample['reference']

        target = sample['target']
        gallery = sample['gallery']
        caption = sample['condition']
        # "change" in self.data_split:
        if "change" in self.data_split:
            caption = f"Change an object of {caption}"
        elif "focus" in self.data_split:
            caption = f"Focus on the object of {caption}"
        reference = self.load_sample(orig_reference)
        target = self.load_sample(target)
        # blip_ref_img = self.load_blip_sample(orig_reference)
        gallery = [self.load_sample(i) for i in gallery]

        if self.preprocess is not None:
            gallery = torch.stack(gallery)
            gallery_and_target = torch.cat([target.unsqueeze(0), gallery])
        else:
            gallery_and_target = [target] + gallery

        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        # By construction, target_rank = 0
        # return reference, caption, ref_img, gallery_and_target, 0
        return reference, caption, gallery_and_target, 0

    def __len__(self):
        return len(self.val_samples)

DILATION = 0.7
PAD_CROP = True

def expand2square(pil_img, background_color=(0, 0, 0)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = PIL.Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = PIL.Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class VAWDataset(Dataset):
    def __init__(self, preprocess) -> None:
        super().__init__()

        self.preprocess = preprocess
        self.dilate = DILATION
        self.pad_crop = PAD_CROP

    def load_cropped_image(self, img):
        image_id = img['image_id']
        bbox = img['instance_bbox']

        # Get image
        image_path = dataset_path / 'GeneCIS' / 'VG_100K_all' / f"{image_id}.jpg"
        im = PIL.Image.open(image_path).convert('RGB')
        im_width, im_height = im.size

        width = bbox[2]
        height = bbox[3]

        if self.dilate:
            orig_left, orig_top = bbox[0], bbox[1]
            left, top = max(0, orig_left - self.dilate * width), max(0, orig_top - self.dilate * height)
            right, bottom = min(im_width, left + (1 + self.dilate) * width), min(im_height,
                                                                                 top + (1 + self.dilate) * height)
        else:
            left, top = bbox[0], bbox[1]
            right, bottom = bbox[0] + width, bbox[1] + height

        im = im.crop((left, top, right, bottom))

        if self.pad_crop:
            if im.mode == 'L':
                bg_color = (0,)
            else:
                bg_color = (0, 0, 0)
            im = expand2square(im, bg_color)
        return im

    def load_sample(self, sample):
        im = self.load_cropped_image(sample)
        if self.preprocess is not None:
            im = self.preprocess(im)
            return im
        else:
            # Existing code to save and encode the image
            with io.BytesIO() as output:
                im.save(output, format="PNG")
                encoded_image = base64.b64encode(output.getvalue()).decode("ascii")
            image_url = f"data:image/png;base64," + encoded_image # 这句话的含义不清晰

            return image_url

class VAWValSubset(VAWDataset):
    def __init__(self, val_split_path, data_split, tokenizer=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        with open(val_split_path) as f:
            val_samples = json.load(f)

        self.val_samples = val_samples
        self.tokenizer = tokenizer
        self.data_split = data_split

    def __getitem__(self, index):
        """
        Follow same return signature as CIRRSubset
            (Except for returning reference object at the end)
        """
        sample = self.val_samples[index]
        orig_reference = sample['reference']

        target = sample['target']
        gallery = sample['gallery']
        caption = sample['condition']
        if "change" in self.data_split:
            caption = f"Change an attribute of {caption}"
        elif "focus" in self.data_split:
            caption = f"Focus on the attribute of {caption}"
            # print(caption)
        reference = self.load_sample(orig_reference)
        target = self.load_sample(target)
        # blip_ref_img = self.load_blip_sample(orig_reference)
        gallery = [self.load_sample(i) for i in gallery]

        if self.preprocess is not None:
            gallery = torch.stack(gallery)
            gallery_and_target = torch.cat([target.unsqueeze(0), gallery])
        else:
            gallery_and_target = [target] + gallery

        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        # By construction, target_rank = 0
        # return reference, caption, blip_ref_img, gallery_and_target, 0
        return reference, caption, gallery_and_target, 0

    def __len__(self):
        return len(self.val_samples)
