"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import json
import random
import os
import re
from typing import Any, List

import numpy as np
from elements import Background, Document
from PIL import Image
from synthtiger import components, layers, templates

seed_value = 3
np.random.seed(seed_value)

class SynthDoG(templates.Template):
    def __init__(self, config=None, split_ratio: List[float] = [0.8, 0.1, 0.1]):
        super().__init__(config)
        if config is None:
            config = {}

        self.quality = config.get("quality", [50, 95])
        self.landscape = config.get("landscape", 0.5)
        self.short_size = config.get("short_size", [720, 1024])
        self.aspect_ratio = config.get("aspect_ratio", [1, 2])
        self.background = Background(config.get("background", {}))
        self.document = Document(config.get("document", {}))
        self.effect = components.Iterator(
            [
                components.Switch(components.RGB()),
                components.Switch(components.Shadow()),
                components.Switch(components.Contrast()),
                components.Switch(components.Brightness()),
                components.Switch(components.MotionBlur()),
                components.Switch(components.GaussianBlur()),
            ],
            **config.get("effect", {}),
        )

        # config for splits
        self.splits = ["train", "validation", "test"]
        self.split_ratio = split_ratio
        self.split_indexes = np.random.choice(3, size=10000, p=split_ratio)
        self.instructions = [
            "OCR this picture with grounding",  
            "Text Spotting",
            "Identify the text in this image",    
            "Where is the text located in this image?",
            "Spot the text within this picture", 
            "Annotate the text and its position",
            "Locate the text in the picture",    
            "What words are present in this image?",   
            "Display text and corresponding locations",
            "Show where the text starts in the image", 
            "Highlight the words from this image",     
            "Point out the text and where it is",
            "Extract text from the image", 
            "Describe the text and its exact position",
            "Read the image and indicate text positions",    
            "Find text and provide coordinates", 
            "Just show me the image text", 
            "Detail the image text and placement",     
            "What is the text content in this image?", 
            "Analyze and reveal the text locations",   
            "Pinpoint text in this image", 
            "Report what text is seen here",   
            "告诉我图片中有哪些文字",
            "图中有哪些文字",  
            "我想知道图中文字内容和位置",  
            "读取图片中的文字位置和内容",  
            "请识别图片中的文本内容",
            "详细描述图片中的文本和位置",  
            "图片文本读取",    
            "标出图中所有文字和其位置",    
            "定位并读取图片中的文字",
            "哪些文字在图片中",
            "图像中的文本是什么",    
            "提取并显示图像中的文字位置",  
            "解析图像并指出文字所在",
            "图片中的文本和对应位置列出来",
            "仅显示图中的文本",
            "展示图片文本及其坐标",  
            "图片中文本位置的解析",  
            "在图中找出文字",  
            "提取图中文本及其开始坐标",    
            "读出并标记图中的所有文字",    
            "仅提取图片中的文字",    
            "展示图中文本和它们的位置",    
            "定位图中文本",    
            "识别并定位图片中的文本",
        ]
        self.instruction_types = [
            True,
            True,
            False,
            True,
            False,
            True,
            True,
            False,
            True,
            True,
            False,
            True,
            False,
            True,
            True,
            True,
            False,
            True,
            False,
            True,
            True,
            False,
            False,
            False,
            True,
            True,
            False,
            True,
            False,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            False,
            True,
            True,
            False,
            True,
            True,
            False,
            True,
            False,
            True,
        ]
        self.instructions_combined = list(zip(self.instructions, self.instruction_types))

    def generate(self):
        landscape, short_size, aspect_ratio = np.random.rand(), np.random.randint(self.short_size[0], self.short_size[1] + 1), np.random.uniform(self.aspect_ratio[0], self.aspect_ratio[1])
        landscape = landscape < self.landscape
        long_size = int(short_size * aspect_ratio)
        size = (long_size, short_size) if landscape else (short_size, long_size)

        bg_layer = self.background.generate(size)
        paper_layer, text_layers, texts = self.document.generate(size)

        rois = [np.array(layer.quad, dtype=int) for layer in text_layers]

        document_group = layers.Group([*text_layers, paper_layer])
        document_space = np.clip(size - document_group.size, 0, None)
        document_group.left = np.random.randint(document_space[0] + 1)
        document_group.top = np.random.randint(document_space[1] + 1)

        layer = layers.Group([*document_group.layers, bg_layer]).merge()
        self.effect.apply([layer])

        image = layer.output(bbox=[0, 0, *size])
        labels_joined = " ".join(texts)
        label = re.sub(r"\s+", " ", labels_joined.strip())
        quality = np.random.randint(self.quality[0], self.quality[1] + 1)

        data = {
            "image": image,
            "label": label,
            "quality": quality,
            "rois": rois
        }

        return data

    def init_save(self, root):
        if not os.path.exists(root):
            os.makedirs(root)
        for split in self.splits:
            image_dirpath = os.path.join(root, split, 'image')
            json_dirpath = os.path.join(root, split)
            os.makedirs(image_dirpath, exist_ok=True)
            os.makedirs(json_dirpath, exist_ok=True)

    def save(self, root, data, idx):
        image = data["image"]
        quality = data["quality"]

        split_idx = self.split_indexes[idx % len(self.split_indexes)]
        output_dirpath = os.path.join(root, self.splits[split_idx])

        image_dirpath = os.path.join(output_dirpath, 'image')

        # change_2
        image_filename = f"image_5_{idx}.jpg"
        image_filepath = os.path.join(image_dirpath, image_filename)

        image = Image.fromarray(image[..., :3].astype(np.uint8))
        image.save(image_filepath, quality=quality)

        json_dirpath = output_dirpath

        # change_3
        metadata_filename = f"metadata_5_{idx // 1000000}.jsonl"
        metadata_filepath = os.path.join(json_dirpath, metadata_filename)

        choice_index = np.random.choice(len(self.instructions_combined))
        instruction, instruction_type = self.instructions_combined[choice_index]

        user_value = f"{instruction}\n<img>{image_filepath}</img>"
        assistant_value = self.create_assistant_reply(instruction_type, data["label"], data["rois"])

        # change_2
        output_data = {
            "id": f"synthdog_output-5-{idx:09d}",
            "conversations": [
                {"from": "user", "value": user_value},
                {"from": "assistant", "value": assistant_value}
            ]
        }

        with open(metadata_filepath, "a") as fp:
            json.dump(output_data, fp, ensure_ascii=False)
            fp.write("\n")
    
    def create_assistant_reply(self, instruction_type, labels, rois):
        # if not instruction_type:
        #     return ''.join([f"<ref>{label}</ref>" for label in labels.split(' ')]) + '<eos>'
        # else:
        reply_parts = []
        for label, single_roi in zip(labels.split(' '), rois):
            roi_str = self.format_roi(single_roi)
            reply_parts.append(f"<ref>{label}</ref><quad>{roi_str}</quad>")
        return ''.join(reply_parts) + '<eos>'

    def format_roi(self, single_roi):
        formatted_coords = ','.join([f"({coord[0]},{coord[1]})" for coord in single_roi])
        return formatted_coords

    def end_save(self, root):
        pass

    def format_metadata(self, image_filename: str, keys: List[str], values: List[Any]):
        """
        Fit gt_parse contents to huggingface dataset's format
        keys and values, whose lengths are equal, are used to constrcut 'gt_parse' field in 'ground_truth' field
        Args:
            keys: List of task_name
            values: List of actual gt data corresponding to each task_name
        """
        assert len(keys) == len(values), "Length does not match: keys({}), values({})".format(len(keys), len(values))

        _gt_parse_v = dict()
        for k, v in zip(keys, values):
            _gt_parse_v[k] = v
        gt_parse = {"gt_parse": _gt_parse_v}
        gt_parse_str = json.dumps(gt_parse, ensure_ascii=False)
        metadata = {"file_name": image_filename, "ground_truth": gt_parse_str}
        return metadata
