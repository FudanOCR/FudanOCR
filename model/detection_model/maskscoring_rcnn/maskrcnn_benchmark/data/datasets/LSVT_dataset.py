import torch
import os
import numpy as np

from maskrcnn_benchmark.structures.bounding_box import RBoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.data.datasets import utils


class LSVTDataset(object):
    def __init__(self, ann_file, root, remove_images_without_annotations, transforms=None):
        self.root = root
        self.annotations = utils.read_json(ann_file)
        if os.path.split(ann_file)[-1] == "new_train_labels.json":
            self.img_list = os.listdir(self.root)
        else:
            self.img_list = [key + '.jpg' for key in list(self.annotations.keys())]
        self.img_list.sort(key=utils.get_img_id)
        self.transforms = transforms

        if remove_images_without_annotations:
            # remove image from ignore_list.txt
            ignore_list_dir = '/'.join(self.root.split('/')[:-1])
            ignore_list = []
            with open(os.path.join(ignore_list_dir, 'ignore_list.txt'), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    ignore_list.append(line.replace('\n', ''))

            for idx, img_name in enumerate(self.img_list):
                if img_name in ignore_list:
                    self.img_list[idx] = ''
                    continue

                # filter illegal
                anno = utils.read_anno(self.annotations, img_name)
                anno = [obj for obj in anno if not obj['illegibility']]
                if not len(anno) > 0:
                    self.img_list[idx] = ''

            self.img_list = [img_name for img_name in self.img_list if not img_name == '']

        self.id_to_img_map = {img_id: img_name for img_id, img_name in enumerate(self.img_list)}
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate([1])
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

    def __getitem__(self, idx):
        img_name = self.id_to_img_map[idx]
        img = utils.pil_load_img(os.path.join(self.root, img_name))
        anno = utils.read_anno(self.annotations, img_name)

        # filter illegal
        anno = [obj for obj in anno if not obj['illegibility']]

        # bounding boxes
        boxes = [utils.generate_rbox(obj["points"], np.array(img).shape[:2]) for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 5)  # guard against no boxes
        target = RBoxList(boxes, img.size, mode="xywha")

        # classes
        classes = [1] * len(anno)
        classes = torch.tensor(classes)
        target.add_field("labels", classes)
        target.add_field("difficult", torch.tensor([0 for i in range(len(classes))]))

        # masks
        masks = [obj["points"].reshape((1, -1)).tolist() for obj in anno]
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)

        # target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        assert target is not None, "{} target is None.".format(img_name)

        return img, target, idx

    def __len__(self):
        return len(self.img_list)

    def get_img_info(self, index):
        img_name = self.id_to_img_map[index]

        return utils.read_info(self.annotations, img_name)
