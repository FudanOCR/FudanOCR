import json
import os


def get_bbox_id(elem):
    return int(elem.lstrip('gt_').rstrip('.jpg').split('_')[1])


if __name__ == '__main__':
    json_path = '/workspace/mnt/group/ocr/wangxunyan/maskscoring_rcnn/crop_train/crop_result_js.json'
    img_path = '/workspace/mnt/group/ocr/wangxunyan/maskscoring_rcnn/crop_train/crop_images_new'
    output_path = '/workspace/mnt/group/ocr/wangxunyan/maskscoring_rcnn/crop_train/new_crop_result_js.json'

    with open(json_path, 'r') as f:
        json_list = json.load(f)

    img_list = os.listdir(img_path)
    img_list.sort(key=get_bbox_id)

    output_dict = {}
    print(len(json_list))
    print(ssss)
    for poly_idx, polygon in enumerate(json_list):
        key = img_list[poly_idx].rstrip('.jpg')
        value = [polygon]

        output_dict[key] = value

    with open(output_path, 'w') as f:
        json.dump(output_dict, f)
