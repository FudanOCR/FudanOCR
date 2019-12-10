import json
import numpy as np
import cv2
import pycocotools.mask as maskUtils

if __name__ == '__main__':
    result_json_path = '/home/shf/fudan_ocr_system/TextSnake_pytorch/output/art_0429/result.json'
    img_josn_path = '/workspace/mnt/group/ocr/qiutairu/dataset/LSVT_test/new_train_labels.json'
    output_json_path = '/workspace/mnt/group/ocr/qiutairu/code/textsnake_pytorch/output/art_0429/coco_result.json'

    # load json file including prediction results
    with open(result_json_path, 'r') as f:
        result_dict = json.load(f)

    # load json file including image size information
    with open(img_josn_path, 'r') as f:
        img_info_dict = json.load(f)

    output_result = []
    for img_idx, (img_key, img_result) in enumerate(result_dict.items(), 1):
        print('[ {} ]/[ {} ]'.format(img_idx, len(result_dict)))

        coco_image_id = int(img_key.lstrip('res_'))

        # image size
        img_id = img_key.replace('res', 'gt')
        img_h, img_w = img_info_dict[img_id]['height'], img_info_dict[img_id]['width']
        coco_segm_size = [img_h, img_w]

        # for each polygon
        for poly_idx, polygon in enumerate(img_result, 1):
            points = polygon['points']
            conf = polygon['confidence']

            # mask
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(points).astype(np.int32)], 1)
            segm = maskUtils.encode(np.asfortranarray(mask))

            # bbox
            bbox = maskUtils.toBbox(segm).tolist()

            output_result.append({
                'image_id': img_id,
                'segmentation': {
                    'counts': str(segm),
                    'size': coco_segm_size
                },
                'bbox': bbox,
                'score': conf,
                'category_id': 1
            })

    with open(output_json_path, 'w') as f:
        json.dump(output_result, f)
