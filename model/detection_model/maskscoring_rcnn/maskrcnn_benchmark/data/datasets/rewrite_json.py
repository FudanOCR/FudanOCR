import json
import os
import cv2


def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    path = '/workspace/mnt/group/ocr/qiutairu/dataset/LSVT_full_train'
    json_name = 'train_labels.json'
    data = read_json(os.path.join(path, json_name))

    new_data = {}
    for key, value in data.items():
        print(key)
        if os.path.exists(os.path.join(path, 'train_images', key+'.jpg')):
            img = cv2.imread(os.path.join(path, 'train_images', key+'.jpg'))
        else:
            img = cv2.imread(os.path.join(path, 'test_images', key+'.jpg'))

        new_data[key] = {
            'polygons': value,
            'license': 1,
            'width': img.shape[1],
            'height': img.shape[0]
        }

    with open(os.path.join(path, 'new_'+json_name), 'w') as f:
        json.dump(new_data, f)
