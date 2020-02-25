import cv2
import os
import numpy as np
from model.detection_model.TextSnake_pytorch.dataset.read_json import read_json, read_dict


def recorder(record, ratio):
    ranges = [key.split('~') for key in record.keys()]
    for range in ranges:
        if int(range[0]) <= ratio < int(range[1]):
            record['{}~{}'.format(range[0], range[1])] += 1
            break
    return record


if __name__ == '__main__':
    path = '/home/shf/fudan_ocr_system/datasets/ICDAR19/'
    json_name = 'train_labels.json'
    maxlen = 1280

    train_files = os.listdir(os.path.join(path, 'train_images'))
    test_files = os.listdir(os.path.join(path, 'test_images'))
    data_dict = read_json(os.path.join(path, json_name))

    legal_record = {
        '1~2': 0,
        '2~3': 0,
        '3~4': 0,
        '4~5': 0,
        '5~6': 0,
        '6~7': 0,
        '7~99999999': 0,
    }

    illegal_record = {
        '1~2': 0,
        '2~3': 0,
        '3~4': 0,
        '4~5': 0,
        '5~6': 0,
        '6~7': 0,
        '7~99999999': 0,
    }

    # max_area = -1
    # min_area = 999999999

    with open('record.txt', 'w') as f:
        for idx, file in enumerate(train_files):
            polygons = read_dict(data_dict, file)
            im = cv2.imread(os.path.join(path, 'train_images', file))
            h, w = im.shape[:2]
            scale = 1.0
            if max(h, w) > maxlen:
                scale = float(maxlen) / h if h > w else float(maxlen) / w
            im = cv2.resize(im, (int(w*scale), int(h*scale)))

            print(idx, file, len(polygons))
            for polygon in polygons:
                polygon.points[:, 0] = (polygon.points[:, 0] * scale).astype(np.int32)
                polygon.points[:, 1] = (polygon.points[:, 1] * scale).astype(np.int32)
                if not polygon.illegibility:
                    drawing = np.zeros(im.shape[:2], np.uint8)
                    _, (w, h), _ = cv2.minAreaRect(polygon.points.astype(np.int32))
                    ratio = float(max(w, h)) / min(w, h)

                    f.write(str(ratio) + '\n')

                    recorder(legal_record, ratio)
                else:
                    drawing = np.zeros(im.shape[:2], np.uint8)
                    _, (w, h), _ = cv2.minAreaRect(polygon.points.astype(np.int32))
                    ratio = float(max(w, h)) / min(w, h)

                    f.write(str(ratio) + '\n')

                    recorder(illegal_record, ratio)

            if idx % 10 == 0:
                print('record: ', legal_record)
                print('illegal: ', illegal_record)

        print('record: ', legal_record)
        print('illegal: ', illegal_record)

    print("Test Images")
    with open('record2.txt', 'w') as f:
        for idx, file in enumerate(test_files):
            polygons = read_dict(data_dict, file)
            im = cv2.imread(os.path.join(path, 'test_images', file))
            h, w = im.shape[:2]
            scale = 1.0
            if max(h, w) > maxlen:
                scale = float(maxlen) / h if h > w else float(maxlen) / w
            im = cv2.resize(im, (int(w * scale), int(h * scale)))

            print(idx, file, len(polygons))

            for polygon in polygons:
                polygon.points[:, 0] = (polygon.points[:, 0] * scale).astype(np.int32)
                polygon.points[:, 1] = (polygon.points[:, 1] * scale).astype(np.int32)
                if not polygon.illegibility:
                    drawing = np.zeros(im.shape[:2], np.uint8)
                    _, (w, h), _ = cv2.minAreaRect(polygon.points.astype(np.int32))
                    ratio = float(max(w, h)) / min(w, h)

                    f.write(str(ratio) + '\n')

                    recorder(legal_record, ratio)
                else:
                    drawing = np.zeros(im.shape[:2], np.uint8)
                    _, (w, h), _ = cv2.minAreaRect(polygon.points.astype(np.int32))
                    ratio = float(max(w, h)) / min(w, h)

                    f.write(str(ratio) + '\n')

                    recorder(illegal_record, ratio)

            if idx % 10 == 0:
                print('record: ', legal_record)
                print('illegal: ', illegal_record)

        print('record: ', legal_record)
        print('illegal: ', illegal_record)


