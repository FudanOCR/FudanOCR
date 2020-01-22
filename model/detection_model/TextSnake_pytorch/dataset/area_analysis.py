import cv2
import os
import numpy as np
from dataset.read_json import read_json, read_dict


def recorder(record, area):
    ranges = [key.split('~') for key in record.keys()]
    for range in ranges:
        if int(range[0]) <= area <= int(range[1]):
            record['{}~{}'.format(range[0], range[1])] += 1
            break
    return record


if __name__ == '__main__':
    path = '/home/shf/fudan_ocr_system/datasets/ICDAR19'
    json_name = 'train_labels.json'
    maxlen = 1280

    train_files = os.listdir(os.path.join(path, 'train_images'))
    test_files = os.listdir(os.path.join(path, 'test_images'))
    data_dict = read_json(os.path.join(path, json_name))

    legal_record = {
        '0~99': 0,
        '100~499': 0,
        '500~999': 0,
        '1000~1999': 0,
        '2000~2999': 0,
        '3000~3999': 0,
        '4000~4999': 0,
        '5000~5999': 0,
        '6000~6999': 0,
        '7000~7999': 0,
        '8000~8999': 0,
        '9000~9999': 0,
        '10000~19999': 0,
        '20000~29999': 0,
        '30000~39999': 0,
        '40000~49999': 0,
        '50000~59999': 0,
        '60000~69999': 0,
        '70000~79999': 0,
        '80000~89999': 0,
        '90000~99999': 0,
        '100000~99999999': 0
    }

    illegal_record = {
        '0000~0099': 0,
        '0100~0499': 0,
        '0500~0999': 0,
        '1000~1999': 0,
        '2000~99999999': 0
    }

    max_area = -1
    min_area = 999999999

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
                    poly_mask = cv2.fillPoly(drawing, np.array([polygon.points], dtype=np.int32), 255)
                    area = np.sum(np.greater(poly_mask, 0))

                    f.write(str(area) + '\n')

                    if area >= max_area:
                        max_area = area
                    if area <= min_area:
                        min_area = area

                    recorder(legal_record, area)
                else:
                    drawing = np.zeros(im.shape[:2], np.uint8)
                    poly_mask = cv2.fillPoly(drawing, np.array([polygon.points], dtype=np.int32), 255)
                    area = np.sum(np.greater(poly_mask, 0))

                    recorder(illegal_record, area)

            if idx % 10 == 0:
                print('record: ', legal_record)
                print('illegal: ', illegal_record)
                print('max_area: {}, min_area: {}'.format(max_area, min_area))

        print('record: ', legal_record)
        print('illegal: ', illegal_record)
        print('max_area: {}, min_area: {}'.format(max_area, min_area))

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
                    poly_mask = cv2.fillPoly(drawing, np.array([polygon.points], dtype=np.int32), 255)
                    area = np.sum(np.greater(poly_mask, 0))

                    f.write(str(area) + '\n')

                    if area >= max_area:
                        max_area = area
                    if area <= min_area:
                        min_area = area

                    recorder(legal_record, area)
                else:
                    drawing = np.zeros(im.shape[:2], np.uint8)
                    poly_mask = cv2.fillPoly(drawing, np.array([polygon.points], dtype=np.int32), 255)
                    area = np.sum(np.greater(poly_mask, 0))

                    recorder(illegal_record, area)

            if idx % 10 == 0:
                print('record: ', legal_record)
                print('illegal: ', illegal_record)
                print('max_area: {}, min_area: {}'.format(max_area, min_area))

        print('record: ', legal_record)
        print('illegal: ', illegal_record)
        print('max_area: {}, min_area: {}'.format(max_area, min_area))


