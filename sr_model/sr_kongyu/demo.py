import yaml
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

from model.instance.ocr_sr import OCRSRModel
# from model.instance.deblur import DeblurModel

if __name__ == '__main__':
    ocr_sr_config = yaml.load(open('configs/OCRSR.yml', 'r', encoding='utf-8'), Loader=yaml.CLoader)
    ocr_sr_model = OCRSRModel(ocr_sr_config)
    ocr_sr_test_img = cv2.imread('sr_test.png', cv2.IMREAD_COLOR)
    ocr_sr_test_img = cv2.resize(ocr_sr_test_img, None, None, 0.5, 0.5, interpolation=cv2.INTER_AREA)
    ocr_sr_test_res = ocr_sr_model(images=[ocr_sr_test_img])
    cv2.imwrite('sr_result.png', ocr_sr_test_res[0])

    # deblur_config = yaml.load(open('configs/Deblur.yml', 'r', encoding='utf-8'), Loader=yaml.CLoader)
    # deblur_model =DeblurModel(deblur_config)
    # deblur_test_img = cv2.imread('deblur_test.png', cv2.IMREAD_COLOR)
    # deblur_test_res = deblur_model(images=[deblur_test_img])
    # cv2.imwrite('deblur_result.png', deblur_test_res[0])
