import os
import cv2
import shutil

cnt = 0

def saveByOrder(image,opt):

    address = './Image_Visualization'
    if opt.BASE.EXPERIMENT_NAME != '':
        address = address + '_' + opt.BASE.EXPERIMENT_NAME

    if opt.VISUALIZE.RECOGNITION_VISUALIZE != True or opt.FUNCTION.VAL_ONLY != True:
        return

    # print("Start Visualize")

    try:
        os.mkdir(address)
    except:
        pass

    global cnt
    for img in image:
        img = img.permute(1,2,0)
        cv2.imwrite(os.path.join(address, str(cnt) + '.jpg'), (img.numpy() + 1) * 128)
        cnt += 1

    # print("Finish!")


def wrong_gt_predict(gt,predict,index,opt):

    address = './Image_Visualization'
    if opt.BASE.EXPERIMENT_NAME != '':
        address = address + '_' + opt.BASE.EXPERIMENT_NAME

    if opt.VISUALIZE.RECOGNITION_VISUALIZE != True or opt.FUNCTION.VAL_ONLY != True:
        return

    try:
        os.mkdir(address+'_wrong')
    except:
        pass

    img = cv2.imread(address+'/'+str(index)+'.jpg')
    print("路径：",address+"_wrong/{0}_{1}_{2}.jpg".format(str(index),gt,predict))
    cv2.imwrite(address+"_wrong/{0}_{1}_{2}.jpg".format(str(index),gt,predict),img)