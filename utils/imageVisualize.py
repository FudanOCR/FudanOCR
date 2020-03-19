import os
import cv2
import shutil

cnt = 0

def saveByOrder_beifen(image,opt):

    address = './Visualization'
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

def saveByOrder(image,target, pred,opt):

    address = './Visualization'
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
        cv2.imwrite(os.path.join(address+"_wrong/{0}_{1}_{2}.jpg".format(str(cnt),target, pred+ '.jpg'), (img.numpy() + 1) * 128))
        cnt += 1

    # print("Finish!")


def wrong_gt_predict(gt,predict,index,opt):

    address = './Visualization'
    if opt.BASE.EXPERIMENT_NAME != '':
        address = address + '_' + opt.BASE.EXPERIMENT_NAME

    if opt.VISUALIZE.RECOGNITION_VISUALIZE != True or opt.FUNCTION.VAL_ONLY != True:
        return

    try:
        os.mkdir(address+'_all')
    except:
        pass

    f = open('')

    img = cv2.imread(address+'/'+str(index)+'.jpg')
    print("路径：",address+"_wrong/{0}_{1}_{2}.jpg".format(str(index),gt,predict))
    cv2.imwrite(address+"_wrong/{0}_{1}_{2}.jpg".format(str(index),gt,predict),img)