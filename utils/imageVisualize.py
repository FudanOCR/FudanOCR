import os
import cv2
import shutil
import numpy as np

cnt = 0


def saveByOrder_beifen(image, opt):
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
        img = img.permute(1, 2, 0)
        cv2.imwrite(os.path.join(address, str(cnt) + '.jpg'), (img.numpy() + 1) * 128)
        cnt += 1

    # print("Finish!")


def visualize(img='', target='', pred='', cnt='', opt='', finish=False, alpha=None):



    address = './Visualization'
    if opt.BASE.EXPERIMENT_NAME != '':
        address = address + '_' + opt.BASE.EXPERIMENT_NAME

    if opt.VISUALIZE.RECOGNITION_VISUALIZE != True or opt.FUNCTION.VAL_ONLY != True:
        return

    if finish:
        f = open(os.path.join(address, 'file.txt'), 'w+', encoding='utf8')
        for file in os.listdir(address):
            f.write(file + '\n')
        f.close()
        return

    if alpha != None:
        index = 0
        attention_center = []
        for i in alpha:
            i = i.tolist()
            # print("第{0}个时间步:".format(cnt))
            # print("关注点为{0}".format(i.index(max(i))))
            '''关注点需要重新定义'''

            point = 0
            for location in range(len(i)):
                point += location * i[location]
            point = int(point)

            # point = i.index(max(i))
            attention_center.append(point)

            '''结束符号'''
            if pred[index] == '$':
                break
            index += 1

    # exit(0)
    # print("***************")

    # if os.path.exists(address):
    #     shutil.rmtree(address)

    try:
        os.mkdir(address)
    except:
        pass

    img = img.permute(1, 2, 0)
    print("可视化路径：", address + "/{0}_{1}_{2}.jpg".format(str(cnt), target, pred))

    '''绘制注意力'''
    # cv2.circle(画布，圆心坐标，半径，颜色，宽度)
    # cv2.circle(image, (300, 300), 40, (0, 255, 0), 2)
    img = (img.numpy() + 1) * 128
    img =  cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if alpha != None:
        for point in attention_center:
            print("注意力点为",point)
            cv2.circle(img, (0+4*point,16), 1, (0, 0, 255), 2)

    cv2.imwrite(address + "/{0}_{1}_{2}.jpg".format(str(cnt), target, pred), img = img)

    # print("Finish!")


def wrong_gt_predict(gt, predict, index, opt):
    address = './Visualization'
    if opt.BASE.EXPERIMENT_NAME != '':
        address = address + '_' + opt.BASE.EXPERIMENT_NAME

    if opt.VISUALIZE.RECOGNITION_VISUALIZE != True or opt.FUNCTION.VAL_ONLY != True:
        return

    try:
        os.mkdir(address + '_all')
    except:
        pass

    f = open('')

    img = cv2.imread(address + '/' + str(index) + '.jpg')
    print("路径：", address + "_wrong/{0}_{1}_{2}.jpg".format(str(index), gt, predict))
    cv2.imwrite(address + "_wrong/{0}_{1}_{2}.jpg".format(str(index), gt, predict), img)
