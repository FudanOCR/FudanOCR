#coding=utf-8

import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf
from concat import joint

import matplotlib.pyplot as pyplot

import locality_aware_nms as nms_locality
import lanms
#import decode_mask
import json

tf.app.flags.DEFINE_string('test_data_path', '/home/cjy/EAST/icdar2015/ch4_test_images', 'the data of test images')
tf.app.flags.DEFINE_string('test_gt_path', '/home/cjy/EAST/icdar2015/Challenge4_Test_Task1_GT', 'the ground truth info of test images')
tf.app.flags.DEFINE_string('gpu_list', '2', '')
tf.app.flags.DEFINE_string('checkpoint_path', '/home/cjy/east_icdar2015_resnet_v1_50_rbox', '')
tf.app.flags.DEFINE_string('output_dir', './cjytest', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')
tf.app.flags.DEFINE_bool('write_json', True, 'do write json files')

import model
from icdar import restore_rectangle, load_annoataion, shrink_poly

FLAGS = tf.app.flags.FLAGS

DEBUG = False


# def get_images():
#     '''
#     find image files in test data path
#     :return: list of files found
#     '''
#     files = []
#     exts = ['jpg', 'png', 'jpeg', 'JPG']
#     for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
#         for filename in filenames:
#             for ext in exts:
#                 if filename.endswith(ext):
#                     files.append(os.path.join(parent, filename))
#                     break
#     print('Find {} images'.format(len(files)))
#     return files


def get_gt_txt(img_name):
    if FLAGS.dataset == 'icdar2015':
        gt_file = os.path.join(FLAGS.test_gt_path, '%s.txt' % img_name)
    elif FLAGS.dataset == 'icdar2017rctw':
        gt_file = os.path.join(FLAGS.test_gt_path, '%s.txt' % img_name)
    else:
        gt_file = os.path.join(FLAGS.test_gt_path, 'gt_%s.txt' % img_name)
    return gt_file


def get_images_icdar2015():
    image_names = os.listdir(FLAGS.test_data_path)
    image_names = [os.path.join(FLAGS.test_data_path, image_name) for image_name in image_names if image_name[0] != '.']
    if DEBUG:
        image_names = image_names[:10]
    image_names.sort()
    print('Find {} images'.format(len(image_names)))
    return image_names


def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    # resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    # resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(image, score_map, geo_map, timer,im_fn, score_map_thresh=0.8, box_thresh=0.3, nms_thres=0.1):
    '''
    restore text boxes from score map and geo map
    :param image:
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map['F_score1'].shape) == 4:
        score_map1 = score_map['F_score1'][0, :, :, 0]
        score_map2 = score_map['F_score2'][0, :, :, 0]
        geo_map1 = geo_map['F_geometry1'][0, :, :, ]
        geo_map2 = geo_map['F_geometry2'][0, :, :, ]
    # filter the score map
#     pyplot.imshow(score_map1)
#     pyplot.savefig('./out/1_'+os.path.basename(im_fn))

#     pyplot.imshow(score_map2)
#     pyplot.savefig('./out/2_'+os.path.basename(im_fn))

    xy_text1 = np.argwhere(score_map1 > score_map_thresh)


    # sort the text boxes via the y axis
    xy_text1 = xy_text1[np.argsort(xy_text1[:, 0])]

    xy_text2 = np.argwhere(score_map2 > score_map_thresh)

    # sort the text boxes via the y axis
    xy_text2 = xy_text2[np.argsort(xy_text2[:, 0])]

    # restore
    start = time.time()
    # print("hello2")
    # print(xy_text1[0,:])
    # print(xy_text2[0,:])
    # print(geo_map1[xy_text1[0, 0], xy_text1[0, 1], :])
    # print(geo_map2[xy_text2[0, 0], xy_text2[0, 1], :])
    text_box_restored1 = restore_rectangle(xy_text1[:, ::-1]*4, geo_map1[xy_text1[:, 0], xy_text1[:, 1], :]) # N*4*2
    text_box_restored2 = restore_rectangle(xy_text2[:, ::-1]*8, geo_map2[xy_text2[:, 0], xy_text2[:, 1], :])  # N*4*2
    print('{} text boxes before nms'.format(text_box_restored1.shape[0]+text_box_restored2.shape[0]))
    # boxes = np.zeros((text_box_restored.shape[0], 10), dtype=np.float32)
    boxes1 = np.zeros((text_box_restored1.shape[0], 9), dtype=np.float32)
    boxes2 = np.zeros((text_box_restored2.shape[0], 9), dtype=np.float32)

    boxes1[:, :8] = text_box_restored1.reshape((-1, 8))
    boxes1[:, 8] = score_map1[xy_text1[:, 0], xy_text1[:, 1]]

    boxes2[:, :8] = text_box_restored2.reshape((-1, 8))
    boxes2[:, 8] = score_map2[xy_text2[:, 0], xy_text2[:, 1]]

    boxes = np.concatenate((boxes1,boxes2),axis = 0)



    # print(boxes.shape)
    # print(boxes[0, :8])
    timer['restore'] = time.time() - start
    # # Re-Score
    # start = time.time()
    # boxes[:, 9] = rescore(image, boxes, score_map > score_map_thresh)
    # timer['rescore'] = time.time() - start



    # if len(score_map1.shape) == 4:
    #     score_map1 = score_map1[0, :, :, 0]
    #     geo_map1 = geo_map1[0, :, :, ]
    # # filter the score map


    # xy_text1 = np.argwhere(score_map1 > score_map_thresh)
    #
    #
    # # sort the text boxes via the y axis
    # xy_text1 = xy_text1[np.argsort(xy_text1[:, 0])]
    # # restore
    # text_box_restored1 = restore_rectangle(xy_text1[:, ::-1]*8, geo_map1[xy_text1[:, 0], xy_text1[:, 1], :]) # N*4*2
    # # boxes = np.zeros((text_box_restored.shape[0], 10), dtype=np.float32)
    # boxes1 = np.zeros((text_box_restored1.shape[0], 9), dtype=np.float32)
    # boxes1[:, :8] = text_box_restored1.reshape((-1, 8))
    # boxes1[:, 8] = score_map1[xy_text1[:, 0], xy_text1[:, 1]]
    # # # Re-Score
    # # start = time.time()
    # # boxes[:, 9] = rescore(image, boxes, score_map > score_map_thresh)
    # # timer['rescore'] = time.time() - start



    # nms part
    start = time.time()
    #boxes = nms_locality.nms_locality(boxes.astype(np.float32), nms_thres)
    # boxes = nms_locality.standard_nms(boxes.astype(np.float32), nms_thres)
    # boxes = nms_locality.two_criterion_nms(boxes.astype(np.float64), nms_thres)

    #boxes = np.concatenate([boxes,boxes1])
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    print(boxes)

    boxes_final = joint(boxes[:,:8])
    print("Fix Bug ",boxes_final)

    for i in range(len(boxes_final)):
        pts = np.array(boxes_final[i].strip().split(',')).astype(np.float32).reshape(-1, 2)
        pts = np.rint(pts).astype(np.int)
        print(pts)
        pts.tolist()
        pts1 = []
        for i in range(len(pts)):
            pts[i] = tuple(pts[i])
            pts1.append(pts[i])
        print(pts1)
        # compute centroid
        cent = (sum([p[0] for p in pts1]) / len(pts1), sum([p[1] for p in pts1]) / len(pts1))
        print(cent)
        # sort by polar angle
        pts1.sort(key=lambda p: math.atan2(p[1] - cent[1], p[0] - cent[0]))

        for i in range(len(pts1)):
            pts1[i] = list(pts1[i])
        print(pts1)
        pts1 = np.array(pts1)



    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # if DEBUG:
    #     boxes = boxes[np.argsort(boxes[:, 8])[::-1]]
    #     boxes = boxes[np.argsort(boxes[:, 9])[::-1]]
    #     boxes = boxes[:10]
    #     print('selected scores: ', boxes[:, 8])
    #     print('selected rescores: ', boxes[:, 9])

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
#     for i, box in enumerate(boxes):
#         mask = np.zeros_like(score_map1, dtype=np.uint8)
#         cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
#         boxes[i, 8] = cv2.mean(score_map1, mask)[0]
#     boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes[:, :9], timer


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def check_bbox(im, bbox):
    refined_bbox = np.zeros_like(bbox)
    for i, ordi in enumerate(bbox):
        x = ordi[0]
        y = ordi[1]
        refined_x = x if x < im.shape[1] else im.shape[1]-1
        refined_x = refined_x if refined_x > 0 else 0
        refined_y = y if y < im.shape[0] else im.shape[0]-1
        refined_y = refined_y if refined_y > 0 else 0
        refined_bbox[i] = np.array([refined_x, refined_y])
    return refined_bbox


def rescore(im, boxes, score_map):
    # im.shape = (704, 1280)
    # score_map.shape = (176, 320)
    ratio = float(score_map.shape[0]) / float(im.shape[0])
    min_x = np.minimum(np.minimum(boxes[:, 0], boxes[:, 2]), np.minimum(boxes[:, 4], boxes[:, 6]))*ratio
    max_x = np.maximum(np.maximum(boxes[:, 0], boxes[:, 2]), np.maximum(boxes[:, 4], boxes[:, 6]))*ratio
    min_y = np.minimum(np.minimum(boxes[:, 1], boxes[:, 3]), np.minimum(boxes[:, 5], boxes[:, 7]))*ratio
    max_y = np.maximum(np.maximum(boxes[:, 1], boxes[:, 3]), np.maximum(boxes[:, 5], boxes[:, 7]))*ratio

    min_x = np.maximum(min_x, np.zeros_like(min_x))
    max_x = np.minimum(max_x, np.full_like(max_x, score_map.shape[1]))
    min_y = np.maximum(min_y, np.zeros_like(min_y))
    max_y = np.minimum(max_y, np.full_like(max_y, score_map.shape[0]))

    # decode score_map to score_mask
    score_mask = decode_mask.decode_image_by_join(score_map)
    cluster_num = score_mask.max()
    score_sum = np.sum(score_mask > 0)

    # calc score_sum of each rect
    rect_mask = np.zeros((boxes.shape[0], score_map.shape[0], score_map.shape[1]), dtype=np.uint8)
    area_intersect = np.zeros((boxes.shape[0]))
    for i, rect in enumerate(rect_mask):
        rect[np.ix_(range(int(min_y[i]), int(max_y[i])), range(int(min_x[i]), int(max_x[i])))] = 1
        max_area_intersect = 0
        for cluster_idx in range(1, cluster_num+1):
            cluster_mask = score_mask == cluster_idx
            if np.sum(rect * cluster_mask) / score_sum > max_area_intersect:
                area_intersect[i] = np.sum(rect * cluster_mask) / score_sum
                max_area_intersect = area_intersect[i]

    # we want the intersect area between rect and score map become larger
    area_intersect_mean = np.mean(area_intersect)
    area_intersect = [math.floor(1/(1+np.exp(-50*(item-area_intersect_mean)))*1000)/1000 for item in area_intersect]

    return area_intersect


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list


    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        f_score, f_geometry = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images_icdar2015()

            for im_fn in im_fn_list:
                im = cv2.imread(im_fn)[:, :, ::-1]#将BGR格式转换为RGB格式
                start_time = time.time()
                # im_resized, (ratio_h, ratio_w) = resize_image(im)
                im_resized, (ratio_h, ratio_w) = resize_image(im, max_side_len=1280)#将图片成比例放缩，最大边不超过max_side_len，并且是32的倍数
                

                timer = {'net': 0, 'restore': 0, 'rescore': 0, 'nms': 0}
                
                start = time.time()
                score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
                timer['net'] = time.time() - start

                boxes, timer = detect(image=im_resized, score_map=score, geo_map=geometry, timer=timer,im_fn = im_fn)
                
                
                print('{} : net {:.0f}ms, restore {:.0f}ms, rescore {:.0f}ms, nms {:.0f}ms'.format(
                    im_fn, timer['net']*1000, timer['restore']*1000, timer['rescore']*1000, timer['nms']*1000))

                
                
                if boxes is not None:
                    scores = boxes[:, 8]
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h

                duration = time.time() - start_time
                print('[timing] {}'.format(duration))

                # add ground truth info
                if FLAGS.test_gt_path is not None:
                    gt_file = get_gt_txt(os.path.basename(im_fn).split('.')[0])#返回图片对应的gt文件路径
                    if not os.path.exists(gt_file):
                        print('text file {} does not exists'.format(gt_file))
                    else:
                        text_polys, text_tags = load_annoataion(gt_file)
                        for idx, box in enumerate(text_polys):
                            if not text_tags[idx]:
                                cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True,
                                              color=(0, 0, 255), thickness=2)#用红线把需要检测的文字区域框起来

                # save to txt file
                if not os.path.exists(os.path.join(FLAGS.output_dir, 'txt')):
                    os.mkdir(os.path.join(FLAGS.output_dir, 'txt'))
                if not os.path.exists(os.path.join(FLAGS.output_dir, 'img')):
                    os.mkdir(os.path.join(FLAGS.output_dir, 'img'))
                if FLAGS.dataset == 'icdar2015':
                    res_file = os.path.join(
                        FLAGS.output_dir, 'txt',
                        'res_{}.txt'.format(
                            os.path.basename(im_fn).split('.')[0]))
                elif FLAGS.dataset == 'icdar2017rctw':
                    res_file = os.path.join(
                        FLAGS.output_dir, 'txt',
                        'task1_{}.txt'.format(
                            os.path.basename(im_fn).split('.')[0]))
                else:
                    res_file = os.path.join(
                        FLAGS.output_dir, 'txt',
                        '{}.txt'.format(
                            os.path.basename(im_fn).split('.')[0]))

                with open(res_file, 'w') as f:
                    if boxes is not None:
                        for idx, box in enumerate(boxes):
                            # to avoid submitting errors
                            box = sort_poly(box.astype(np.int32))
                            # box = check_bbox(im, box)
                            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                continue
                            if FLAGS.dataset == 'icdar2015':
                                f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                                    box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0],
                                    box[3, 1],
                                ))
                            elif FLAGS.dataset == 'icdar2017rctw':
                                f.write('{},{},{},{},{},{},{},{},{:.3f}\r\n'.format(
                                    box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0],
                                    box[3, 1], scores[idx]
                                ))

                            cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True,
                                          color=(255, 255, 0), thickness=1)

                # write image
                if not FLAGS.no_write_images:
                    img_path = os.path.join(FLAGS.output_dir,'img', os.path.basename(im_fn))
                    cv2.imwrite(img_path, im[:, :, ::-1])

                ######################################
                # save to json file
                ######################################


                if DEBUG:
                    ######################################
                    # draw shrinked gt-box
                    ######################################
                    if FLAGS.test_gt_path is not None:
                        gt_file = get_gt_txt(os.path.basename(im_fn).split('.')[0])
                        if not os.path.exists(gt_file):
                            print('text file {} does not exists'.format(gt_file))
                        else:
                            text_polys, text_tags = load_annoataion(gt_file)
                            for idx, box in enumerate(text_polys):
                                if not text_tags[idx]:
                                    r = [None, None, None, None]
                                    for i in range(4):
                                        r[i] = min(np.linalg.norm(box[i] - box[(i + 1) % 4]),
                                                   np.linalg.norm(box[i] - box[(i - 1) % 4]))
                                    # shrink to 0.3
                                    shrinked_poly = shrink_poly(box.copy(), r).astype(np.int32)[np.newaxis, :, :]
                                    cv2.polylines(im[:, :, ::-1],
                                                  [shrinked_poly.astype(np.int32).reshape((-1, 1, 2))], True,
                                                  color=(0, 0, 255), thickness=1)

                    ######################################
                    # draw masked_image
                    ######################################
                    # im_resized.shape = (704, 1280, 3)
                    # score.shape = (1, 176, 320, 1)
                    if not FLAGS.no_write_images:
                        score_thresh = 0.8
                        score_mask = (score[0, :, :, 0] > score_thresh).astype(np.int32)*[255]
                        score_mask = cv2.resize(score_mask, (im.shape[1], im.shape[0]),
                                                interpolation=cv2.INTER_NEAREST)
                        score_map = np.expand_dims(score_mask, axis=-1)
                        score_map = np.concatenate((score_map, np.zeros_like(score_map), np.zeros_like(score_map)),
                                                   axis=-1).astype(np.int32)
                        based_im = im[:, :, ::-1].astype(np.int32)
                        masked_im = cv2.addWeighted(based_im, 0.5, score_map, 0.5, gamma=0)
                        score_path = os.path.join(FLAGS.output_dir, '%s_score.jpg' % os.path.basename(im_fn).split('.')[0])
                        cv2.imwrite(score_path, masked_im)

        if FLAGS.write_json:
            # need to delete data.json first
            path = '/workspace/mnt/group/general-reg/denglei/code/EAST/result2/txt/'
            f = os.listdir(path)

            result = {}

            for i in f:
                res = path + i
                count = i[7:].strip('.txt')
                key = 'res_' + count
                pic_box = []
                with open(res, 'r') as file:
                    while 1:
                        line = file.readline()
                        line = line.strip('\n')
                        if line != '':
                            # box = [[]]
                            box = []
                            list = line.split(',')
                            for i in range(int(len(list))):
                                box.append(int(list[i]))
                            box = np.array(box, dtype=np.int32).reshape(-1, 2)
                            box = box.tolist()
                            # print(box)
                            rbox = {'points': box,
                                    'confidence': 1}
                            pic_box.append(rbox)
                        if not line:
                            break
                result[key] = pic_box

            j_output = json.dumps(result, ensure_ascii=False)
            with open('/workspace/mnt/group/general-reg/denglei/code/EAST/calculate/LSVT_result_nms.json',
                      "w") as f:
                f.write(j_output)


if __name__ == '__main__':
    tf.app.run()
