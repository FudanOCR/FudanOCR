def test_TextSnake(config_file):

    import sys
    sys.path.append('./detection_model/TextSnake_pytorch')

    import os
    import time
    import numpy as np
    import torch
    import json

    import torch.backends.cudnn as cudnn
    import torch.utils.data as data
    import torch.nn.functional as func

    from dataset.total_text import TotalText
    from network.textnet import TextNet
    from util.detection import TextDetector
    from util.augmentation import BaseTransform, EvalTransform
    from util.config import config as cfg, update_config, print_config
    from util.misc import to_device, fill_hole
    from util.option import BaseOptions
    from util.visualize import visualize_detection
    import cv2

    from yacs.config import CfgNode as CN

    def read_config_file(config_file):
        # 用yaml重构配置文件
        f = open(config_file)
        opt = CN.load_cfg(f)
        return opt

    opt = read_config_file(config_file)

    def result2polygon(image, result):
        """ convert geometric info(center_x, center_y, radii) into contours
        :param result: (list), each with (n, 3), 3 denotes (x, y, radii)
        :return: (np.ndarray list), polygon format contours
        """
        conts = []
        for instance in result:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for disk in instance:
                for x, y, r in disk:
                    cv2.circle(mask, (int(x), int(y)), int(r), (1), -1)

            cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(cont) > 0:
                # for item in cont:
                #     conts.append(item)
                conts.append(cont[0])

        conts = [cont[:, 0, :] for cont in conts]
        return conts


    def mask2conts(mask):
        conts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        conts = [cont[:, 0, :] for cont in conts]
        return conts


    def rescale_result(image, polygons, H, W):
        ori_H, ori_W = image.shape[:2]
        image = cv2.resize(image, (W, H))
        for polygon in polygons:
            cont = polygon['points']
            cont[:, 0] = (cont[:, 0] * W / ori_W).astype(int)
            cont[:, 1] = (cont[:, 1] * H / ori_H).astype(int)
            polygon['points'] = cont
        return image, polygons


    def rescale_padding_result(image, polygons, ori_h, ori_w):
        h, w = image.shape[:2]
        # get no-padding image size
        resize_h = ori_h if ori_h % 32 == 0 else (ori_h // 32) * 32
        resize_w = ori_w if ori_w % 32 == 0 else (ori_w // 32) * 32
        ratio = float(h) / resize_h if resize_h > resize_w else float(w) / resize_w
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        # crop no-padding image
        no_padding_image = image[0:resize_h, 0:resize_w, ::-1]
        no_padding_image = cv2.resize(no_padding_image, (ori_w, ori_h))

        # rescale points
        for polygon in polygons:
            polygon['points'][:, 0] = (polygon['points'][:, 0] * float(ori_w) / resize_w).astype(np.int32)
            polygon['points'][:, 1] = (polygon['points'][:, 1] * float(ori_h) / resize_h).astype(np.int32)

        return no_padding_image, polygons


    def calc_confidence(contours, score_map):
        polygons = []
        for cnt in contours:
            drawing = np.zeros(score_map.shape[1:], np.int8)
            mask = cv2.fillPoly(drawing, [cnt.astype(np.int32)], 1)
            area = np.sum(np.greater(mask, 0))
            if not area > 0:
                continue

            confidence = np.sum(mask * score_map[0]) / area

            polygon = {
                'points': cnt,
                'confidence': confidence
            }

            polygons.append(polygon)

        return polygons


    def load_model(model, model_path):
        print('Loading from {}'.format(model_path))
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict['model'])


    def inference(model, detector, test_loader):
        gt_json_path = os.path.join('/home/shf/fudan_ocr_system/datasets/', opt.dataset, 'train_labels.json')
        #gt_json_path = '/workspace/mnt/group/ocr/wangxunyan/maskscoring_rcnn/crop_train/crop_result_js.json'
        with open(gt_json_path, 'r') as f:
            gt_dict = json.load(f)

        model.eval()
        result = dict()

        for i, (img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta) in enumerate(test_loader):
            timer = {'model': 0, 'detect': 0, 'viz': 0, 'restore': 0}
            start = time.time()

            img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map = to_device(
                img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map)
            # inference
            output = model(img)
            if opt.multi_scale:
                size_h, size_w = img.shape[2:4]
                img_rescale = func.interpolate(img, scale_factor=0.5, mode='nearest')
                output_rescale = model(img_rescale)
                output_rescale = func.interpolate(output_rescale, size=(size_h, size_w), mode='nearest')

            timer['model'] = time.time()-start

            for idx in range(img.size(0)):
                start = time.time()
                print('detect {} / {} images: {}.'.format(i, len(test_loader), meta['image_id'][idx]))
                tr_pred = output[idx, 0:2].softmax(dim=0).data.cpu().numpy()
                tcl_pred = output[idx, 2:4].softmax(dim=0).data.cpu().numpy()
                sin_pred = output[idx, 4].data.cpu().numpy()
                cos_pred = output[idx, 5].data.cpu().numpy()
                radii_pred = output[idx, 6].data.cpu().numpy()

                # tr_pred_mask = 1 / (1 + np.exp(-12*tr_pred[1]+3))
                tr_pred_mask = np.where(tr_pred[1] > detector.tr_conf_thresh, 1, tr_pred[1])
                # tr_pred_mask = fill_hole(tr_pred_mask)

                tcl_pred_mask = (tcl_pred * tr_pred_mask)[1] > detector.tcl_conf_thresh

                if opt.multi_scale:
                    tr_pred_rescale = output_rescale[idx, 0:2].sigmoid().data.cpu().numpy()
                    tcl_pred_rescale = output_rescale[idx, 2:4].softmax(dim=0).data.cpu().numpy()

                    tr_pred_scale_mask = np.where(tr_pred_rescale[1] + tr_pred[1] > 1, 1, tr_pred_rescale[1] + tr_pred[1])
                    tr_pred_mask = tr_pred_scale_mask

                    # weighted adding
                    origin_ratio = 0.5
                    rescale_ratio = 0.5
                    tcl_pred = (tcl_pred * origin_ratio + tcl_pred_rescale * rescale_ratio).astype(np.float32)
                    tcl_pred_mask = (tcl_pred * tr_pred_mask)[1] > detector.tcl_conf_thresh

                batch_result = detector.complete_detect(tr_pred_mask, tcl_pred_mask, sin_pred, cos_pred, radii_pred)  # (n_tcl, 3)
                timer['detect'] = time.time()-start

                start = time.time()
                # visualization
                img_show = img[idx].permute(1, 2, 0).cpu().numpy()
                img_show = ((img_show * opt.stds + opt.means) * 255).astype(np.uint8)
                H, W = meta['Height'][idx].item(), meta['Width'][idx].item()

                # get pred_contours
                contours = result2polygon(img_show, batch_result)

                if opt.viz:
                    resize_H = H if H % 32 == 0 else (H // 32) * 32
                    resize_W = W if W % 32 == 0 else (W // 32) * 32

                    ratio = float(img_show.shape[0]) / resize_H if resize_H > resize_W else float(img_show.shape[1]) / resize_W
                    resize_H = int(resize_H * ratio)
                    resize_W = int(resize_W * ratio)

                    gt_info = gt_dict[int(meta['image_id'][idx].lstrip('gt_').rstrip('.jpg').split('_')[1])]

                    gt_contours = []
                    # for gt in gt_info:
                    #     if not gt['illegibility']:
                    #         gt_cont = np.array(gt['points'])
                    #         gt_cont[:, 0] = (gt_cont[:, 0] * float(resize_W) / W).astype(np.int32)
                    #         gt_cont[:, 1] = (gt_cont[:, 1] * float(resize_H) / H).astype(np.int32)
                    #         gt_contours.append(gt_cont)
                    gt_cont = np.array(gt_info['points'])
                    gt_cont[:, 0] = gt_cont[:, 0] * float(resize_W) / float(W)
                    gt_cont[:, 1] = gt_cont[:, 1] * float(resize_H) / float(H)
                    gt_contours.append(gt_cont.astype(np.int32))
                    illegal_contours = mask2conts(meta['illegal_mask'][idx].cpu().numpy())

                    predict_vis = visualize_detection(img_show, tr_pred_mask,
                                                      tcl_pred_mask.astype(np.uint8), contours.copy())
                    gt_vis = visualize_detection(img_show, tr_mask[idx].cpu().numpy(),
                                                 tcl_mask[idx].cpu().numpy(), gt_contours, illegal_contours)
                    im_vis = np.concatenate([predict_vis, gt_vis], axis=0)
                    path = os.path.join(opt.vis_dir, meta['image_id'][idx])
                    cv2.imwrite(path, im_vis)
                timer['viz'] = time.time()-start

                start = time.time()
                polygons = calc_confidence(contours, tr_pred)
                img_show, polygons = rescale_padding_result(img_show, polygons, H, W)

                # filter too small polygon
                for i, poly in enumerate(polygons):
                    if cv2.contourArea(poly['points']) < 100:
                        polygons[i] = []
                polygons = [item for item in polygons if item != []]

                # convert np.array to list
                for polygon in polygons:
                    polygon['points'] = polygon['points'].tolist()

                result[meta['image_id'][idx].replace('.jpg', '').replace('gt', 'res')] = polygons
                timer['restore'] = time.time()-start

            print('Cost time {:.2f}s: model {:.2f}s, detect {:.2f}s, viz {:.2f}s, restore {:.2f}s'.format(timer['model']+timer['detect']+timer['viz']+timer['restore'],
                                                                                                          timer['model'],
                                                                                                          timer['detect'],
                                                                                                          timer['viz'],
                                                                                                          timer['restore']))

        # write to json file
        with open(os.path.join(opt.output_dir, 'result.json'), 'w') as f:
            json.dump(result, f)
            print("Output json file in {}.".format(opt.output_dir))


    torch.cuda.set_device(opt.num_device)
    option = BaseOptions(config_file)
    args = option.initialize()

    update_config(opt, args)
    print_config(opt)
    data_root = os.path.join(opt.data_root, opt.dataset)
    testset = TotalText(
        data_root=data_root,
        ignore_list=os.path.join(data_root, 'ignore_list.txt'),
        is_training=False,
        transform=EvalTransform(size=1280, mean=opt.means, std=opt.stds)
        # transform=BaseTransform(size=1280, mean=opt.means, std=opt.stds)
    )
    test_loader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=opt.num_workers)

    # Model
    model = TextNet(backbone=opt.backbone, output_channel=7)
    model_path = os.path.join(opt.save_dir, opt.exp_name, \
                              'textsnake_{}_{}.pth'.format(model.backbone_name, opt.checkepoch))
    load_model(model, model_path)

    # copy to cuda
    model = model.to(opt.device)
    if opt.cuda:
        cudnn.benchmark = True
    detector = TextDetector(tcl_conf_thresh=0.3, tr_conf_thresh=1.0)  # 0.3

    # check vis_dir and output_dir exist
    if opt.viz:
        if not os.path.exists(opt.vis_dir):
            os.mkdir(opt.vis_dir)
    if not os.path.exists(opt.output_dir):
        os.mkdir(opt.output_dir)

    print('Start testing TextSnake.')

    inference(model, detector, test_loader)

    print('End.')

