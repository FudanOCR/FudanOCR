from engine.trainer import Trainer
class TextSnake_Trainer(Trainer):

    def __init__(self, modelObject, opt, train_loader, val_loader):
        Trainer.__init__(self, modelObject, opt, train_loader, val_loader)
        from model.detection_model.TextSnake_pytorch.util import global_data
        global_data._init()

    def to_device(self, *tensors):
        return (t.to(self.opt.TEXTSNAKE.device) for t in tensors)

    def pretreatment(self, data, test=False):

        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = data
        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map = self.to_device(
            img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map)

        return (img,)

    def posttreatment(self, modelResult, pretreatmentData, originData, test=False):
        if test == False:
            img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = originData
            img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map = self.to_device(
                img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map)
            tr_loss, tcl_loss, sin_loss, cos_loss, radii_loss = \
                self.criterion(modelResult, tr_mask, tcl_mask, sin_map, cos_map, radius_map, train_mask)
            loss = tr_loss + tcl_loss + sin_loss + cos_loss + radii_loss
            return loss

        else:
            import numpy as np
            import cv2
            import json
            from PIL import Image
            from model.detection_model.TextSnake_pytorch.util.detection import TextDetector
            from model.detection_model.TextSnake_pytorch.util.visualize import visualize_detection
            from model.detection_model.TextSnake_pytorch.util import global_data

            Image.MAX_IMAGE_PIXELS = None
            result = dict()
            detector = TextDetector(tcl_conf_thresh=0.3, tr_conf_thresh=1.0)
            img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = originData
            img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map = self.to_device(
                img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map)
            tr_loss, tcl_loss, sin_loss, cos_loss, radii_loss = \
                self.criterion(modelResult, tr_mask, tcl_mask, sin_map, cos_map, radius_map, train_mask)
            loss = tr_loss + tcl_loss + sin_loss + cos_loss + radii_loss
            for idx in range(img.size(0)):
                # print('detect {} / {} images: {}.'.format(i, len(test_loader), meta['image_id'][idx]))
                tr_pred = modelResult[idx, 0:2].softmax(dim=0).data.cpu().numpy()
                tcl_pred = modelResult[idx, 2:4].softmax(dim=0).data.cpu().numpy()
                sin_pred = modelResult[idx, 4].data.cpu().numpy()
                cos_pred = modelResult[idx, 5].data.cpu().numpy()
                radii_pred = modelResult[idx, 6].data.cpu().numpy()

                # tr_pred_mask = 1 / (1 + np.exp(-12*tr_pred[1]+3))
                tr_pred_mask = np.where(tr_pred[1] > detector.tr_conf_thresh, 1, tr_pred[1])
                # tr_pred_mask = fill_hole(tr_pred_mask)

                tcl_pred_mask = (tcl_pred * tr_pred_mask)[1] > detector.tcl_conf_thresh

                batch_result = detector.complete_detect(tr_pred_mask, tcl_pred_mask, sin_pred, cos_pred,
                                                        radii_pred)  # (n_tcl, 3)
                # visualization
                img_show = img[idx].permute(1, 2, 0).cpu().numpy()
                img_show = ((img_show * self.opt.TEXTSNAKE.stds + self.opt.TEXTSNAKE.means) * 255).astype(np.uint8)
                H, W = meta['Height'][idx].item(), meta['Width'][idx].item()

                # get pred_contours
                contours = []
                for instance in batch_result:
                    mask = np.zeros(img_show.shape[:2], dtype=np.uint8)
                    for disk in instance:
                        for x, y, r in disk:
                            cv2.circle(mask, (int(x), int(y)), int(r), (1), -1)

                    cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if len(cont) > 0:
                        # for item in cont:
                        #     conts.append(item)
                        contours.append(cont[0])

                contours = [cont[:, 0, :] for cont in contours]

                polygons = []
                for cnt in contours:
                    drawing = np.zeros(tr_pred.shape[1:], np.int8)
                    mask = cv2.fillPoly(drawing, [cnt.astype(np.int32)], 1)
                    area = np.sum(np.greater(mask, 0))
                    if not area > 0:
                        continue

                    confidence = np.sum(mask * tr_pred[0]) / area

                    polygon = {
                        'points': cnt,
                        'confidence': confidence
                    }

                    polygons.append(polygon)

                h, w = img_show.shape[:2]
                # get no-padding image size
                resize_h = H if H % 32 == 0 else (H // 32) * 32
                resize_w = W if W % 32 == 0 else (W // 32) * 32
                ratio = float(h) / resize_h if resize_h > resize_w else float(w) / resize_w
                resize_h = int(resize_h * ratio)
                resize_w = int(resize_w * ratio)

                # crop no-padding image
                no_padding_image = img_show[0:resize_h, 0:resize_w, ::-1]
                no_padding_image = cv2.resize(no_padding_image, (W, H))

                # rescale points
                for polygon in polygons:
                    polygon['points'][:, 0] = (polygon['points'][:, 0] * float(W) / resize_w).astype(np.int32)
                    polygon['points'][:, 1] = (polygon['points'][:, 1] * float(H) / resize_h).astype(np.int32)

                img_show = no_padding_image

                # filter too small polygon
                for i, poly in enumerate(polygons):
                    if cv2.contourArea(poly['points']) < 100:
                        polygons[i] = []
                polygons = [item for item in polygons if item != []]

                # convert np.array to list
                for polygon in polygons:
                    polygon['points'] = polygon['points'].tolist()

                result[meta['image_id'][idx].replace('.jpg', '').replace('gt', 'res')] = polygons

            global_data._update(result)

            # print("Output json file in {}.".format(self.opt.ADDRESS.OUTPUT_DIR))
            return loss

    def res2json(self):
        import os
        from model.detection_model.TextSnake_pytorch.util import global_data
        global_data._reset()

        result_dir = self.opt.ADDRESS.DET_RESULT_DIR
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        jpath = os.path.join(result_dir, 'result.json')
        if os.path.isfile(jpath):
            os.remove(jpath)
        return jpath