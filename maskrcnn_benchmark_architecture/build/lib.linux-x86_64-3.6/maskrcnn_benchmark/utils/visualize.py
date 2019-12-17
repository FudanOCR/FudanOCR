import numpy as np
from PIL import Image, ImageDraw


def vis_image(img, boxes, cls_prob=None, mode=0):
    # img = cv2.imread(image_path)
    # cv2.setMouseCallback("image", trigger)

    draw = ImageDraw.Draw(img)

    for idx in range(len(boxes)):
        cx, cy, w, h, angle = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]
        # need a box score larger than thresh

        lt = [cx - w / 2, cy - h / 2, 1]
        rt = [cx + w / 2, cy - h / 2, 1]
        lb = [cx - w / 2, cy + h / 2, 1]
        rb = [cx + w / 2, cy + h / 2, 1]

        pts = []

        pts.append(lt)
        pts.append(rt)
        pts.append(rb)
        pts.append(lb)

        angle = -angle

        cos_cita = np.cos(np.pi / 180 * angle)
        sin_cita = np.sin(np.pi / 180 * angle)

        M0 = np.array([[1, 0, 0], [0, 1, 0], [-cx, -cy, 1]])
        M1 = np.array([[cos_cita, sin_cita, 0], [-sin_cita, cos_cita, 0], [0, 0, 1]])
        M2 = np.array([[1, 0, 0], [0, 1, 0], [cx, cy, 1]])
        rotation_matrix = M0.dot(M1).dot(M2)

        rotated_pts = np.dot(np.array(pts), rotation_matrix)

        # rotated_pts[rotated_pts <= 0] = 1
        # rotated_pts[rotated_pts > img.shape[1]] = img.shape[1] - 1

        if mode == 1:
            draw.line((int(rotated_pts[0, 0]), int(rotated_pts[0, 1]), int(rotated_pts[1, 0]), int(rotated_pts[1, 1])), fill=(0, 255, 0))
            draw.line((int(rotated_pts[1, 0]), int(rotated_pts[1, 1]), int(rotated_pts[2, 0]), int(rotated_pts[2, 1])),
                      fill=(0, 255, 0))
            draw.line((int(rotated_pts[2, 0]), int(rotated_pts[2, 1]), int(rotated_pts[3, 0]), int(rotated_pts[3, 1])),
                      fill=(0, 255, 0))
            draw.line((int(rotated_pts[3, 0]), int(rotated_pts[3, 1]), int(rotated_pts[0, 0]), int(rotated_pts[0, 1])),
                      fill=(0, 255, 0))

        elif mode == 0:
            draw.line((int(rotated_pts[0, 0]), int(rotated_pts[0, 1]), int(rotated_pts[1, 0]), int(rotated_pts[1, 1])),
                      fill=(0, 0, 255))
            draw.line((int(rotated_pts[1, 0]), int(rotated_pts[1, 1]), int(rotated_pts[2, 0]), int(rotated_pts[2, 1])),
                      fill=(0, 0, 255))
            draw.line((int(rotated_pts[2, 0]), int(rotated_pts[2, 1]), int(rotated_pts[3, 0]), int(rotated_pts[3, 1])),
                      fill=(0, 0, 255))
            draw.line((int(rotated_pts[3, 0]), int(rotated_pts[3, 1]), int(rotated_pts[0, 0]), int(rotated_pts[0, 1])),
                      fill=(0, 0, 255))

        elif mode == 2:
            draw.line((int(rotated_pts[0, 0]), int(rotated_pts[0, 1]), int(rotated_pts[1, 0]), int(rotated_pts[1, 1])),
                      fill=(255, 255, 0))
            draw.line((int(rotated_pts[1, 0]), int(rotated_pts[1, 1]), int(rotated_pts[2, 0]), int(rotated_pts[2, 1])),
                      fill=(255, 255, 0))
            draw.line((int(rotated_pts[2, 0]), int(rotated_pts[2, 1]), int(rotated_pts[3, 0]), int(rotated_pts[3, 1])),
                      fill=(255, 255, 0))
            draw.line((int(rotated_pts[3, 0]), int(rotated_pts[3, 1]), int(rotated_pts[0, 0]), int(rotated_pts[0, 1])),
                      fill=(255, 255, 0))
        '''
        if not cls_prob is None:
            score = cls_prob[idx]
            if cx < img.shape[1] and cy < img.shape[0] and cx > 0 and cy > 0:
                cv2.putText(img, str(score), (int(cx), int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 122), 2)
        '''
    # cv2.imshow("image", cv2.resize(img, (1024, 768)))
    # cv2.wait Key(0)
    del draw
    return img
