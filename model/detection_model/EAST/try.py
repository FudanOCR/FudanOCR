import cv2
import tensorflow as tf
import numpy as np

print('hello')
path = '/workspace/mnt/group/general-reg/denglei/code/EAST/input.jpg'
path1 = '/workspace/mnt/group/general-reg/denglei/code/EAST/temp/'
img = cv2.imread(path)
print(img.shape)
cv2.imwrite(path1 + "resource.jpg", img)
#img = np.expand_dims(img, 0)
# adjust_brightness
bright_img = tf.image.adjust_brightness(img, delta=.5)
bright_img = tf.squeeze(bright_img)
with tf.Session() as sess:
    # 运行 'init' op
    result = sess.run(bright_img)
result = np.uint8(result)

rand_image = tf.image.random_brightness(img, max_delta=.5)
rand_image = tf.squeeze(rand_image)
with tf.Session() as sess:
    # 运行 'init' op
    result2 = sess.run(rand_image)
result2 = np.uint8(result2)

cv2.imwrite(path1 + "result.jpg", result)
cv2.imwrite(path1 + "result2.jpg", result2)
cv2.waitKey(0)
cv2.destroyAllWindows()



