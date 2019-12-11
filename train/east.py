def train_east(config_yaml):
    import sys
    sys.path.append('./detection_model/EAST')

    import time
    import numpy as np
    import tensorflow as tf
    from tensorflow.contrib import slim
    import cv2
    from yacs.config import CfgNode as CN
    import model
    import icdar

    def read_config_file(config_file):
        # 用yaml重构配置文件
        f = open(config_file)
        opt = CN.load_cfg(f)
        return opt

    # TODO 这里需要一些适配处理
    FLAGS = read_config_file(config_yaml)

    gpus = list(range(len(FLAGS.gpu_list.split(','))))

    def tower_loss(images, score_maps1, geo_maps1, training_masks1, score_maps2, geo_maps2, training_masks2,
                   reuse_variables=None):
        # Build inference graph
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
            f_score, f_geometry = model.model(images, is_training=True)

        model_loss1 = model.loss(score_maps1, f_score['F_score1'],
                                 geo_maps1, f_geometry['F_geometry1'],
                                 training_masks1)
        model_loss2 = model.loss(score_maps2, f_score['F_score2'],
                                 geo_maps2, f_geometry['F_geometry2'],
                                 training_masks2)

        model_loss = model_loss1 + model_loss2

        total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # add summary
        if reuse_variables is None:
            #         tf.summary.image('input', images)
            #         tf.summary.image('score_map', score_maps)
            #         tf.summary.image('score_map_pred', f_score * 255)
            #         tf.summary.image('geo_map_0', geo_maps[:, :, :, 0:1])
            #         tf.summary.image('geo_map_0_pred', f_geometry[:, :, :, 0:1])
            #         tf.summary.image('training_masks', training_masks)
            tf.summary.scalar('model_loss1', model_loss1)
            tf.summary.scalar('model_loss2', model_loss2)
            tf.summary.scalar('model_loss', model_loss)
            tf.summary.scalar('total_loss', total_loss)

        return total_loss, model_loss

    def average_gradients(tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)

            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    if not tf.gfile.Exists(FLAGS.checkpoint_path):
        tf.gfile.MkDir(FLAGS.checkpoint_path)
    else:
        if not FLAGS.restore:
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_path)
            tf.gfile.MkDir(FLAGS.checkpoint_path)

    input_images = tf.placeholder(tf.float32, shape=[None, 512, 512, 3], name='input_images')
    input_score_maps1 = tf.placeholder(tf.float32, shape=[None, 128, 128, 1], name='input_score_maps1')
    input_score_maps2 = tf.placeholder(tf.float32, shape=[None, 64, 64, 1], name='input_score_maps2')
    if FLAGS.geometry == 'RBOX':
        input_geo_maps1 = tf.placeholder(tf.float32, shape=[None, 128, 128, 5], name='input_geo_maps1')
        input_geo_maps2 = tf.placeholder(tf.float32, shape=[None, 64, 64, 5], name='input_geo_maps2')
    else:
        input_geo_maps1 = tf.placeholder(tf.float32, shape=[None, 128, 128, 8], name='input_geo_maps1')
        input_geo_maps2 = tf.placeholder(tf.float32, shape=[None, 64, 64, 8], name='input_geo_maps2')
    input_training_masks1 = tf.placeholder(tf.float32, shape=[None, 128, 128, 1], name='input_training_masks1')
    input_training_masks2 = tf.placeholder(tf.float32, shape=[None, 64, 64, 1], name='input_training_masks2')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=2000, decay_rate=0.94,
                                               staircase=True)
    # add summary
    tf.summary.scalar('learning_rate', learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)
    # opt = tf.train.MomentumOptimizer(learning_rate, 0.9)

    # split

    print('gpu', len(gpus))
    input_images_split = tf.split(input_images, len(gpus))
    input_score_maps_split1 = tf.split(input_score_maps1, len(gpus))
    input_geo_maps_split1 = tf.split(input_geo_maps1, len(gpus))
    input_training_masks_split1 = tf.split(input_training_masks1, len(gpus))

    input_score_maps_split2 = tf.split(input_score_maps2, len(gpus))
    input_geo_maps_split2 = tf.split(input_geo_maps2, len(gpus))
    input_training_masks_split2 = tf.split(input_training_masks2, len(gpus))

    tower_grads = []
    reuse_variables = None
    for i, gpu_id in enumerate(gpus):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('model_%d' % gpu_id) as scope:
                iis = input_images_split[i]
                isms1 = input_score_maps_split1[i]
                igms1 = input_geo_maps_split1[i]
                itms1 = input_training_masks_split1[i]

                isms2 = input_score_maps_split2[i]
                igms2 = input_geo_maps_split2[i]
                itms2 = input_training_masks_split2[i]
                total_loss, model_loss = tower_loss(iis, isms1, igms1, itms1, isms2, igms2, itms2, reuse_variables)

                batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                reuse_variables = True

                grads = opt.compute_gradients(total_loss)
                tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()
    # save moving average
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # batch norm updates
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_path, tf.get_default_graph())

    init = tf.global_variables_initializer()

    if FLAGS.pretrained_model_path is not None:
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path,
                                                             slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if FLAGS.restore:
            print('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            saver.restore(sess, ckpt)
        else:
            sess.run(init)
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)

        data_generator = icdar.get_batch(num_workers=FLAGS.num_readers,
                                         input_size=FLAGS.input_size,
                                         batch_size=FLAGS.batch_size_per_gpu * len(gpus))

        start = time.time()
        for step in range(FLAGS.max_steps):
            data = next(data_generator)

            #             print('hello:',data[2]['score_map1'][0].shape)
            #             print('hello:',data[2]['score_map2'][0].shape)
            #             print('hello:',data[3]['geo_map1'][0].shape)
            #             print('hello:',data[3]['geo_map2'][0].shape)

            # debug
            # import cv2
            #            print(type(data[0]))
            # cv2.imwrite('input.jpg', data[0][0])

            ml, tl, _ = sess.run([model_loss, total_loss, train_op], feed_dict={input_images: data[0],
                                                                                input_score_maps1: data[2][
                                                                                    'score_map1'],
                                                                                input_geo_maps1: data[3]['geo_map1'],
                                                                                input_training_masks1: data[4][
                                                                                    'training_mask1'],
                                                                                input_score_maps2: data[2][
                                                                                    'score_map2'],
                                                                                input_geo_maps2: data[3]['geo_map2'],
                                                                                input_training_masks2: data[4][
                                                                                    'training_mask2']})
            if np.isnan(tl):
                print('Loss diverged, stop training')
                break

            if step % 10 == 0:
                avg_time_per_step = (time.time() - start) / 10
                avg_examples_per_second = (10 * FLAGS.batch_size_per_gpu * len(gpus)) / (time.time() - start)
                start = time.time()
                print(
                    'Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, {:.2f} examples/second'.format(
                        step, ml, tl, avg_time_per_step, avg_examples_per_second))

            if step % FLAGS.save_checkpoint_steps == 0:
                saver.save(sess, FLAGS.checkpoint_path + 'model.ckpt', global_step=global_step)

            if step % FLAGS.save_summary_steps == 0:
                _, tl, summary_str = sess.run([train_op, total_loss, summary_op], feed_dict={input_images: data[0],
                                                                                             input_score_maps1: data[2][
                                                                                                 'score_map1'],
                                                                                             input_geo_maps1: data[3][
                                                                                                 'geo_map1'],
                                                                                             input_training_masks1:
                                                                                                 data[4][
                                                                                                     'training_mask1'],
                                                                                             input_score_maps2: data[2][
                                                                                                 'score_map2'],
                                                                                             input_geo_maps2: data[3][
                                                                                                 'geo_map2'],
                                                                                             input_training_masks2:
                                                                                                 data[4][
                                                                                                     'training_mask2']})
                summary_writer.add_summary(summary_str, global_step=step)