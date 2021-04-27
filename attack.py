import numpy as np
import tensorflow as tf
import model


def fgsm_run(x, y, images_pl, labels_pl, logits_pl, exp_config, sess, kwargs=dict()):
    eps = kwargs['eps']
    loss = model.loss(logits_pl,
                      labels_pl,
                      nlabels=exp_config.nlabels,
                      loss_type=exp_config.loss_type,
                      weight_decay=exp_config.weight_decay)

    grad_pl, = tf.gradients(loss, images_pl)

    grad = sess.run([grad_pl], feed_dict={images_pl: x, labels_pl: y})[0]

    assert grad is not None

    adv_x = x + eps * np.sign(grad)

    return adv_x


def pgd(x, y, images_pl, labels_pl, logits_pl, exp_config, sess, kwargs=dict()):
    epochs = kwargs['epochs']
    alpha = kwargs['alpha']
    eps = kwargs['eps']

    loss = model.loss(logits_pl,
                      labels_pl,
                      nlabels=exp_config.nlabels,
                      loss_type=exp_config.loss_type,
                      weight_decay=exp_config.weight_decay)

    grad_pl, = tf.gradients(loss, images_pl)

    X_adv = x.copy()
    for i in range(epochs):
        grad = sess.run([grad_pl], feed_dict={images_pl: x, labels_pl: y})[0]
        added = np.sign(grad)
        step_output = alpha * added
        X_adv = X_adv + np.clip(step_output, -eps, eps)

    return X_adv


def pgd_conv(x, y, images_pl, labels_pl, logits_pl, exp_config, sess, eps=None, step_alpha=None, num_steps=None, sizes=None,
             weights=None):
    mask_tensor_shape = [1] + list(exp_config.image_size)

    # compute loss
    loss = model.loss(logits_pl,
                      labels_pl,
                      nlabels=exp_config.nlabels,
                      loss_type=exp_config.loss_type,
                      weight_decay=exp_config.weight_decay)

    crafting_input = x.copy()
    crafting_output = crafting_input
    # crafting_target = y.copy()
    for i in range(num_steps):
        grad_pl, = tf.gradients(loss, images_pl)
        grad = sess.run([grad_pl], feed_dict={images_pl: crafting_input,
                                              labels_pl: y})[0]
        assert grad is not None
        added = np.sign(grad)
        step_output = crafting_input + step_alpha * added
        total_adv = step_output - x
        total_adv = np.clip(total_adv, -eps, eps)
        crafting_output = x + total_adv
        crafting_input = crafting_output

    added = crafting_output - x
    print('PDG DONE')

    for i in range(num_steps * 2):
        temp = tf.nn.conv2d(input=added, filter=weights[0], padding='SAME', data_format='NHWC')
        for j in range(len(sizes) - 1):
            temp = temp + tf.nn.conv2d(input=added, filter=weights[j + 1], padding='SAME', data_format='NHWC')

        temp = temp / float(len(sizes))  # average over multiple convolutions

        temp = temp.eval(session=sess)

        grad_pl, = tf.gradients(loss, images_pl)
        grad = sess.run([grad_pl], feed_dict={images_pl: x + temp,
                                              labels_pl: y})[0]
        assert grad is not None
        added = added + step_alpha * np.sign(grad)
        added = np.clip(added, -eps, eps)

    print('SMOOTH PGD1 DONE')

    temp = tf.nn.conv2d(input=added, filter=weights[0], padding='SAME', data_format='NHWC')
    for j in range(len(sizes) - 1):
        temp = temp + tf.nn.conv2d(input=added, filter=weights[j + 1], padding='SAME', data_format='NHWC')
    temp = temp / float(len(sizes))
    temp = temp.eval(session=sess)
    crafting_output = x + temp

    print('SMOOTH PGD2 DONE')

    return crafting_output


def add_gaussian_noise(x, adv_x, sess, num_of_trials=5, eps=None, sizes=None, weights=None):
    crafting_outputs = []
    for k in range(num_of_trials):
        crafted_input = adv_x.copy()

        crafted_input = crafted_input + np.random.randn(crafted_input.shape[0],
                                                        crafted_input.shape[1],
                                                        crafted_input.shape[2],
                                                        crafted_input.shape[3]) * np.random.randint(1, 10)  # add noise
        added = crafted_input - x
        # print(type(added), type(weights[0]))
        # print(x.shape, adv_x.shape, crafted_input.shape, added.shape, weights[0].shape)
        temp = tf.nn.conv2d(input=added, filter=tf.cast(weights[0], dtype=tf.float64), padding='SAME',
                            data_format='NHWC')
        for j in range(len(sizes) - 1):
            temp = temp + tf.nn.conv2d(input=added, filter=tf.cast(weights[j + 1], dtype=tf.float64), padding='SAME',
                                       data_format='NHWC')

        temp = temp / float(len(sizes))  # average over multiple convolutions

        temp = temp.eval(session=sess)

        total_adv = np.clip(temp, -eps, eps)
        crafting_output = x + total_adv
        crafting_outputs.append(crafting_output)

    return crafting_outputs


def smoothed_pgd(input_x, logits, kwargs=dict()):
    pass


def adaptive_mask(input_x, logits, kwargs=dict()):
    pass
