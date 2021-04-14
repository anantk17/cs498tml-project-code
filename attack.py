import numpy as np
import tensorflow as tf
import model

def fgsm(x, y, images_pl, logits_pl, exp_config, sess, kwargs = dict()):

    eps = kwargs['eps']

    mask_tensor_shape = [1] + list(exp_config.image_size)
    
    labels_pl = tf.placeholder(tf.uint8, shape=mask_tensor_shape)
    
    #compute loss
    loss = model.loss(logits_pl,
                    labels_pl,
                    nlabels=exp_config.nlabels, 
                    loss_type=exp_config.loss_type, 
                    weight_decay=exp_config.weight_decay)
    
    grad_pl, = tf.gradients(loss, images_pl)
    grad = sess.run([grad_pl], feed_dict = {images_pl : x, labels_pl : y})[0]

    assert grad is not None

    adv_x = x + eps * np.sign(grad)

    return adv_x

def pgd(input_x, logits, kwargs = dict()):
    pass

def smoothed_pgd(input_x, logits, kwargs = dict()):
    pass

def adaptive_mask(input_x, logits, kwargs = dict()):
    pass