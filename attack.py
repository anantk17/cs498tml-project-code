import numpy as np
import tensorflow as tf
import model

def optimize_linear(grad, eps, ord=np.inf):
    #optimal_perturbation = None
    if ord == np.inf:
        optimal_perturbation = tf.sign(grad)
        optimal_perturbation = tf.stop_gradient(optimal_perturbation)
    else:
        raise NotImplementedError("ord {} not supported".format(ord))

    scaled_perturbation = tf.multiply(eps, optimal_perturbation)
    return scaled_perturbation

def fgsm(x, logits, exp_config, kwargs = dict()):

    eps = kwargs['eps']
    ord = kwargs['ord']
    #get predictions from logits -> here we get one prediction for each of the pixels
    #assert logits.op.type != 'Softmax'

    preds_max = tf.reduce_max(logits, axis = 3, keepdims = True,name=None,reduction_indices=None)
    print("PREDS_MAX_SHAPE", preds_max.shape)
    y = tf.to_float(tf.equal(logits, preds_max))
    print("Y_SHAPE", y.shape)
    y = tf.stop_gradient(y)
    print("Y_SHAPE", y.shape)
    y = tf.reduce_sum(y, axis = 3, name=None,reduction_indices=None)
    print("Y_SHAPE", y.shape)
    y = tf.to_int32(y)
    print("Y_SHAPE, LOGITS_SHAPE", y.shape, logits.shape)

    #compute loss
    loss = model.loss(logits, y,
                    nlabels=int(exp_config.nlabels), 
                    loss_type=exp_config.loss_type, 
                    weight_decay=exp_config.weight_decay)
    
    grad, = tf.gradients(loss, x)
    assert grad is not None
    optimal_perturbation = optimize_linear(grad,eps,ord)

    adv_x = x + optimal_perturbation

    return adv_x 

def pgd(input_x, logits, kwargs = dict()):
    pass

def smoothed_pgd(input_x, logits, kwargs = dict()):
    pass

def adaptive_mask(input_x, logits, kwargs = dict()):
    pass