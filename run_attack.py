from importlib.machinery import SourceFileLoader
import config.system as sys_config
import os
import glob
import logging
import utils
import tensorflow as tf
import model as model
import image_utils
import argparse
from skimage import transform
import numpy as np
import attack as adv_attack
import matplotlib.pyplot as plt
import acdc_data
import train
from background_generator import BackgroundGenerator

np.random.seed(0)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

sys_config.setup_GPU_environment()

ATTACKS = ['fgsm','pgd']

def generate_adversarial_examples(input_folder, output_path, model_path, attack, attack_args, exp_config):
    nx, ny = exp_config.image_size[:2]
    batch_size = 1
    num_channels = exp_config.nlabels

    image_tensor_shape = [batch_size] + list(exp_config.image_size) + [1]
    images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')

    logits_pl = model.inference(images_pl,exp_config=exp_config, training=tf.constant(False, dtype=tf.bool))

    data = acdc_data.load_and_maybe_process_data(
        input_folder=sys_config.data_root,
        preprocessing_folder=sys_config.preproc_folder,
        mode=exp_config.data_mode,
        size=exp_config.image_size,
        target_resolution=exp_config.target_resolution,
        force_overwrite=False,
        split_test_train=True
    )

    images = data['images_test']
    labels = data['masks_test']
    
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        checkpoint_path = utils.get_latest_model_checkpoint_path(model_path,'model_best_dice.ckpt')
        saver.restore(sess, checkpoint_path)

        for batch in BackgroundGenerator(train.iterate_minibatches(images,labels,1)):
            x,y = batch

            if attack == 'fgsm':
                adv_x = adv_attack.fgsm(x,y, images_pl, logits_pl, exp_config, sess, attack_args)
            elif attack == 'pgd':
                adv_x = adv_attack.pgd(x,y,images_pl, logits_pl, exp_config, sess, attack_args )
            else:
                raise NotImplementedError

            non_adv_mask_out = sess.run([tf.arg_max(tf.nn.softmax(logits_pl), dimension=-1)], feed_dict={images_pl : x})
            adv_mask_out = sess.run([tf.arg_max(tf.nn.softmax(logits_pl), dimension=-1)], feed_dict={images_pl : adv_x})

            fig = plt.figure()
            ax1 = fig.add_subplot(241)
            ax1.imshow(np.squeeze(x), cmap='gray')
            ax5 = fig.add_subplot(242)
            ax5.imshow(np.squeeze(adv_x), cmap='gray')
            ax2 = fig.add_subplot(243)
            ax2.imshow(np.squeeze(y))
            ax3 = fig.add_subplot(244)
            ax3.imshow(np.squeeze(non_adv_mask_out))
            ax4 = fig.add_subplot(245)
            ax4.imshow(np.squeeze(adv_mask_out))
            plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to evaluate a neural network model on the ACDC challenge data")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment folder (assuming you are in the working directory)")
    parser.add_argument("ATTACK",type=str,help="Algorithm to generate adversarial examples", choices=ATTACKS)
    args = parser.parse_args()
    
    #Setup model configuration
    base_path = sys_config.project_root
    model_path = os.path.join(base_path, args.EXP_PATH)
    config_file = glob.glob(model_path + '/*py')[0]
    config_module = config_file.split('/')[-1].rstrip('.py')
    exp_config = SourceFileLoader(fullname=config_module, path=os.path.join(config_file)).load_module()
 
    logging.warning("GENERATING EXAMPLES FOR TESTING SET")
    
    #Setup input and output paths
    input_path = sys_config.test_data_root
    output_path = os.path.join(model_path,'adversarial_examples_'+ args.ATTACK)
    image_path = os.path.join(output_path, 'image')
    diff_path = os.path.join(output_path,'difference')
    utils.makefolder(image_path)
    utils.makefolder(diff_path)

    attack_args = {'eps' : 0.3, 'ord' : np.inf, 'epochs' : 10}

    generate_adversarial_examples(input_path, 
                                  output_path,
                                  model_path,
                                  attack=args.ATTACK,
                                  attack_args=attack_args,
                                  exp_config=exp_config)



