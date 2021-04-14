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

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

sys_config.setup_GPU_environment()

ATTACKS = ['fgsm']

def generate_adversarial_examples(input_folder, output_path, model_path, attack, attack_args, exp_config):
    nx, ny = exp_config.image_size[:2]
    batch_size = 1
    num_channels = exp_config.nlabels

    image_tensor_shape = [batch_size] + list(exp_config.image_size) + [1]
    images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')

    logits_pl = model.inference(images_pl,exp_config,training=tf.constant(False, dtype=tf.bool))#exp_config.model_handle(images_pl, tf.constant(False, dtype=tf.bool), nlabels=exp_config.nlabels)
    #mask_pl, softmax_pl = model.predict(images_pl, exp_config)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        checkpoint_path = utils.get_latest_model_checkpoint_path(model_path,'model_best_dice.ckpt')
        saver.restore(sess, checkpoint_path)

        for folder in os.listdir(input_folder):
            folder_path = os.path.join(input_folder, folder)

            if os.path.isdir(folder_path):
                infos = {}
                for line in open(os.path.join(folder_path, 'Info.cfg')):
                    label, value = line.split(':')
                    infos[label] = value.rstrip('\n').lstrip(' ')

                patient_id = folder.lstrip('patient')
                ED_frame = int(infos['ED'])
                ES_frame = int(infos['ES'])

                for file in glob.glob(os.path.join(folder_path, 'patient???_frame??.nii.gz')):
                    
                    logging.info(' ----- Doing image: -------------------------')
                    logging.info('Doing: %s' % file)
                    logging.info(' --------------------------------------------')

                    file_base = file.split('.nii.gz')[0]

                    frame = int(file_base.split('frame')[-1])
                    img_dat = utils.load_nii(file)
                    img = img_dat[0].copy()
                    img = image_utils.normalise_image(img)

                    #Assuming image is 2D
                    pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
                    scale_vector = (pixel_size[0] / exp_config.target_resolution[0],
                                    pixel_size[1] / exp_config.target_resolution[1])
                    predictions = []

                    for zz in range(img.shape[2]):

                        slice_img = np.squeeze(img[:,:,zz])
                        slice_rescaled = transform.rescale(slice_img,
                                                        scale_vector,
                                                        order=1,
                                                        preserve_range=True,
                                                        multichannel=False,
                                                        anti_aliasing=True,
                                                        mode='constant')

                        x, y = slice_rescaled.shape

                        x_s = (x - nx) // 2
                        y_s = (y - ny) // 2
                        x_c = (nx - x) // 2
                        y_c = (ny - y) // 2

                        # Crop section of image for prediction
                        if x > nx and y > ny:
                            slice_cropped = slice_rescaled[x_s:x_s+nx, y_s:y_s+ny]
                        else:
                            slice_cropped = np.zeros((nx,ny))
                            if x <= nx and y > ny:
                                slice_cropped[x_c:x_c+ x, :] = slice_rescaled[:,y_s:y_s + ny]
                            elif x > nx and y <= ny:
                                slice_cropped[:, y_c:y_c + y] = slice_rescaled[x_s:x_s + nx, :]
                            else:
                                slice_cropped[x_c:x_c+x, y_c:y_c + y] = slice_rescaled[:, :]

                        network_input = np.float32(np.tile(np.reshape(slice_cropped, (nx, ny, 1)), (batch_size, 1, 1, 1)))
                        #nw_ip2 = network_input.copy()
                        logits_out = sess.run([logits_pl], feed_dict={images_pl: network_input})
                        prediction_cropped = logits_out[0]
                        print("Type check",type(network_input), type(prediction_cropped))
                        print("Shape check", network_input.shape, prediction_cropped.shape)
                        adversarial_output = None

                        if attack == 'fgsm':
                            adversarial_output = adv_attack.fgsm(network_input,prediction_cropped,exp_config,attack_args)
                        else:
                            raise NotImplementedError

                        if frame == ED_frame:
                            frame_suffix = '_ED'
                        elif frame == ES_frame:
                            frame_suffix = '_ES'
                        else:
                            raise ValueError('Frame doesnt correspond to ED or ES. frame = %d, ED = %d, ES = %d' %
                                             (frame, ED_frame, ES_frame))
                        
                        adv_image_file_name = os.path.join(output_path,'image','patient' + patient_id + frame_suffix + '.nii.gz')
                        utils.save_nii(adv_image_file_name, adversarial_output, img_dat[1], img_dat[2])

                        fig = plt.figure()
                        ax1 = fig.add_subplot(241)
                        ax1.imshow(np.squeeze(img), cmap='gray')
                        ax2 = fig.add_subplot(242)
                        ax2.imshow(np.squeeze(adversarial_output), cmap='gray')
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

    attack_args = {'eps' : 0.3, 'ord' : np.inf}

    generate_adversarial_examples(input_path, 
                                  output_path,
                                  model_path,
                                  attack=args.ATTACK,
                                  attack_args=attack_args,
                                  exp_config=exp_config)



