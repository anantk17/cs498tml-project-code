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
import json
import copy

np.random.seed(0)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

sys_config.setup_GPU_environment()

ATTACKS = ['fgsm', 'pgd', 'spgd']


def gaussian_kernel(size: int,
                    mean: float,
                    std: float,
                    ):
    """Makes 2D gaussian Kernel for convolution."""

    d = tf.distributions.Normal(mean, std)

    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))

    gauss_kernel = tf.einsum('i,j->ij', vals, vals)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)


def generate_adversarial_examples(input_folder, output_path, model_path, attack, attack_args, exp_config,
                                  add_gaussian=False):
    nx, ny = exp_config.image_size[:2]
    batch_size = 1
    num_channels = exp_config.nlabels

    image_tensor_shape = [batch_size] + list(exp_config.image_size) + [1]
    mask_tensor_shape = [batch_size] + list(exp_config.image_size)
    images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')
    labels_pl = tf.placeholder(tf.uint8, shape=mask_tensor_shape, name='labels')
    logits_pl = model.inference(images_pl, exp_config=exp_config, training=tf.constant(False, dtype=tf.bool))
    eval_loss = model.evaluation(logits_pl, labels_pl, images_pl, nlabels=exp_config.nlabels,
                                 loss_type=exp_config.loss_type)

    data = acdc_data.load_and_maybe_process_data(
        input_folder=sys_config.data_root,
        preprocessing_folder=sys_config.preproc_folder,
        mode=exp_config.data_mode,
        size=exp_config.image_size,
        target_resolution=exp_config.target_resolution,
        force_overwrite=False,
        split_test_train=True
    )

    images = data['images_test'][:20]
    labels = data['masks_test'][:20]

    print("Num images train {} test {}".format(len(data['images_train']), len(images)))

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    baseline_closs = 0.0
    baseline_cdice = 0.0
    attack_closs = 0.0
    attack_cdice = 0.0
    l2_diff_sum = 0.0
    ln_diff_sum = 0.0
    ln_diff = 0.0
    l2_diff = 0.0
    batches = 0
    result_dict = []

    with tf.Session() as sess:
        results = []
        sess.run(init)
        checkpoint_path = utils.get_latest_model_checkpoint_path(model_path, 'model_best_dice.ckpt')
        saver.restore(sess, checkpoint_path)

        for batch in BackgroundGenerator(train.iterate_minibatches(images, labels, batch_size)):
            x, y = batch
            batches += 1

            if batches != 9:
              continue
            
            non_adv_mask_out = sess.run([tf.arg_max(tf.nn.softmax(logits_pl), dimension=-1)], feed_dict={images_pl: x})

            if attack == 'fgsm':
                adv_x = adv_attack.fgsm_run(x, y, images_pl, labels_pl, logits_pl, exp_config, sess, attack_args)
            elif attack == 'pgd':
                adv_x = adv_attack.pgd(x, y, images_pl, labels_pl, logits_pl, exp_config, sess, attack_args)
            elif attack == 'spgd':
                adv_x = adv_attack.pgd_conv(x, y, images_pl, labels_pl, logits_pl, exp_config, sess, **attack_args)
            else:
                raise NotImplementedError
            adv_x = [adv_x]

            if add_gaussian:
                print('adding gaussian noise')
                adv_x = adv_attack.add_gaussian_noise(x, adv_x[0], sess, eps=attack_args['eps'],
                                                      sizes=attack_args['sizes'], weights=attack_args['weights'])

            for i in range(len(adv_x)):
                l2_diff = np.average(np.squeeze(np.linalg.norm(adv_x[i] - x, axis=(1, 2))))
                ln_diff = np.average(np.squeeze(np.linalg.norm(adv_x[i] - x, axis=(1, 2), ord=np.inf)))

                l2_diff_sum += l2_diff
                ln_diff_sum += ln_diff

                print(l2_diff, l2_diff)

                adv_mask_out = sess.run([tf.arg_max(tf.nn.softmax(logits_pl), dimension=-1)],
                                        feed_dict={images_pl: adv_x[i]})

                closs, cdice = sess.run(eval_loss, feed_dict={images_pl: x, labels_pl: y})
                baseline_closs = closs + baseline_closs
                baseline_cdice = cdice + baseline_cdice

                adv_closs, adv_cdice = sess.run(eval_loss, feed_dict={images_pl: adv_x[i], labels_pl: y})
                attack_closs = adv_closs + attack_closs
                attack_cdice = adv_cdice + attack_cdice

                partial_result = dict({
                  'attack' : attack,
                  'attack_args' : {k : attack_args[k] for k in ['eps','step_alpha','epochs']}, #
                  'baseline_closs' : closs,
                  'baseline_cdice' : cdice,
                  'attack_closs' : adv_closs,
                  'attack_cdice' : adv_cdice,
                  'attack_l2_diff' : l2_diff,
                  'attack_ln_diff' : ln_diff
                })

                jsonString = json.dumps(str(partial_result))

                #results.append(copy.deepcopy(result_dict))

                with open("eval_results/{}-{}-{}-{}-metrics.json".format(attack, add_gaussian, batches,i),"w") as jsonFile:
                  jsonFile.write(jsonString)

                image_gt = "eval_results/ground-truth-{}-{}-{}-{}.pdf".format(attack, add_gaussian, batches, i)
                plt.imshow(np.squeeze(x), cmap='gray')
                plt.imshow(np.squeeze(y),cmap='viridis',alpha=0.7)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(image_gt,format='pdf')
                plt.clf()

                image_benign = "eval_results/benign-{}-{}-{}-{}.pdf".format(attack, add_gaussian, batches, i)
                plt.imshow(np.squeeze(x), cmap='gray')
                plt.imshow(np.squeeze(non_adv_mask_out),cmap='viridis',alpha=0.7)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(image_benign,format='pdf')
                plt.clf()

                image_adv = "eval_results/adversarial-{}-{}-{}-{}.pdf".format(attack, add_gaussian, batches, i)
                plt.imshow(np.squeeze(adv_x[i]), cmap='gray')
                plt.imshow(np.squeeze(adv_mask_out),cmap='viridis',alpha=0.7)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(image_adv,format='pdf')
                plt.clf()

                plt.imshow(np.squeeze(adv_x[i]), cmap='gray')
                image_adv_input = "eval_results/adv-input-{}-{}-{}-{}.pdf".format(attack, add_gaussian, batches, i)
                plt.tight_layout()
                plt.axis('off')
                plt.savefig(image_adv_input,format='pdf')
                plt.clf()

                plt.imshow(np.squeeze(x), cmap='gray')
                image_adv_input = "eval_results/benign-input-{}-{}-{}-{}.pdf".format(attack, add_gaussian, batches, i)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(image_adv_input,format='pdf')
                plt.clf()

                print(attack_closs, attack_cdice, l2_diff, ln_diff)

        print("Evaluation results")
        print("{} Attack Params {}".format(attack, attack_args))
        print("Baseline metrics: Avg loss {}, Avg DICE Score {} ".format(baseline_closs / (batches*len(adv_x)),
                                                                         baseline_cdice / (batches*len(adv_x))))
        print("{} Attack effectiveness: Avg loss {}, Avg DICE Score {} ".format(attack, attack_closs / (batches*len(adv_x)),
                                                                                attack_cdice / (batches*len(adv_x))))
        print("{} Attack visibility: Avg l2-norm diff {} Avg l-inf-norm diff {}".format(attack, l2_diff_sum / (batches*len(adv_x)),
                                                                                        ln_diff_sum / (batches*len(adv_x))))
        result_dict = dict({
            'attack' : attack,
            'attack_args' : {k : attack_args[k] for k in ['eps','step_alpha','epochs']}, #
            'baseline_closs_avg' : baseline_closs / batches,
            'baseline_cdice_avg' : baseline_cdice / batches,
            'attack_closs_avg' : attack_closs / batches,
            'attack_cdice_avg' : attack_cdice / batches,
            'attack_l2_diff' : l2_diff_sum / batches,
            'attack_ln_diff' : ln_diff_sum / batches
          })

        results.append(copy.deepcopy(result_dict))
        print(results)
        
        jsonString = json.dumps(results)
        with open("eval_results/{}-results.json".format(attack),"w") as jsonFile:
          jsonFile.write(jsonString)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to evaluate a neural network model on the ACDC challenge data")
    parser.add_argument("EXP_PATH", type=str,
                        help="Path to experiment folder (assuming you are in the working directory)")
    parser.add_argument("ATTACK", type=str, help="Algorithm to generate adversarial examples", choices=ATTACKS)
    parser.add_argument("GAUSSIAN", type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        help="Perform gaussian attack with ATTACK as reference (default False)", default=False)
    args = parser.parse_args()

    # Setup model configuration
    base_path = sys_config.project_root
    model_path = os.path.join(base_path, args.EXP_PATH)
    config_file = glob.glob(model_path + '/*py')[0]
    config_module = config_file.split('/')[-1].rstrip('.py')
    exp_config = SourceFileLoader(fullname=config_module, path=os.path.join(config_file)).load_module()

    logging.warning("GENERATING EXAMPLES FOR TESTING SET")

    # Setup input and output paths
    input_path = sys_config.test_data_root
    output_path = os.path.join(model_path, 'adversarial_examples_' + args.ATTACK)
    image_path = os.path.join(output_path, 'image')
    diff_path = os.path.join(output_path, 'difference')
    utils.makefolder(image_path)
    utils.makefolder(diff_path)

    if args.ATTACK == 'spgd':
        sizes = [5, 7, 11, 15, 19]
        sigmas = [1.0, 3.0, 5.0, 10.0, 15.0]
        print('sizes:', sizes)
        print('sigmas:', sigmas)
        crafting_sizes = []
        crafting_weights = []
        for size in sizes:
            for sigma in sigmas:
                crafting_sizes.append(size)
                weight = gaussian_kernel(size, size / 2, sigma)[:, :, tf.newaxis, tf.newaxis]
                crafting_weights.append(weight)
        print(crafting_sizes)
        print(crafting_weights)

        attack_args = {'eps': 5, 'step_alpha': 1, 'epochs': 10, 'sizes': crafting_sizes,
                       'weights': crafting_weights}
        generate_adversarial_examples(input_path,
                                      output_path,
                                      model_path,
                                      attack=args.ATTACK,
                                      attack_args=attack_args,
                                      exp_config=exp_config,
                                      add_gaussian=args.GAUSSIAN)
    elif args.ATTACK == 'pgd':
        attack_args = {'step_alpha': 0.025, 'eps': 5, 'ord': np.inf, 'epochs': 10}

        generate_adversarial_examples(input_path,
                                      output_path,
                                      model_path,
                                      attack=args.ATTACK,
                                      attack_args=attack_args,
                                      exp_config=exp_config)

    elif args.ATTACK == 'fgsm':
        attack_args = {'step_alpha': 0.25, 'eps': 5, 'ord': np.inf, 'epochs': 10}

        generate_adversarial_examples(input_path,
                                      output_path,
                                      model_path,
                                      attack=args.ATTACK,
                                      attack_args=attack_args,
                                      exp_config=exp_config)
