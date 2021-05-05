# Adversarial Attacks against MRI Segmentation

This repository contains code to run adversarial attacks against the state-of-the-art cardiac cardiac segmentation networks as described in this paper: [An Exploration of 2D and 3D Deep Learning Techniques for Cardiac MR Image Segmentation](https://arxiv.org/abs/1709.04496). 

The code and instructions for training the target model can be found here - https://github.com/baumgach/acdc_segmenter.

## Requirements 

- Python 3.5 
- Tensorflow (tested with tensorflow 1.15)
- The package requirements are given in `requirements.txt`

## Running the code locally

Open the `config/system.py` and edit all the paths there to match your system.

Launch attacks by running the following
``` python evaluate acdc_logdir/unet2D_bn_modified_wxent <attack> <Add Gaussian noise>```

For example, ```python evaluate acdc_logdir/unet2D_bn_modified_wxent fgsm False```

where you have to adapt the line to match your experiment. Note that, the path must be given relative to your
working directory. Giving the full path will not work.

## References

```
@article{baumgartner2017exploration,
  title={An Exploration of {2D} and {3D} Deep Learning Techniques for Cardiac {MR} Image Segmentation},
  author={Baumgartner, Christian F and Koch, Lisa M and Pollefeys, Marc and Konukoglu, Ender},
  journal={arXiv preprint arXiv:1709.04496},
  year={2017}
}
```
## Contributors:
1. Abhinav Garg (garg19@illinois.edu)
2. Anant Kandikuppa (anantk3@illinois.edu)
