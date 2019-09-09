import os
import glob

from segmentation import *
from utils.image_io import prepare_image
import numpy as np
from cv2.ximgproc import guidedFilter

from matplotlib import pyplot as plt

from _utils import *

image_dir = '../double_dip/images'
prior_hint_dir_fg = '../double_dip/saliency/output_fg'
prior_hint_dir_bg = '../double_dip/saliency/output_bg'
saliency_dir_fg = '../double_dip/saliency/output_fg'
saliency_dir_bg = '../double_dip/saliency/output_bg'

def _clear_output():
    os.system('rm -rf output/*')
    
def _make_dir(image_name):
    os.system('mkdir output/{}'.format(image_name))
    
def _copy_output_to_dir(image_name):
    os.system('mv output/*{}* output/{}/'.format(image_name, image_name))
    
def _plot_learning_curve(image_name, s_obj, title):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss', color=color)
    ax1.plot(s_obj.learning_curve, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx() 

    color = 'tab:blue'
    ax2.set_ylabel('PSNR', color=color)
    ax2.plot(s_obj.psnr_learning_curve, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout() 
    plt.title(title)
    fig.savefig('output/{}_{}.jpg'.format(
        image_name.split('.')[0], '_'.join(title.split(' '))))

def _compare_learning_curve(image_name, s_obj_1, s_obj_2, title, legend):
    fig = plt.figure()
    plt.plot(s_obj_1.psnr_learning_curve)
    plt.plot(s_obj_2.psnr_learning_curve)
    plt.legend(legend)
    plt.title(title)
    fig.savefig('output/{}_{}.jpg'.format(
        image_name.split('.')[0], '_'.join(title.split(' '))))
        

_clear_output()    
for input_path in glob.glob(image_dir + '/*'):

    image_name = os.path.basename(input_path).split('.')[0]
    print('----> started training on image << {} >>'.format(image_name))
    
    if image_name == 'eagle':
        continue ###!!!
        
    _make_dir(image_name)
    
    im = prepare_image(os.path.join(image_dir, image_name + '.jpg'))

    orig_fg = prepare_image(os.path.join(saliency_dir_fg, image_name + '.jpg'))
    orig_bg = prepare_image(os.path.join(saliency_dir_bg, image_name + '.jpg'))


    prior_hint_name = image_name.split('.')[0] + '_cluster_hint' + '.jpg' 
    prior_fg = prepare_image(os.path.join(prior_hint_dir_fg, prior_hint_name))
    prior_bg = prepare_image(os.path.join(prior_hint_dir_bg, prior_hint_name))


    # Configs 
    stage_1_iter = 500
    stage_2_iter = 500
    show_every = 200

    # Original training 
    s = Segmentation(
        "{}_orig".format(image_name), 
        im, 
        bg_hint=orig_bg, 
        fg_hint=orig_fg,
        plot_during_training=True,
        show_every=show_every,
        first_step_iter_num=stage_1_iter,
        second_step_iter_num=stage_2_iter)

    s.optimize()
    s.finalize()
    _plot_learning_curve(image_name, s, 'original learning curve')

    #  Prior-based hint training
    s_prior = Segmentation(
        "{}_prior".format(image_name), 
        im, 
        bg_hint=prior_bg, 
        fg_hint=prior_fg,
        plot_during_training=True,
        show_every=show_every,
        first_step_iter_num=stage_1_iter,
        second_step_iter_num=stage_2_iter)

    s_prior.optimize()
    s_prior.finalize()
    _plot_learning_curve(image_name, s_prior, 'prior hint learning curve')

    _compare_learning_curve(
        image_name, 
        s, 
        s_prior, 
        'orig vs prior hint learning curve', 
        ['orig', 'with_prior'])

    # Debug mask 
    def _fix_mask(src_image, learned_mask):
        """
        fixing the masks using soft matting
        :return:
        """
        
        new_mask = guidedFilter(
            src_image.transpose(1, 2, 0).astype(np.float32),
            learned_mask[0].astype(np.float32),
            radius=7,
            eps=1e-4)
        
        def to_bin(x):
            v = np.zeros_like(x)
            v[x > 0.5] = 1
            return v

        return to_bin(np.array([new_mask]))

    src_image = s_prior.images[0]
    learned_mask_np = torch_to_np(s_prior.mask_net_outputs[0])
    fixed_mask_np = s_prior.fixed_masks[0]
    better_mask_np = _fix_mask(src_image, learned_mask_np)

    better_mask = np_to_pil(better_mask_np)
    better_mask.save('output/{}_better_mask_prior.jpg'.format(image_name))

    src_image = s.images[0]
    learned_mask_np = torch_to_np(s.mask_net_outputs[0])
    fixed_mask_np = s.fixed_masks[0]
    better_mask_np = _fix_mask(src_image, learned_mask_np)
    
    better_mask = np_to_pil(better_mask_np)
    better_mask.save('output/{}_better_mask_orig.jpg'.format(image_name))
    
    _copy_output_to_dir(image_name)