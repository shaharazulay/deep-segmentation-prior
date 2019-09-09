from torch.autograd import Variable

import PIL
from PIL import Image

import scipy.misc
import numpy as np
import glob
import os

from _model import ConvAutoencoder, skip, collect_feature_maps
from _spectral import get_spectral_histogram
from _cluster import get_cluster_segmentation
from _utils import *

from net import skip, get_noise  # code migratred from DoubleDIP

from matplotlib import pyplot as plt

image_dir = '../double_dip/images'
prior_output_dir = '../double_dip/priors'
hint_output_dir = '../double_dip/saliency/output_scaled'


for input_path in glob.glob(image_dir + '/*'):
    
    image_name = os.path.basename(input_path).split('.')[0]
    print('started processing image {}'.format(image_name))

    img_pil = Image.open(input_path)
    img_pil = crop_image_by_multiplier(img_pil, d=32)
    img_np = pil_to_np(img_pil)

    device = 'cpu'

    gen_model = lambda: skip(
        2, 3,
        num_channels_down=[8, 16, 32],
        num_channels_up=[8, 16, 32],
        num_channels_skip=[0, 0, 0],
        upsample_mode='bilinear',
        filter_size_down=3,
        filter_size_up=3,
        need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU').to(device)

    learning_rate = 0.01
    l1_loss = nn.L1Loss()

    gen_optimizer = lambda: torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate)


    # Define noise input 
    input_type = 'noise'
    input_depth = 2

    gen_noise = lambda: get_noise(
            input_depth,
            input_type,
            (img_np.shape[1], img_np.shape[2]))\
        .type(torch.FloatTensor)\
        .detach()

    gen_noise().shape


    print("creating prior...")
    # Train model for several epochs 
    num_iter = 50
    num_runs_for_stablility = 11

    reduce_ = lambda z: {
        k: np.median(np.array([d.get(k) for d in z]), axis=0)
        for k in set().union(*z)
    }

    X = torch.from_numpy(img_np).unsqueeze(0)
    X = Variable(X)

    total_recon_curve = []
    for run in range(num_runs_for_stablility):
        print(">>> run {} out of {} for stabliltiy".format(run + 1, num_runs_for_stablility))
        noise = gen_noise()
        model = gen_model()
        model.train()
        optimizer = gen_optimizer()
        
        learning_curve = []
        recon_curve = []

        for epoch in range(num_iter):

            X_rec = model(noise)  # inference
        
            loss = l1_loss(X_rec, X)
            learning_curve.append(loss)

            if (epoch + 1) % 50 == 0:
                print("epoch:: {}, LOSS = {}".format(epoch + 1, loss))
                X_rec_np = torch_to_np(X_rec)
                recon_curve.append((epoch + 1, X_rec_np))

            loss.backward()
            optimizer.step()
            model.zero_grad()
        
        total_recon_curve.append(dict(recon_curve))
        
    avg_recon_curve = reduce_(total_recon_curve)

    # Save a final prior for segmentation 
    target_epochs = 50
    prior_result = avg_recon_curve[target_epochs]

    image = np_to_pil(prior_result)
    image.save(os.path.join(prior_output_dir, '{}_prior.jpg'.format(image_name)))

    # Save prior change over epochs
    # TODO

    # Spectral Analysis
    print("creating spectral analysis...")
    inputs = [img_np, prior_result]

    spectral_hist = []
    for image in inputs:
        spectral_hist.append(get_spectral_histogram(image))
        

    plt.clf()
    plt.semilogy(spectral_hist[0])
    plt.semilogy(spectral_hist[1])
    plt.legend(['original', 'prior'])
    plt.xlabel('Spatial Frequency')
    plt.ylabel('Power Spectrum')
    plt.title('{}'.format(image_name))
    plt.savefig(os.path.join(prior_output_dir, '{}_spectral.jpg'.format(image_name)))


    # Segment the Image (using KMeans)
    print("creating the segmentation hint...")
    window_size = 7
    inputs = [img_np, prior_result]

    cluster_segmentations = []
    for image in inputs:
        result = get_cluster_segmentation(image, window_size=window_size, n_clusters=2)
        cluster_segmentations.append(result)

    image = np_to_pil(cluster_segmentations[-1])
    image.save(os.path.join(hint_output_dir, '{}_cluster_hint.jpg'.format(image_name)))



