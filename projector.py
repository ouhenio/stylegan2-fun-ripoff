# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
from time import perf_counter

import click
from typing import List
import imageio
import numpy as np
import PIL.Image

import torch
import torch.nn.functional as F
from torchvision import models

import dnnlib
from dnnlib.util import format_time, copy_files_and_create_dirs
import legacy

from torch_utils.gen_utils import w_to_img, make_run_dir, save_config, compress_video
from tqdm import tqdm
from pytorch_ssim import SSIM  # from https://github.com/Po-Hsun-Su/pytorch-ssim


# ----------------------------------------------------------------------------


class VGG16Features(torch.nn.Module):
    """
    Use pre-trained VGG16 provided by PyTorch. Code modified from lainwired/pacifinapacific
    https://github.com/pacifinapacific/StyleGAN_LatentEditor. My modification is I that we can use
    the ReLU activation if we want, or the pure conv1_1, conv1_2, conv3_2, and conv4_2 activations.

    My conclusions are that it's best to have one model of VGG, so I will use the one provided by NVIDIA
    as it is both easier to slice and it can return LPIPS if so desired.
    """
    # Image2StyleGAN: How to Embed Images into the StyleGAN latent space? https://arxiv.org/abs/1904.03189,
    #                   layers = [0, 2, 12, 19]
    # Image2StyleGAN++: How to Edit the Embedded Images? https://arxiv.org/abs/1911.11544,
    #                   layers = [0, 2, 7, 14], but make sure to return conv3_3 twice for the Style Loss
    def __init__(self, device, use_relu=False):
        super(VGG16Features, self).__init__()
        # Load and partition the model
        vgg16 = models.vgg16(pretrained=True).to(device)
        self.vgg16_features = vgg16.features
        self.avgpool = vgg16.avgpool  # TODO: more work can be done to partition any part of the model, but not my jam
        self.classifier = vgg16.classifier

        self.conv1_1 = torch.nn.Sequential()
        self.conv1_2 = torch.nn.Sequential()
        self.conv3_2 = torch.nn.Sequential()
        self.conv4_2 = torch.nn.Sequential()

        layers = [0, 2, 12, 19]
        if use_relu:
            layers = [layer + 1 for layer in layers]

        for i in range(layers[0] + 1):
            self.conv1_1.add_module(str(i), self.vgg16_features[i])

        for i in range(layers[0] + 1, layers[1] + 1):
            self.conv1_2.add_module(str(i), self.vgg16_features[i])

        for i in range(layers[1] + 1, layers[2] + 1):
            self.conv3_2.add_module(str(i), self.vgg16_features[i])

        for i in range(layers[2] + 1, layers[3] + 1):
            self.conv4_2.add_module(str(i), self.vgg16_features[i])

        # We're not optimizing VGG16
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        conv1_1 = self.slice0(x)
        conv1_2 = self.slice1(conv1_1)
        conv3_2 = self.slice2(conv1_2)
        conv4_2 = self.slice3(conv3_2)

        conv1_1 = conv1_1 / torch.numel(conv1_1)
        conv1_2 = conv1_2 / torch.numel(conv1_2)
        conv3_2 = conv3_2 / torch.numel(conv3_2)
        conv4_2 = conv4_2 / torch.numel(conv4_2)

        return conv1_1, conv1_2, conv3_2, conv4_2


class VGG16FeaturesNVIDIA(torch.nn.Module):
    def __init__(self, vgg16):
        super(VGG16FeaturesNVIDIA, self).__init__()
        # ReLU is already included in the output of every conv output
        self.conv1_1 = vgg16.layers.conv1
        self.conv1_2 = vgg16.layers.conv2
        self.pool1 = vgg16.layers.pool1

        self.conv2_1 = vgg16.layers.conv3
        self.conv2_2 = vgg16.layers.conv4
        self.pool2 = vgg16.layers.pool2

        self.conv3_1 = vgg16.layers.conv5
        self.conv3_2 = vgg16.layers.conv6
        self.conv3_3 = vgg16.layers.conv7
        self.pool3 = vgg16.layers.pool3

        self.conv4_1 = vgg16.layers.conv8
        self.conv4_2 = vgg16.layers.conv9
        self.conv4_3 = vgg16.layers.conv10
        self.pool4 = vgg16.layers.pool4

        self.conv5_1 = vgg16.layers.conv11
        self.conv5_2 = vgg16.layers.conv12
        self.conv5_3 = vgg16.layers.conv13
        self.pool5 = vgg16.layers.pool5
        self.adavgpool = torch.nn.AdaptiveAvgPool2d(output_size=(7, 7))  # We need this for 256x256 images (> 224x224)

        self.fc1 = vgg16.layers.fc1
        self.fc2 = vgg16.layers.fc2
        self.fc3 = vgg16.layers.fc3
        self.softmax = vgg16.layers.softmax

    def get_layers_features(self, x: torch.Tensor, layers: List[str], normed: bool = False, sqrt_normed: bool = False):
        """
        x is an image/tensor of shape [1, 3, 256, 256], and layers is a list of the names of the layers you wish
        to return in order to compare the activations/features with another image.

        Example:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            img1 = torch.randn(1, 3, 256, 256, device=device)
            img2 = torch.randn(1, 3, 256, 256, device=device)
            layers = ['conv1_1', 'conv1_2', 'conv3_3', 'conv3_3', 'fc3']  # Indeed, return twice conv3_3

            vgg16 = VGG16FeaturesNVIDIA(device=device)

            # Get the desired features from the layers list
            features1 = vgg16(img1, layers)
            features2 = vgg16(img2, layers)

            # Get, e.g., the MSE loss between the two features
            mse = torch.nn.MSELoss(reduction='mean')
            loss = sum(map(lambda x, y: mse(x, y), features1, features2))
        """
        # Legend: => conv2d, -> max pool 2d, ~> adaptive average pool 2d, ->> fc layer; shapes of input/output are shown
        assert layers is not None
        conv1_1 = self.conv1_1(x)                         # [1, 3, 256, 256] => [1, 64, 256, 256]
        conv1_2 = self.conv1_2(conv1_1)                   # [1, 64, 256, 256] => [1, 64, 256, 256]

        conv2_1 = self.conv2_1(self.pool1(conv1_2))       # [1, 64, 256, 256] -> [1, 64, 128, 128] => [1, 128, 128, 128]
        conv2_2 = self.conv2_2(conv2_1)                   # [1, 128, 128, 128] => [1, 128, 128, 128]

        conv3_1 = self.conv3_1(self.pool2(conv2_2))       # [1, 128, 128, 128] -> [1, 128, 64, 64] => [1, 256, 64, 64]
        conv3_2 = self.conv3_2(conv3_1)                   # [1, 256, 64, 64] => [1, 256, 64, 64]
        conv3_3 = self.conv3_3(conv3_2)                   # [1, 256, 64, 64] => [1, 256, 64, 64]

        conv4_1 = self.conv4_1(self.pool3(conv3_3))       # [1, 256, 64, 64] -> [1, 256, 32, 32] => [1, 512, 32, 32]
        conv4_2 = self.conv4_2(conv4_1)                   # [1, 512, 32, 32] => [1, 512, 32, 32]
        conv4_3 = self.conv4_3(conv4_2)                   # [1, 512, 32, 32] => [1, 512, 32, 32]

        conv5_1 = self.conv5_1(self.pool4(conv4_3))       # [1, 512, 32, 32] -> [1, 512, 16, 16] => [1, 512, 16, 16]
        conv5_2 = self.conv5_2(conv5_1)                   # [1, 512, 16, 16] => [1, 512, 16, 16]
        conv5_3 = self.conv5_3(conv5_2)                   # [1, 512, 16, 16] => [1, 512, 16, 16]

        adavgpool = self.adavgpool(self.pool5(conv5_3))   # [1, 512, 16, 16] -> [1, 512, 8, 8] ~> [1, 512, 7, 7]
        fc1 = self.fc1(adavgpool)                         # [1, 512, 7, 7] ->> [1, 4096]; w/ReLU
        fc2 = self.fc2(fc1)                               # [1, 4096] ->> [1, 4096]; w/ReLU
        fc3 = self.softmax(self.fc3(fc2))                 # [1, 4096] ->> [1, 1000]; w/o ReLU; apply softmax

        result_list = list()
        for layer in layers:
            if normed:
                # Divide each layer by the number of elements in it
                result_list.append(eval(layer) / torch.numel(eval(layer)))
            elif sqrt_normed:
                # Divide each layer by the square root of the number of elements in it
                result_list.append(eval(layer) / torch.tensor(torch.numel(eval(layer)), dtype=float).sqrt())
            else:
                result_list.append(eval(layer))
        return result_list


# ----------------------------------------------------------------------------


def project(
        G,
        target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        *,
        projection_seed: int,
        truncation_psi: float,
        num_steps: int = 1000,
        w_avg_samples: int = 10000,
        initial_learning_rate: float = 0.1,
        initial_noise_factor: float = 0.05,
        constant_learning_rate: bool = False,
        lr_rampdown_length: float = 0.25,
        lr_rampup_length: float = 0.05,
        noise_ramp_length: float = 0.75,
        regularize_noise_weight: float = 1e5,
        project_in_wplus: bool = False,
        loss_paper: str = 'sgan2',  # ['sgan2' | 'im2sgan']  TODO: try one with gradients from CLIP
        normed: bool = False,
        sqrt_normed: bool = False,
        start_wavg: bool = True,
        device: torch.device) -> torch.Tensor:  # output shape: [num_steps, C, 512], C depending on resolution of G
    """
    Projecting a 'target' image into the W latent space. The user has an option to project into W+, where all elements
    in the latent vector are different. Likewise, the projection process can start from the W midpoint or from a random
    point, though results have shown that starting from the midpoint (start_wavg) yields the best results.
    """
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)

    # Compute w stats.
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    if project_in_wplus:  # Thanks to @pbaylies for a clean way on how to do this
        print('Projecting in W+ latent space...')
        if start_wavg:
            print(f'Starting from W midpoint using {w_avg_samples} samples...')
            w_avg = torch.mean(w_samples, axis=0, keepdims=True)  # [1, L, C]
        else:
            print(f'Starting from a random vector (seed: {projection_seed})...')
            z = np.random.RandomState(projection_seed).randn(1, G.z_dim)
            w_avg = G.mapping(torch.from_numpy(z).to(device), None)  # [1, L, C]
            w_avg = G.mapping.w_avg + truncation_psi * (w_avg - G.mapping.w_avg)
        w_std = (torch.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
    else:
        print('Projecting in W latent space...')
        w_samples = w_samples[:, :1, :]  # [N, 1, C]
        if start_wavg:
            print(f'Starting from W midpoint using {w_avg_samples} samples...')
            w_avg = torch.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
        else:
            print(f'Starting from a random vector (seed: {projection_seed})...')
            z = np.random.RandomState(projection_seed).randn(1, G.z_dim)
            w_avg = G.mapping(torch.from_numpy(z).to(device), None)[:, :1, :]  # [1, 1, C]; fake w_avg
            w_avg = G.mapping.w_avg + truncation_psi * (w_avg - G.mapping.w_avg)
        w_std = (torch.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
    # Setup noise inputs.
    noise_buffs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}

    # Features for target image. Reshape to 256x256 if it's larger to use with VGG16
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')

    # Load the VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Define the new losses
    if loss_paper == 'sgan2':
        target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    elif loss_paper == 'im2sgan':
        # Use specific layers
        vgg16 = VGG16FeaturesNVIDIA(vgg16)
        # Too cumbersome to add as command-line arg, so we leave it here; use whatever you need, as many times as needed
        layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2',
                  'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'fc1', 'fc2', 'fc3']
        target_features = vgg16.get_layers_features(target_images, layers, normed=normed, sqrt_normed=sqrt_normed)
        # Uncomment the next line if you also want to use LPIPS features
        # lpips_target_features = vgg16(target_images, resize_images=False, return_lpips=True)

        mse = torch.nn.MSELoss(reduction='mean')
        ssim_out = SSIM()  # can be used as a loss; recommended usage: ssim_loss = 1 - ssim_out(img1, img2)

    w_opt = w_avg.clone().detach().requires_grad_(True)
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_buffs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_buffs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2

        if constant_learning_rate:
            # Turn off the rampup/rampdown of the learning rate
            lr_ramp = 1.0
        else:
            lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        if project_in_wplus:
            ws = w_opt + w_noise
        else:
            ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        if loss_paper == 'sgan2':
            synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
            dist = (target_features - synth_features).square().sum()

            # Noise regularization.
            reg_loss = 0.0
            for v in noise_buffs.values():
                noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)
            loss = dist + reg_loss * regularize_noise_weight
            # Print in the same line (avoid cluttering the commandline)
            n_digits = int(np.log10(num_steps)) + 1 if num_steps > 0 else 1
            message = f'step {step + 1:{n_digits}d}/{num_steps}: dist {dist:.7e} | loss {loss.item():.7e}'
            print(message, end='\r')

            last_status = {'dist': dist.item(), 'loss': loss.item()}

        elif loss_paper == 'im2sgan':
            # Uncomment to also use LPIPS features as loss (must be better fine-tuned):
            # lpips_synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)

            synth_features = vgg16.get_layers_features(synth_images, layers, normed=normed, sqrt_normed=sqrt_normed)
            percept_error = sum(map(lambda x, y: mse(x, y), target_features, synth_features))

            # Also uncomment to add the LPIPS loss to the perception error (to-be better fine-tuned)
            # percept_error += 1e1 * (lpips_target_features - lpips_synth_features).square().sum()

            # Pixel-level MSE
            mse_error = mse(synth_images, target_images) / (G.img_channels * G.img_resolution * G.img_resolution)
            ssim_loss = ssim_out(target_images, synth_images)  # tracking SSIM (can also be added the total loss)
            loss = percept_error + mse_error  # + 1e-2 * (1 - ssim_loss)  # needs to be fine-tuned

            # Noise regularization.
            reg_loss = 0.0
            for v in noise_buffs.values():
                noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)
            loss += reg_loss * regularize_noise_weight
            # We print in the same line (avoid cluttering the commandline)
            n_digits = int(np.log10(num_steps)) + 1 if num_steps > 0 else 1
            message = f'step {step + 1:{n_digits}d}/{num_steps}: percept loss {percept_error.item():.7e} | ' \
                      f'pixel mse {mse_error.item():.7e} | ssim {ssim_loss.item():.7e} | loss {loss.item():.7e}'
            print(message, end='\r')

            last_status = {'percept_error': percept_error.item(),
                           'pixel_mse': mse_error.item(),
                           'ssim': ssim_loss.item(),
                           'loss': loss.item()}

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_buffs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    # Save run config
    run_config = {
        'num_steps': num_steps,
        'w_avg_samples': w_avg_samples,
        'initial_learning_rate': initial_learning_rate,
        'initial_noise_factor': initial_noise_factor,
        'constant_learning_rate': constant_learning_rate,
        'lr_rampdown_length': lr_rampdown_length,
        'lr_rampup_length': lr_rampup_length,
        'noise_ramp_length': noise_ramp_length,
        'regularize_noise_weight': regularize_noise_weight,
        'project_in_wplus': project_in_wplus,
        'loss_paper': loss_paper,
        'start_wavg': start_wavg,
        'projection_seed': projection_seed,
        'truncation_psi': truncation_psi,
        'elapsed_time': '',
        'last_commandline_status': last_status
    }

    if project_in_wplus:
        return w_out, run_config  # [num_steps, L, C]
    return w_out.repeat([1, G.mapping.num_ws, 1]), run_config  # [num_steps, 1, C] => [num_steps, L, C]


# ----------------------------------------------------------------------------


@click.command()
@click.pass_context
@click.option('--network', '-net', 'network_pkl', type=click.Path(exists=True, dir_okay=False), help='Network pickle filename', required=True)
@click.option('--target', '-t', 'target_fname', type=click.Path(exists=True, dir_okay=False), help='Target image file to project to', required=True, metavar='FILE')
# Optimization options
@click.option('--num-steps', '-nsteps', help='Number of optimization steps', type=click.IntRange(min=0), default=1000, show_default=True)
@click.option('--init-lr', '-lr', 'initial_learning_rate', type=float, help='Initial learning rate of the optimization process', default=0.1, show_default=True)
@click.option('--constant-lr', 'constant_learning_rate', is_flag=True, help='Add flag to use a constant learning rate throughout the optimization (turn off the rampup/rampdown)')
@click.option('--reg-noise-weight', '-regw', 'regularize_noise_weight', type=float, help='Noise weight regularization', default=1e5, show_default=True)
@click.option('--seed', type=int, help='Random seed', default=303, show_default=True)
# Video options
@click.option('--save-video', '-video', is_flag=True, help='Save an mp4 video of optimization progress')
@click.option('--compress', is_flag=True, help='Compress video with ffmpeg-python; same resolution, lower memory size')
@click.option('--fps', type=int, help='FPS for the mp4 video of optimization progress (if saved)', default=30, show_default=True)
# Options on which space to project to (W or W+) and where to start: the middle point of W (w_avg) or a specific seed
@click.option('--project-in-wplus', '-wplus', is_flag=True, help='Project in the W+ latent space')
@click.option('--start-wavg', '-wavg', type=bool, help='Start with the average W vector, ootherwise will start from a random seed (provided by user)', default=True, show_default=True)
@click.option('--projection-seed', type=int, help='Seed to start projection from', default=None, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi to use in projection when using a projection seed', default=0.7, show_default=True)
# Decide the loss to use when projecting (you must select the VGG16 features/layers to use in the im2sgan loss)
@click.option('--loss-paper', '-loss', type=click.Choice(['sgan2', 'im2sgan']), help='Loss to use', default='sgan2', show_default=True)
@click.option('--vgg-normed', 'normed', is_flag=True, help='Add flag to norm the VGG16 features by the number of elements per layer')
@click.option('--vgg-sqrt-normed', 'sqrt_normed', is_flag=True, help='Add flag to norm the VGG16 features by the square root of the number of elements per layer')
@click.option('--save-every-step', '-saveall', is_flag=True, help='Save every step taken in the projection (npy and image).')
# Extra parameters for saving the results
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out'), show_default=True, metavar='DIR')
@click.option('--description', '-desc', type=str, help='Extra description to add to the experiment name', default='')
def run_projection(
        ctx: click.Context,
        network_pkl: str,
        target_fname: str,
        num_steps: int,
        initial_learning_rate: float,
        constant_learning_rate: bool,
        regularize_noise_weight: float,
        seed: int,
        save_video: bool,
        compress: bool,
        fps: int,
        project_in_wplus: bool,
        start_wavg: bool,
        projection_seed: int,
        truncation_psi: float,
        loss_paper: str,
        normed: bool,
        sqrt_normed: bool,
        save_every_step: bool,
        outdir: str,
        description: str,
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python projector.py --target=~/mytarget.png --project-in-wplus --save-video --num-steps=1000 --save-every-step \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # If we're not starting from the W midpoint, assert the user fed a seed to start from
    if not start_wavg:
        if projection_seed is None:
            ctx.fail('Provide a seed to start from if not starting from the midpoint. Use "--projection-seed" to do so')

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)

    # Load target image.
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps, run_config = project(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device),
        num_steps=num_steps,
        initial_learning_rate=initial_learning_rate,
        constant_learning_rate=constant_learning_rate,
        regularize_noise_weight=regularize_noise_weight,
        project_in_wplus=project_in_wplus,
        start_wavg=start_wavg,
        projection_seed=projection_seed,
        truncation_psi=truncation_psi,
        loss_paper=loss_paper,
        normed=normed,
        sqrt_normed=sqrt_normed,
        device=device
    )
    elapsed_time = format_time(perf_counter()-start_time)
    print(f'\nElapsed time: {elapsed_time}')
    run_config['elapsed_time'] = elapsed_time
    # Make the run dir automatically
    desc = 'projection-wplus' if project_in_wplus else 'projection-w'
    desc = f'{desc}-wavgstart' if start_wavg else f'{desc}-seed{projection_seed}start'
    desc = f'{desc}-{description}' if len(description) != 0 else desc
    desc = f'{desc}-{loss_paper}'
    run_dir = make_run_dir(outdir, desc)

    # Save the configuration used
    ctx.obj = {
        'network_pkl': network_pkl,
        'description': description,
        'target_fname': target_fname,
        'outdir': run_dir,
        'save_video': save_video,
        'seed': seed,
        'video_fps': fps,
        'save_every_step': save_every_step,
        'run_config': run_config
    }
    # Save the run configuration
    save_config(ctx=ctx, run_dir=run_dir)
    copy_files_and_create_dirs(files=[(__file__, run_dir)])

    # Render debug output: optional video and projected image and W vector.
    result_name = os.path.join(run_dir, 'proj')
    npy_name = os.path.join(run_dir, 'projected')
    # If we project in W+, add to the name of the results
    if project_in_wplus:
        result_name, npy_name = f'{result_name}_wplus', f'{npy_name}_wplus'
    # Either in W or W+, we can start from the W midpoint or one given by the projection seed
    if start_wavg:
        result_name, npy_name = f'{result_name}_wavg', f'{npy_name}_wavg'
    else:
        result_name, npy_name = f'{result_name}_seed-{projection_seed}', f'{npy_name}_seed-{projection_seed}'

    # Save the target image
    target_pil.save(os.path.join(run_dir, 'target.jpg'))

    if save_every_step:
        # Save every projected frame and W vector. TODO: This can be optimized to be saved as training progresses
        n_digits = int(np.log10(num_steps)) + 1 if num_steps > 0 else 1
        for step in tqdm(range(num_steps), desc='Saving projection results', unit='steps'):
            w = projected_w_steps[step]
            synth_image = w_to_img(G, dlatents=w, noise_mode='const')[0]
            PIL.Image.fromarray(synth_image, 'RGB').save(f'{result_name}_step{step:0{n_digits}d}.jpg')
            np.save(f'{npy_name}_step{step:0{n_digits}d}.npy', w.unsqueeze(0).cpu().numpy())
    else:
        # Save only the final projected frame and W vector.
        print('Saving projection results...')
        projected_w = projected_w_steps[-1]
        synth_image = w_to_img(G, dlatents=projected_w, noise_mode='const')[0]
        PIL.Image.fromarray(synth_image, 'RGB').save(f'{result_name}_final.jpg')
        np.save(f'{npy_name}_final.npy', projected_w.unsqueeze(0).cpu().numpy())

    # Save the optimization video and compress it if so desired
    if save_video:
        video = imageio.get_writer(f'{result_name}.mp4', mode='I', fps=fps, codec='libx264', bitrate='16M')
        print(f'Saving optimization progress video "{result_name}.mp4"')
        for projected_w in projected_w_steps:
            synth_image = w_to_img(G, dlatents=projected_w, noise_mode='const')[0]
            video.append_data(np.concatenate([target_uint8, synth_image], axis=1))  # left side target, right projection
        video.close()

    if save_video and compress:
        # Compress the video; might fail, and is a basic command that can also be better optimized
        compress_video(original_video=f'{result_name}.mp4',
                       original_video_name=f'{result_name.split(os.sep)[-1]}',  # make code OS-independent (hopefully)
                       outdir=run_dir,
                       ctx=ctx)

# ----------------------------------------------------------------------------


if __name__ == "__main__":
    run_projection()  # pylint: disable=no-value-for-parameter


# ----------------------------------------------------------------------------
