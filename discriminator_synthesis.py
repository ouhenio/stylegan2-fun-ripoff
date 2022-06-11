import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

import PIL
from PIL import Image

try:
    import ffmpeg
except ImportError:
    raise ImportError('ffmpeg-python not found! Install it via "pip install ffmpeg-python"')

try:
    import skvideo.io
except ImportError:
    raise ImportError('scikit-video not found! Install it via "pip install scikit-video"')

import scipy.ndimage as nd
import numpy as np

import os
import click
from typing import Union, Tuple, Optional, List
from collections import OrderedDict
from tqdm import tqdm

from torch_utils.gen_utils import parse_fps, make_run_dir, save_config, w_to_img, create_image_grid, compress_video, \
    parse_new_center, get_w_from_seed
import dnnlib
import legacy
from projector import VGG16FeaturesNVIDIA

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import moviepy.editor

# ----------------------------------------------------------------------------


@click.group()
def main():
    pass


# ----------------------------------------------------------------------------

def get_available_layers(max_resolution: int) -> List[str]:
    """Helper function to get the available layers given a max resolution (first block in the Discriminator)"""
    max_res_log2 = int(np.log2(max_resolution))
    block_resolutions = [2**i for i in range(max_res_log2, 2, -1)]

    available_layers = ['from_rgb']
    for block_res in block_resolutions:
        # We don't add the skip layer, as it's the same as conv1 (due to in-place addition; could be changed)
        available_layers.extend([f'b{block_res}_conv0', f'b{block_res}_conv1'])
    # We also skip 'b4_mbstd', as it doesn't add any new information compared to b8_conv1
    available_layers.extend(['b4_conv', 'fc', 'out'])
    return available_layers


def parse_layers(s: str) -> List[str]:
    """Helper function for parsing a string of comma-separated layers and returning a list of the individual layers"""
    str_list = s.split(',')

    # Get all the possible layers up to resolution 1024
    all_available_layers = get_available_layers(max_resolution=1024)

    for layer in str_list:
        message = f'{layer} is not a possible layer! Available layers: {all_available_layers}'
        # We also let the user choose all the layers
        assert layer in all_available_layers or layer == 'all', message

    return str_list


# ----------------------------------------------------------------------------


# TODO: move all feature extractor to its own file (this one and VGG16; eventually the CLIP one as well)
class DiscriminatorFeatures(torch.nn.Module):
    def __init__(self, D):
        super(DiscriminatorFeatures, self).__init__()

        # assert D.init_kwargs.architecture == 'resnet'  # removed as some resnet models don't have this attribute
        self.block_resolutions = D.block_resolutions

        # For loop to get all the inner features of the trained Discriminator with a resnet architecture
        for res in self.block_resolutions:
            if res == D.img_resolution:
                setattr(self, 'from_rgb', eval(f'D.b{res}.fromrgb'))
            setattr(self, f'b{res}_skip', eval(f'D.b{res}.skip'))
            setattr(self, f'b{res}_conv0', eval(f'D.b{res}.conv0'))
            setattr(self, f'b{res}_conv1', eval(f'D.b{res}.conv1'))

        # Unique, last block with a fc/out, so we can extract features in a regular fashion
        self.b4_mbstd = D.b4.mbstd
        self.b4_conv = D.b4.conv
        self.adavgpool = nn.AdaptiveAvgPool2d(4)  # Necessary if images are of different resolution than D.img_resolution
        self.fc = D.b4.fc
        self.out = D.b4.out

    def get_block_resolutions(self):
        """Get the block resolutions available for the current Discriminator"""
        return self.block_resolutions

    def get_layers_features(self,
                            x: torch.Tensor,            # Input image
                            layers: List[str] = None,
                            normed: bool = False,
                            sqrt_normed: bool = False) -> List[torch.Tensor]:
        """
        Get the feature of a specific layer of the Discriminator (with resnet architecture). The following shows the
        shapes of an image, x, as it flows through the different blocks that compose the Discriminator.

        *** Legend: => conv2d, -> flatten, ->> fc layer, ~> mbstd layer, +> adaptive average pool ***

        # First block / DiscriminatorBlock
        from_rgb = self.from_rgb(x)                                         # [1, 3, 1024, 1024] => [1, 32, 1024, 1024]
        b1024_skip = self.b1024_skip(from_rgb, gain=np.sqrt(0.5))           # [1, 32, 1024, 1024] => [1, 64, 512, 512]
        b1024_conv0 = self.b1024_conv0(from_rgb)                            # [1, 32, 1024, 1024] => [1, 32, 1024, 1024]
        b1024_conv1 = self.b1024_conv1(b1024_conv0, gain=np.sqrt(0.5))      # [1, 32, 1024, 1024] => [1, 64, 512, 512]
        b1024_conv1 = b1024_skip.add_(b1024_conv1)                          # [1, 64, 512, 512]

        # Second block / DiscriminatorBlock
        b512_skip = self.b512_skip(b1024_conv1, gain=np.sqrt(0.5))          # [1, 64, 512, 512] => [1, 128, 256, 256]
        b512_conv0 = self.b512_conv0(b1024_conv1)                           # [1, 64, 512, 512] => [1, 64, 512, 512]
        b512_conv1 = self.b512_conv1(b512_conv0, gain=np.sqrt(0.5))         # [1, 64, 512, 512] => [1, 128, 256, 256]
        b512_conv1 = b512_skip.add_(b512_conv1)                             # [1, 128, 256, 256]

        # Third block / DiscriminatorBlock
        b256_skip = self.b256_skip(b512_conv1, gain=np.sqrt(0.5))           # [1, 128, 256, 256] => [1, 256, 128, 128]
        b256_conv0 = self.b256_conv0(b512_conv1)                            # [1, 128, 256, 256] => [1, 128, 256, 256]
        b256_conv1 = self.b256_conv1(b256_conv0, gain=np.sqrt(0.5))         # [1, 128, 256, 256] => [1, 256, 128, 128]
        b256_conv1 = b256_skip.add_(b256_conv1)                             # [1, 256, 128, 128]

        # Fourth block / DiscriminatorBlock
        b128_skip = self.b128_skip(b256_conv1, gain=np.sqrt(0.5))           # [1, 256, 128, 128] => [1, 512, 64 ,64]
        b128_conv0 = self.b128_conv0(b256_conv1)                            # [1, 256, 128, 128] => [1, 256, 128, 128]
        b128_conv1 = self.b128_conv1(b128_conv0, gain=np.sqrt(0.5))         # [1, 256, 128, 128] => [1, 512, 64, 64]
        b128_conv1 = b128_skip.add_(b128_conv1)                             # [1, 512, 64, 64]

        # Fifth block / DiscriminatorBlock
        b64_skip = self.b64_skip(b128_conv1, gain=np.sqrt(0.5))             # [1, 512, 64, 64] => [1, 512, 32, 32]
        b64_conv0 = self.b64_conv0(b128_conv1)                              # [1, 512, 64, 64] => [1, 512, 64, 64]
        b64_conv1 = self.b64_conv1(b64_conv0, gain=np.sqrt(0.5))            # [1, 512, 64, 64] => [1, 512, 32, 32]
        b64_conv1 = b64_skip.add_(b64_conv1)                                # [1, 512, 32, 32]

        # Sixth block / DiscriminatorBlock
        b32_skip = self.b32_skip(b64_conv1, gain=np.sqrt(0.5))              # [1, 512, 32, 32] => [1, 512, 16, 16]
        b32_conv0 = self.b32_conv0(b64_conv1)                               # [1, 512, 32, 32] => [1, 512, 32, 32]
        b32_conv1 = self.b32_conv1(b32_conv0, gain=np.sqrt(0.5))            # [1, 512, 32, 32] => [1, 512, 16, 16]
        b32_conv1 = b32_skip.add_(b32_conv1)                                # [1, 512, 16, 16]

        # Seventh block / DiscriminatorBlock
        b16_skip = self.b16_skip(b32_conv1, gain=np.sqrt(0.5))              # [1, 512, 16, 16] => [1, 512, 8, 8]
        b16_conv0 = self.b16_conv0(b32_conv1)                               # [1, 512, 16, 16] => [1, 512, 16, 16]
        b16_conv1 = self.b16_conv1(b16_conv0, gain=np.sqrt(0.5))            # [1, 512, 16, 16] => [1, 512, 8, 8]
        b16_conv1 = b16_skip.add_(b16_conv1)                                # [1, 512, 8, 8]

        # Eighth block / DiscriminatorBlock
        b8_skip = self.b8_skip(b16_conv1, gain=np.sqrt(0.5))                # [1, 512, 8, 8] => [1, 512, 4, 4]
        b8_conv0 = self.b8_conv0(b16_conv1)                                 # [1, 512, 8, 8] => [1, 512, 8, 8]
        b8_conv1 = self.b8_conv1(b8_conv0, gain=np.sqrt(0.5))               # [1, 512, 8, 8] => [1, 512, 4, 4]
        b8_conv1 = b8_skip.add_(b8_conv1)                                   # [1, 512, 4, 4]

        # Ninth block / DiscriminatorEpilogue
        b4_mbstd = self.b4_mbstd(b8_conv1)                                  # [1, 512, 4, 4] ~> [1, 513, 4, 4]
        b4_conv = self.adavgpool(self.b4_conv(b4_mbstd))                    # [1, 513, 4, 4] => [1, 512, 4, 4] +> [1, 512, 4, 4]
        fc = self.fc(b4_conv.flatten(1))                                    # [1, 512, 4, 4] -> [1, 8192] ->> [1, 512]
        out = self.out(fc)                                                  # [1, 512] ->> [1, 1]
        """
        assert not (normed and sqrt_normed), 'Choose one of the normalizations!'

        # Return the full output if no layers are indicated
        if layers is None:
            layers = ['out']

        features_dict = OrderedDict()  # Can just be a dictionary, but I plan to use the order of the features later on
        features_dict['from_rgb'] = self.from_rgb(x)    # [1, 3, D.img_resolution, D.img_resolution] =>
        #                                                                => [1, 32, D.img_resolution, D.img_resolution]

        for idx, res in enumerate(self.block_resolutions):

            # conv0 and skip from the first block use from_rgb
            if idx == 0:
                features_dict[f'b{res}_skip'] = getattr(self, f'b{res}_skip')(
                    features_dict['from_rgb'], gain=np.sqrt(0.5))
                features_dict[f'b{res}_conv0'] = getattr(self, f'b{res}_conv0')(features_dict['from_rgb'])

            # The rest use the previous block's conv1
            else:
                features_dict[f'b{res}_skip'] = getattr(self, f'b{res}_skip')(
                    features_dict[f'b{self.block_resolutions[idx - 1]}_conv1'], gain=np.sqrt(0.5)
                )
                features_dict[f'b{res}_conv0'] = getattr(self, f'b{res}_conv0')(
                    features_dict[f'b{self.block_resolutions[idx - 1]}_conv1']
                )
            # Finally, pass the current block's conv0 and do the skip connection addition
            features_dict[f'b{res}_conv1'] = getattr(self, f'b{res}_conv1')(features_dict[f'b{res}_conv0'],
                                                                            gain=np.sqrt(0.5))
            features_dict[f'b{res}_conv1'] = features_dict[f'b{res}_skip'].add_(features_dict[f'b{res}_conv1'])

        # Irrespective of the image size/model size, the last block will be the same:
        features_dict['b4_mbstd'] = self.b4_mbstd(features_dict['b8_conv1'])  # [1, 512, 4, 4] ~> [1, 513, 4, 4]
        features_dict['b4_conv'] = self.b4_conv(features_dict['b4_mbstd'])    # [1, 513, 4, 4] => [1, 512, 4, 4]
        features_dict['b4_conv'] = self.adavgpool(features_dict['b4_conv'])   # [1, 512, 4, 4] +> [1, 512, 4, 4]  (Needed if x's resolution is not D.img_resolution)
        features_dict['fc'] = self.fc(features_dict['b4_conv'].flatten(1))    # [1, 512, 4, 4] -> [1, 8192] ->> [1, 512]
        features_dict['out'] = self.out(features_dict['fc'])                  # [1, 512] ->> [1, 1]

        result_list = list()
        for layer in layers:
            # Two options to normalize, otherwise we only add the unmodified output; recommended if using more than one layer
            if normed:
                result_list.append(features_dict[layer] / torch.numel(features_dict[layer]))
            elif sqrt_normed:
                result_list.append(features_dict[layer] / torch.tensor(torch.numel(features_dict[layer]),
                                                                        dtype=torch.float).sqrt())
            else:
                result_list.append(features_dict[layer])

        return result_list


# ----------------------------------------------------------------------------
# DeepDream code; modified from Erik Linder-Norén's repository: https://github.com/eriklindernoren/PyTorch-Deep-Dream

def get_image(seed: int = 0,
              starting_image: Union[str, os.PathLike] = None,
              image_size: int = 1024) -> Tuple[PIL.Image.Image, str]:
    """Set the random seed (NumPy + PyTorch), as well as get an image from a path or generate a random one with the seed"""
    torch.manual_seed(seed)
    rnd = np.random.RandomState(seed)

    # Load image or generate a random one if none is provided
    if starting_image is not None:
        image = Image.open(starting_image).convert('RGB').resize((image_size, image_size), Image.LANCZOS)
    else:
        starting_image = f'random_image-seed_{seed}.jpg'
        image = Image.fromarray(rnd.randint(0, 255, (image_size, image_size, 3), dtype='uint8'))

    return image, starting_image


def crop_resize_rotate(img: PIL.Image.Image,
                       crop_size: int = None,
                       new_size: int = None,
                       rotation_deg: float = None,
                       translate_x: float = 0.0,
                       translate_y: float = 0.0) -> PIL.Image.Image:
    """Center-crop the input image into a square of sides crop_size; can be resized to new_size; rotated rotation_deg counter-clockwise"""
    # Center-crop the input image
    if crop_size is not None:
        w, h = img.size                                         # Input image width and height
        img = img.crop(box=((w - crop_size) // 2,               # Left pixel coordinate
                            (h - crop_size) // 2,               # Upper pixel coordinate
                            (w + crop_size) // 2,               # Right pixel coordinate
                            (h + crop_size) // 2))              # Lower pixel coordinate
    # Resize
    if new_size is not None:
        img = img.resize(size=(new_size, new_size),             # Requested size of the image in pixels; (width, height)
                         resample=Image.LANCZOS)                # Resampling filter
    # Rotation and translation
    if rotation_deg is not None:
        img = img.rotate(angle=rotation_deg,                    # Angle to rotate image, counter-clockwise
                         resample=PIL.Image.BICUBIC,            # Resampling filter; options: [PIL.Image.NEAREST | PIL.Image.BILINEAR | PIL.Image.BICUBIC]
                         expand=False,                          # If True, the whole rotated image will be shown
                         translate=(translate_x, translate_y),  # Translate the image, from top-left corner (post-rotation)
                         fillcolor=(0, 0, 0))                   # Black background
    # TODO: tile the background
    return img


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def deprocess(image_np: torch.Tensor) -> np.ndarray:
    image_np = image_np.squeeze().transpose(1, 2, 0)
    image_np = image_np * std.reshape((1, 1, 3)) + mean.reshape((1, 1, 3))
    # image_np = (image_np + 1.0) / 2.0
    image_np = np.clip(image_np, 0.0, 1.0)
    image_np = (255 * image_np).astype('uint8')
    return image_np


def clip(image_tensor: torch.Tensor) -> torch.Tensor:
    """Clamp per channel"""
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[0, c] = torch.clamp(image_tensor[0, c], -m / s, (1 - m) / s)
    return image_tensor


def dream(image: PIL.Image.Image,
          model: torch.nn.Module,
          layers: List[str],
          normed: bool = False,
          sqrt_normed: bool = False,
          iterations: int = 20,
          lr: float = 1e-2) -> np.ndarray:
    """ Updates the image to maximize outputs for n iterations """
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    image = Variable(Tensor(image), requires_grad=True)
    for i in range(iterations):
        model.zero_grad()
        out = model.get_layers_features(image, layers=layers, normed=normed, sqrt_normed=sqrt_normed)
        loss = sum(layer.norm() for layer in out)                   # More than one layer may be used
        loss.backward()
        avg_grad = np.abs(image.grad.data.cpu().numpy()).mean()
        norm_lr = lr / avg_grad
        image.data += norm_lr * image.grad.data
        image.data = clip(image.data)
        # image.data = torch.clamp(image.data, -1.0, 1.0)
        image.grad.data.zero_()
    return image.cpu().data.numpy()


def deep_dream(image: PIL.Image.Image,
               model: torch.nn.Module,
               model_resolution: int,
               layers: List[str],
               normed: bool,
               sqrt_normed: bool,
               iterations: int,
               lr: float,
               octave_scale: float,
               num_octaves: int,
               unzoom_octave: bool = False,
               disable_inner_tqdm: bool = False) -> np.ndarray:
    """ Main deep dream method """
    # Center-crop and resize
    image = crop_resize_rotate(img=image, crop_size=min(image.size), new_size=model_resolution)
    # Preprocess image
    image = preprocess(image)
    # image = torch.from_numpy(np.array(image)).permute(-1, 0, 1) / 127.5 - 1.0  # alternative
    image = image.unsqueeze(0).cpu().data.numpy()
    # Extract image representations for each octave
    octaves = [image]
    for _ in range(num_octaves - 1):
        # Alternatively, see if we get better results with: https://www.tensorflow.org/tutorials/generative/deepdream#taking_it_up_an_octave
        octave = nd.zoom(octaves[-1], (1, 1, 1 / octave_scale, 1 / octave_scale), order=1)
        # Necessary for StyleGAN's Discriminator, as it cannot handle any image size
        if unzoom_octave:
            octave = nd.zoom(octave, np.array(octaves[-1].shape) / np.array(octave.shape), order=1)
        octaves.append(octave)

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(tqdm(octaves[::-1], desc=f'Dreaming w/layers {"|".join(x for x in layers)}',
                                              disable=disable_inner_tqdm)):
        if octave > 0:
            # Upsample detail to new octave dimension
            detail = nd.zoom(detail, np.array(octave_base.shape) / np.array(detail.shape), order=1)
        # Add deep dream detail from previous octave to new base
        input_image = octave_base + detail
        # Get new deep dream image
        dreamed_image = dream(input_image, model, layers, normed, sqrt_normed, iterations, lr)
        # Extract deep dream details
        detail = dreamed_image - octave_base

    return deprocess(dreamed_image)


# ----------------------------------------------------------------------------


@main.command(name='style-transfer')
def style_transfer_discriminator():
    print('Coming soon!')
    # Reference: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html


# ----------------------------------------------------------------------------


@main.command(name='dream')
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
# Synthesis options
@click.option('--seed', type=int, help='Random seed to use', default=0)
@click.option('--starting-image', type=str, help='Path to image to start from', default=None)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)', default=None)
@click.option('--lr', 'learning_rate', type=float, help='Learning rate', default=1e-2, show_default=True)
@click.option('--iterations', '-it', type=int, help='Number of gradient ascent steps per octave', default=20, show_default=True)
# Layer options
@click.option('--layers', type=parse_layers, help='Layers of the Discriminator to use as the features. If "all", will generate a dream image per available layer in the loaded model', default=['b16_conv1'], show_default=True)
@click.option('--normed', 'norm_model_layers', is_flag=True, help='Add flag to divide the features of each layer of D by its number of elements')
@click.option('--sqrt-normed', 'sqrt_norm_model_layers', is_flag=True, help='Add flag to divide the features of each layer of D by the square root of its number of elements')
# Octaves options
@click.option('--num-octaves', type=int, help='Number of octaves', default=5, show_default=True)
@click.option('--octave-scale', type=float, help='Image scale between octaves', default=1.4, show_default=True)
@click.option('--unzoom-octave', type=bool, help='Set to True for the octaves to be unzoomed (this will be slower)', default=True, show_default=True)
# Extra parameters for saving the results
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out', 'discriminator_synthesis'), show_default=True, metavar='DIR')
@click.option('--description', '-desc', type=str, help='Additional description name for the directory path to save results', default='', show_default=True)
def discriminator_dream(
        ctx: click.Context,
        network_pkl: Union[str, os.PathLike],
        seed: int,
        starting_image: Union[str, os.PathLike],
        class_idx: Optional[int],  # TODO: conditional model
        learning_rate: float,
        iterations: int,
        layers: List[str],
        norm_model_layers: bool,
        sqrt_norm_model_layers: bool,
        num_octaves: int,
        octave_scale: float,
        unzoom_octave: bool,
        outdir: Union[str, os.PathLike],
        description: str,
):
    print(f'Loading networks from "{network_pkl}"...')
    # Define the model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        D = legacy.load_network_pkl(f)['D'].eval().requires_grad_(False).to(device)  # type: ignore

    # Get the model resolution (image resizing and getting available layers)
    model_resolution = D.img_resolution

    # We will use the features of the Discriminator, on the layer specified by the user
    model = DiscriminatorFeatures(D).requires_grad_(False).to(device)

    if 'all' in layers:
        # Get all the available layers in a list
        available_layers = get_available_layers(max_resolution=model.get_block_resolutions()[0])

        # Get the image and image name
        image, starting_image = get_image(seed=seed, starting_image=starting_image, image_size=model_resolution)

        # Make the run dir in the specified output directory
        desc = 'discriminator-dream-all_layers'
        desc = f'{desc}-{description}' if len(description) != 0 else desc
        run_dir = make_run_dir(outdir, desc)

        # Save starting image
        image.save(os.path.join(run_dir, f'{os.path.basename(starting_image).split(".")[0]}.jpg'))

        # Save the configuration used
        ctx.obj = {
            'network_pkl': network_pkl,
            'synthesis_options': {
                'seed': seed,
                'starting_image': starting_image,
                'class_idx': class_idx,
                'learning_rate': learning_rate,
                'iterations': iterations
            },
            'layer_options': {
                'layer': available_layers,
                'norm_model_layers': norm_model_layers,
                'sqrt_norm_model_layers': sqrt_norm_model_layers
            },
            'octaves_options': {
                'num_octaves': num_octaves,
                'octave_scale': octave_scale,
                'unzoom_octave': unzoom_octave
            },
            'extra_parameters': {
                'outdir': run_dir,
                'description': description
            }
        }
        # Save the run configuration
        save_config(ctx=ctx, run_dir=run_dir)

        # For each layer:
        for av_layer in available_layers:
            # Extract deep dream image
            dreamed_image = deep_dream(image, model, model_resolution, layers=[av_layer], normed=norm_model_layers,
                                       sqrt_normed=sqrt_norm_model_layers, iterations=iterations, lr=learning_rate,
                                       octave_scale=octave_scale, num_octaves=num_octaves, unzoom_octave=unzoom_octave)

            # Save the resulting dreamed image
            filename = f'layer-{av_layer}_dreamed_{os.path.basename(starting_image).split(".")[0]}.jpg'
            Image.fromarray(dreamed_image, 'RGB').save(os.path.join(run_dir, filename))

    else:
        # Get the image and image name
        image, starting_image = get_image(seed=seed, starting_image=starting_image, image_size=model_resolution)

        # Extract deep dream image
        dreamed_image = deep_dream(image, model, model_resolution, layers=layers, normed=norm_model_layers,
                                   sqrt_normed=sqrt_norm_model_layers, iterations=iterations, lr=learning_rate,
                                   octave_scale=octave_scale, num_octaves=num_octaves, unzoom_octave=unzoom_octave)

        # Make the run dir in the specified output directory
        desc = f'discriminator-dream-layers_{"-".join(x for x in layers)}'
        desc = f'{desc}-{description}' if len(description) != 0 else desc
        run_dir = make_run_dir(outdir, desc)

        # Save the configuration used
        ctx.obj = {
            'network_pkl': network_pkl,
            'seed': seed,
            'starting_image': starting_image,
            'class_idx': class_idx,
            'learning_rate': learning_rate,
            'iterations': iterations,
            'layer': layers,
            'norm_model_layers': norm_model_layers,
            'sqrt_norm_model_layers': sqrt_norm_model_layers,
            'octave_scale': octave_scale,
            'num_octaves': num_octaves,
            'unzoom_octave': unzoom_octave,
            'outdir': run_dir,
            'description': description
        }
        # Save the run configuration
        save_config(ctx=ctx, run_dir=run_dir)

        # Save the resulting image and initial image
        filename = f'dreamed_{os.path.basename(starting_image)}'
        Image.fromarray(dreamed_image, 'RGB').save(os.path.join(run_dir, filename))
        image.save(os.path.join(run_dir, os.path.basename(starting_image)))


# ----------------------------------------------------------------------------


@main.command(name='dream-zoom')
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
# Synthesis options
@click.option('--seed', type=int, help='Random seed to use', default=0, show_default=True)
@click.option('--starting-image', type=str, help='Path to image to start from', default=None)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)', default=None)
@click.option('--lr', 'learning_rate', type=float, help='Learning rate', default=5e-3, show_default=True)
@click.option('--iterations', '-it', type=click.IntRange(min=1), help='Number of gradient ascent steps per octave', default=10, show_default=True)
# Layer options
@click.option('--layers', type=parse_layers, help='Layers of the Discriminator to use as the features. If None, will default to the output of D.', default=['b16_conv1'], show_default=True)
@click.option('--normed', 'norm_model_layers', is_flag=True, help='Add flag to divide the features of each layer of D by its number of elements')
@click.option('--sqrt-normed', 'sqrt_norm_model_layers', is_flag=True, help='Add flag to divide the features of each layer of D by the square root of its number of elements')
# Octaves options
@click.option('--num-octaves', type=click.IntRange(min=1), help='Number of octaves', default=5, show_default=True)
@click.option('--octave-scale', type=float, help='Image scale between octaves', default=1.4, show_default=True)
@click.option('--unzoom-octave', type=bool, help='Set to True for the octaves to be unzoomed (this will be slower)', default=False, show_default=True)
# Individual frame manipulation options
@click.option('--pixel-zoom', type=int, help='How many pixels to zoom per step (positive for zoom in, negative for zoom out, padded with black)', default=2, show_default=True)
@click.option('--rotation-deg', '-rot', type=float, help='Rotate image counter-clockwise per frame (padded with black)', default=0.0, show_default=True)
@click.option('--translate-x', '-tx', type=float, help='Translate the image in the horizontal axis per frame (from left to right, padded with black)', default=0.0, show_default=True)
@click.option('--translate-y', '-ty', type=float, help='Translate the image in the vertical axis per frame (from top to bottom, padded with black)', default=0.0, show_default=True)
# Video options
@click.option('--fps', type=parse_fps, help='FPS for the mp4 video of optimization progress (if saved)', default=25, show_default=True)
@click.option('--duration-sec', type=float, help='Duration length of the video', default=15.0, show_default=True)
@click.option('--reverse-video', is_flag=True, help='Add flag to reverse the generated video')
# Extra parameters for saving the results
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out', 'discriminator_synthesis'), show_default=True, metavar='DIR')
@click.option('--description', '-desc', type=str, help='Additional description name for the directory path to save results', default='', show_default=True)
def discriminator_dream_zoom(
        ctx: click.Context,
        network_pkl: Union[str, os.PathLike],
        seed: int,
        starting_image: Union[str, os.PathLike],
        class_idx: Optional[int],  # TODO: conditional model
        learning_rate: float,
        iterations: int,
        layers: List[str],
        norm_model_layers: bool,
        sqrt_norm_model_layers: bool,
        num_octaves: int,
        octave_scale: float,
        unzoom_octave: bool,
        pixel_zoom: int,
        rotation_deg: float,
        translate_x: int,
        translate_y: int,
        fps: int,
        duration_sec: float,
        reverse_video: bool,
        outdir: Union[str, os.PathLike],
        description: str,
):
    print(f'Loading networks from "{network_pkl}"...')
    # Define the model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        D = legacy.load_network_pkl(f)['D'].eval().requires_grad_(False).to(device)  # type: ignore

    # Get the model resolution (for resizing the starting image if needed)
    model_resolution = D.img_resolution
    zoom_size = model_resolution - 2 * pixel_zoom

    # We will use the features of the Discriminator, on the layer specified by the user
    model = DiscriminatorFeatures(D).requires_grad_(False).to(device)

    # Get the image and image name
    image, starting_image = get_image(seed=seed, starting_image=starting_image, image_size=model_resolution)

    # Make the run dir in the specified output directory
    desc = 'discriminator-dream-zoom'
    desc = f'{desc}-{description}' if len(description) != 0 else desc
    run_dir = make_run_dir(outdir, desc)

    # Save the configuration used
    ctx.obj = {
        'network_pkl': network_pkl,
        'synthesis_options': {
            'seed': seed,
            'starting_image': starting_image,
            'class_idx': class_idx,
            'learning_rate': learning_rate,
            'iterations': iterations
        },
        'layer_options': {
            'layers': layers,
            'norm_model_layers': norm_model_layers,
            'sqrt_norm_model_layers': sqrt_norm_model_layers
        },
        'octaves_options': {
            'num_octaves': num_octaves,
            'octave_scale': octave_scale,
            'unzoom_octave': unzoom_octave
        },
        'frame_manipulation_options': {
            'pixel_zoom': pixel_zoom,
            'rotation_deg': rotation_deg,
            'translate_x': translate_x,
            'translate_y': translate_y,
        },
        'video_options': {
            'fps': fps,
            'duration_sec': duration_sec,
            'reverse_video': reverse_video,
        },
        'extra_parameters': {
            'outdir': run_dir,
            'description': description
        }
    }
    # Save the run configuration
    save_config(ctx=ctx, run_dir=run_dir)

    num_frames = int(np.rint(duration_sec * fps))  # Number of frames for the video
    n_digits = int(np.log10(num_frames)) + 1       # Number of digits for naming each frame

    # Save the starting image
    image.save(os.path.join(run_dir, f'dreamed_{0:0{n_digits}d}.jpg'))

    for idx, frame in enumerate(tqdm(range(num_frames), desc='Dreaming...', unit='frame')):
        # Zoom in after the first frame
        if idx > 0:
            image = crop_resize_rotate(image, crop_size=zoom_size, new_size=model_resolution,
                                       rotation_deg=rotation_deg, translate_x=translate_x, translate_y=translate_y)
        # Extract deep dream image
        dreamed_image = deep_dream(image, model, model_resolution, layers=layers, normed=norm_model_layers,
                                   sqrt_normed=sqrt_norm_model_layers, iterations=iterations,
                                   lr=learning_rate, octave_scale=octave_scale, num_octaves=num_octaves,
                                   unzoom_octave=unzoom_octave, disable_inner_tqdm=True)

        # Save the resulting image and initial image
        filename = f'dreamed_{idx + 1:0{n_digits}d}.jpg'
        Image.fromarray(dreamed_image, 'RGB').save(os.path.join(run_dir, filename))

        # Now, the dreamed image is the starting image
        image = Image.fromarray(dreamed_image, 'RGB')

    # Save the final video
    print('Saving video...')
    stream = ffmpeg.input(os.path.join(run_dir, 'dreamed_*.jpg'), pattern_type='glob', framerate=fps)
    stream = ffmpeg.output(stream, os.path.join(run_dir, 'dream-zoom.mp4'), crf=20, pix_fmt='yuv420p')
    ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)  # I dislike ffmpeg's console logs, so I turn them off

    if reverse_video:
        # Save the reversed video apart from the original one, so the user can compare both
        stream = ffmpeg.input(os.path.join(run_dir, 'dream-zoom.mp4'))
        stream = stream.video.filter('reverse')
        stream = ffmpeg.output(stream, os.path.join(run_dir, 'dream-zoom_reversed.mp4'), crf=20, pix_fmt='yuv420p')
        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)  # ibidem


# ----------------------------------------------------------------------------


def normalize_image(image: Union[PIL.Image.Image, np.ndarray]) -> np.ndarray:
    """Change dynamic range of an image from [0, 255] to [-1, 1]"""
    image = np.array(image, dtype=np.float32)
    image = image / 127.5 - 1.0
    return image


def get_video_information(mp4_filename: Union[str, os.PathLike],
                          max_length_seconds: float = None,
                          starting_second: float = 0.0) -> Tuple[int, float, int, int, int, int]:
    """Take a mp4 file and return a list containing each frame as a NumPy array"""
    metadata = skvideo.io.ffprobe(mp4_filename)
    # Get video properties
    fps = int(np.rint(eval(metadata['video']['@avg_frame_rate'])))
    total_video_num_frames = int(metadata['video']['@nb_frames'])
    video_duration = float(metadata['video']['@duration'])
    video_width = int(metadata['video']['@width'])
    video_height = int(metadata['video']['@height'])
    # Maximum number of frames to return (if not provided, return the full video)
    if max_length_seconds is None:
        print('Considering the full video...')
        max_length_seconds = video_duration
    if starting_second != 0.0:
        print('Using part of the video...')
        starting_second = min(starting_second, video_duration)
        max_length_seconds = min(video_duration - starting_second, max_length_seconds)
    max_num_frames = int(np.rint(max_length_seconds * fps))
    max_frames = min(total_video_num_frames, max_num_frames)
    returned_duration = min(video_duration, max_length_seconds)
    # Frame to start from
    starting_frame = int(np.rint(starting_second * fps))

    return fps, returned_duration, starting_frame, max_frames, video_width, video_height


def get_video_frames(mp4_filename: Union[str, os.PathLike],
                     run_dir: Union[str, os.PathLike],
                     starting_frame: int,
                     max_frames: int,
                     center_crop: bool = False,
                     save_selected_frames: bool = False) -> np.ndarray:
    """Get all the frames of a video as a np.ndarray"""
    # DEPRECATED
    print('Getting video frames...')
    frames = skvideo.io.vread(mp4_filename)  # TODO: crazy things with scikit-video
    frames = frames[starting_frame:min(starting_frame + max_frames, len(frames)), :, :, :]
    frames = np.transpose(frames, (0, 3, 2, 1))  # NHWC => NCWH
    if center_crop:
        frame_width, frame_height = frames.shape[2], frames.shape[3]
        min_side = min(frame_width, frame_height)
        frames = frames[:, :, (frame_width - min_side) // 2:(frame_width + min_side) // 2, (frame_height - min_side) // 2:(frame_height + min_side) // 2]

    if save_selected_frames:
        skvideo.io.vwrite(os.path.join(run_dir, 'selected_frames.mp4'), np.transpose(frames, (0, 3, 2, 1)))
    return frames


# Here for now, might move to its own file if encoding with the Discriminator results fruitless
@main.command(name='visual-reactive')
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
# Encoder options
@click.option('--encoder', type=click.Choice(['discriminator', 'vgg16', 'clip']), help='Choose the model to encode each frame into the latent space Z.', default='discriminator', show_default=True)
@click.option('--vgg16-layer', type=click.Choice(['conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'adavgpool', 'fc1', 'fc2']), help='Choose the layer to use from VGG16 (if used as encoder)', default='adavgpool', show_default=True)
# Source video options
@click.option('--source-video', '-video', 'video_file', type=click.Path(exists=True, dir_okay=False), help='Path to video file', required=True)
@click.option('--max-video-length', type=click.FloatRange(min=0.0, min_open=True), help='How many seconds of the video to take (from the starting second)', default=None, show_default=True)
@click.option('--starting-second', type=click.FloatRange(min=0.0), help='Second to start the video from', default=0.0, show_default=True)
@click.option('--frame-transform', type=click.Choice(['none', 'center-crop', 'resize']), help='Transform to apply to the individual frame.')
@click.option('--center-crop', is_flag=True, help='Center-crop each frame of the video')
@click.option('--save-selected-frames', is_flag=True, help='Save the selected frames of the input video after the selected transform')
# Synthesis options
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--new-center', type=parse_new_center, help='New center for the W latent space; a seed (int) or a path to a dlatent (.npy/.npz)', default=None)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
# Video options
@click.option('--compress', is_flag=True, help='Add flag to compress the final mp4 file with ffmpeg-python (same resolution, lower file size)')
# Extra parameters for saving the results
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out'), show_default=True, metavar='DIR')
@click.option('--description', '-desc', type=str, help='Description name for the directory path to save results', default='', show_default=True)
def visual_reactive_interpolation(
        ctx: click.Context,
        network_pkl: Union[str, os.PathLike],
        encoder: str,
        vgg16_layer: str,
        video_file: Union[str, os.PathLike],
        max_video_length: float,
        starting_second: float,
        frame_transform: str,
        center_crop: bool,
        save_selected_frames: bool,
        truncation_psi: float,
        new_center: Tuple[str, Union[int, np.ndarray]],
        noise_mode: str,
        outdir: Union[str, os.PathLike],
        description: str,
        compress: bool,
        smoothing_sec: float = 0.1  # For Gaussian blur; the lower, the faster the reaction; higher leads to more generated frames being the same
):
    print(f'Loading networks from "{network_pkl}"...')

    # Define the model (load both D, G, and the features of D)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if encoder == 'discriminator':
        print('Loading Discriminator and its features...')
        with dnnlib.util.open_url(network_pkl) as f:
            D = legacy.load_network_pkl(f)['D'].eval().requires_grad_(False).to(device)  # type: ignore

        D_features = DiscriminatorFeatures(D).requires_grad_(False).to(device)
        del D
    elif encoder == 'vgg16':
        print('Loading VGG16 and its features...')
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            vgg16 = torch.jit.load(f).eval().to(device)

        vgg16_features = VGG16FeaturesNVIDIA(vgg16).requires_grad_(False).to(device)
        del vgg16

    elif encoder == 'clip':
        print('Loading CLIP model...')
        try:
            import clip
        except ImportError:
            raise ImportError('clip not installed! Install it via "pip install git+https://github.com/openai/CLIP.git"')
        model, preprocess = clip.load('ViT-B/32', device=device)
        model = model.requires_grad_(False)  # Otherwise OOM



    print('Loading Generator...')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].eval().requires_grad_(False).to(device)  # type: ignore

    if new_center is None:
        # Stick to the tracked center of W during training
        w_avg = G.mapping.w_avg
    else:
        new_center, new_center_value = new_center
        # We get the new center using the int (a seed) or recovered dlatent (an np.ndarray)
        if isinstance(new_center_value, int):
            new_center = f'seed_{new_center}'
            w_avg = get_w_from_seed(G, device, new_center_value, truncation_psi=1.0)  # We want the pure dlatent
        elif isinstance(new_center_value, np.ndarray):
            w_avg = torch.from_numpy(new_center_value).to(device)
        else:
            ctx.fail('Error: New center has strange format! Only an int (seed) or a file (.npy/.npz) are accepted!')

    # Create the run dir with the given name description; add slowdown if different than the default (1)
    description = 'visual-reactive' if len(description) == 0 else description
    run_dir = make_run_dir(outdir, description)
    # Name of the video
    video_name, _ = os.path.splitext(video_file)
    video_name = video_name.split(os.sep)[-1]  # Get the actual name of the video
    mp4_name = f'visual-reactive_{video_name}'

    # Get all the frames of the video and its properties
    # TODO: resize the frames to the size of the network (G.img_resolution)
    fps, max_video_length, starting_frame, max_frames, width, height = get_video_information(video_file,
                                                                                             max_video_length,
                                                                                             starting_second)

    videogen = skvideo.io.vreader(video_file)
    fake_dlatents = list()
    if save_selected_frames:
        # skvideo.io.vwrite sets FPS=25, so we have to manually enter it via FFmpeg
        # TODO: use only ffmpeg-python
        writer = skvideo.io.FFmpegWriter(os.path.join(run_dir, f'selected-frames_{video_name}.mp4'),
                                         inputdict={'-r': str(fps)})

    for idx, frame in enumerate(tqdm(videogen, desc=f'Getting frames+latents of "{video_name}"', unit='frames')):
        # Only save the frames that the user has selected
        if idx < starting_frame:
            continue
        if idx > starting_frame + max_frames:
            break

        if center_crop:
            frame_width, frame_height = frame.shape[1], frame.shape[0]
            min_side = min(frame_width, frame_height)
            frame = frame[(frame_height - min_side) // 2:(frame_height + min_side) // 2, (frame_width - min_side) // 2:(frame_width + min_side) // 2, :]

        if save_selected_frames:
            writer.writeFrame(frame)

        # Get fake latents
        if encoder == 'discriminator':
            frame = normalize_image(frame)  # [0, 255] => [-1, 1]
            frame = torch.from_numpy(np.transpose(frame, (2, 1, 0))).unsqueeze(0).to(device)  # HWC => CWH => NCWH, N=1
            fake_z = D_features.get_layers_features(frame, layers=['fc'])[0]

        elif encoder == 'vgg16':
            preprocess = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
            frame = preprocess(frame).unsqueeze(0).to(device)
            fake_z = vgg16_features.get_layers_features(frame, layers=[vgg16_layer])[0]
            fake_z = fake_z.view(1, 512, -1).mean(2)

        elif encoder == 'clip':
            frame = Image.fromarray(frame)  # [0, 255]
            frame = preprocess(frame).unsqueeze(0).to(device)
            fake_z = model.encode_image(frame)

        # Normalize the latent so that it's ~N(0, 1)
        # fake_z = fake_z / fake_z.max()
        fake_z = (fake_z - fake_z.mean()) / fake_z.std()

        # Get dlatent
        fake_w = G.mapping(fake_z, None)
        # Truncation trick
        fake_w = w_avg + (fake_w - w_avg) * truncation_psi
        fake_dlatents.append(fake_w)

    if save_selected_frames:
        # Close the video writer
        writer.close()

    # Set the fake_dlatents as a torch tensor; we can't just do torch.tensor(fake_dlatents) as with NumPy :(
    fake_dlatents = torch.cat(fake_dlatents, 0)
    # Smooth out so larger changes in the scene are the ones that affect the generation
    fake_dlatents = torch.from_numpy(nd.gaussian_filter(fake_dlatents.cpu(),
                                                        sigma=[smoothing_sec * fps, 0, 0])).to(device)

    # Auxiliary function for moviepy
    def make_frame(t):
        # Get the frame, dlatent, and respective image
        frame_idx = int(np.clip(np.round(t * fps), 0, len(fake_dlatents) - 1))
        fake_w = fake_dlatents[frame_idx]
        image = w_to_img(G, fake_w, noise_mode)
        # Create grid for this timestamp
        grid = create_image_grid(image, (1, 1))
        # Grayscale => RGB
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2)
        return grid

    # Generate video using the respective make_frame function
    videoclip = moviepy.editor.VideoClip(make_frame, duration=max_video_length)
    videoclip.set_duration(max_video_length)

    # Change the video parameters (codec, bitrate) if you so desire
    final_video = os.path.join(run_dir, f'{mp4_name}.mp4')
    videoclip.write_videofile(final_video, fps=fps, codec='libx264', bitrate='16M')

    # Compress the video (lower file size, same resolution, if successful)
    if compress:
        compress_video(original_video=final_video, original_video_name=mp4_name, outdir=run_dir, ctx=ctx)
    # TODO: merge the videos side by side

    # Save the configuration used
    new_center = 'w_avg' if new_center is None else new_center
    ctx.obj = {
        'network_pkl': network_pkl,
        'encoder_options': {
            'encoder': encoder,
            'vgg16_layer': vgg16_layer,
        },
        'source_video_options': {
            'source_video': video_file,
            'sorce_video_params': {
                'fps': fps,
                'height': height,
                'width': width,
                'length': max_video_length,
                'starting_frame': starting_frame,
                'total_frames': max_frames
            },
            'max_video_length': max_video_length,
            'starting_second': starting_second,
            'frame_transform': frame_transform,
            'center_crop': center_crop,
            'save_selected_frames': save_selected_frames
        },
        'synthesis_options': {
            'truncation_psi': truncation_psi,
            'new_center': new_center,
            'noise_mode': noise_mode,
            'smoothing_sec': smoothing_sec
        },
        'video_options': {
            'compress': compress
        },
        'extra_parameters': {
            'outdir': run_dir,
            'description': description
        }
    }

    save_config(ctx=ctx, run_dir=run_dir)

# ----------------------------------------------------------------------------


if __name__ == '__main__':
    main()


# ----------------------------------------------------------------------------
