import click
from typing import Union, Optional
import os

from torch_utils.gen_utils import parse_slowdown, parse_fps, make_run_dir, w_to_img, create_image_grid, save_config, \
    double_slowdown, compress_video

import numpy as np
import scipy
import torch

import dnnlib
import legacy

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import moviepy.editor

# ----------------------------------------------------------------------------


@click.group()
def main():
    pass


# ----------------------------------------------------------------------------


@main.command(name='mirror-video')
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seed', type=int, help='Random seed to use', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--slowdown', type=parse_slowdown, help='Slow down the video by this amount; will be approximated to the nearest power of 2', default='1', show_default=True)
@click.option('--duration-sec', '-sec', type=float, help='Duration length of the video', default=30.0, show_default=True)
@click.option('--fps', type=parse_fps, help='Video FPS.', default=30, show_default=True)
@click.option('--compress', is_flag=True, help='Add flag to compress the final mp4 file with ffmpeg-python (same resolution, lower file size)')
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out'), show_default=True, metavar='DIR')
@click.option('--description', '-desc', type=str, help='Description name for the directory path to save results', default='', show_default=True)
def mirror_random_video(
        ctx: click.Context,
        network_pkl: Union[str, os.PathLike],
        seed: Optional[int],
        truncation_psi: float,
        class_idx: Optional[int],
        noise_mode: str,
        slowdown: int,
        duration_sec: float,
        fps: int,
        outdir: Union[str, os.PathLike],
        description: str,
        compress: bool,
        smoothing_sec: Optional[float] = 3.0  # for Gaussian blur; won't be a parameter, change at own risk
):
    """
        Generate a random interpolation video using a pretrained network.

        Examples:

        \b
        # Generate a 30-second long, truncated MetFaces video at 30 FPS:
        python generate.py random-video --seed=0  --trunc=0.7 \\
            --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """
    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    # Create the run dir with the given name description; add slowdown if different than the default (1)
    description = 'mirror-video' if len(description) == 0 else description
    description = f'{description}-{slowdown}xslowdown' if slowdown != 1 else description
    run_dir = make_run_dir(outdir, description)

    # Number of frames in the video and its total duration in seconds
    num_frames = int(np.rint(duration_sec * fps))
    total_duration = duration_sec * slowdown

    print('Generating latent vectors...')
    # TODO: let another helper function handle each case, we will use it for the grid
    # If there's more than one seed provided and the shape isn't specified by the user
    grid_size = (2, 1)
    # Shape of the latents to generate
    shape = [num_frames, G.z_dim]
    # Get the z latents
    all_latents = np.random.RandomState(seed).randn(*shape).astype(np.float32)

    # Let's smooth out the random latents so that now they form a loop (and are correctly generated in a 512-dim space)
    all_latents = scipy.ndimage.gaussian_filter(all_latents, sigma=[smoothing_sec * fps, 0], mode='wrap')
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))

    # Save the configuration used
    ctx.obj = {
        'network_pkl': network_pkl,
        'seed': seed,
        'truncation_psi': truncation_psi,
        'class_idx': class_idx,
        'noise_mode': noise_mode,
        'slowdown': slowdown,
        'duration_sec': duration_sec,
        'video_fps': fps,
        'run_dir': run_dir,
        'description': description,
        'compress': compress,
        'smoothing_sec': smoothing_sec
    }
    # Save the run configuration
    save_config(ctx=ctx, run_dir=run_dir)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    # Name of the video will change if we use slowdown
    mp4_name = f'mirror-seed-{seed}-{slowdown}xslowdown' if slowdown != 1 else f'mirror-seed-{seed}'

    # Let's slowdown the video, if so desired
    while slowdown > 1:
        all_latents, duration_sec, num_frames = double_slowdown(latents=all_latents,
                                                                duration=duration_sec,
                                                                frames=num_frames)
        slowdown //= 2

    # Map to W and do truncation trick
    w_avg = G.mapping.w_avg
    all_w = G.mapping(torch.from_numpy(all_latents).to(device), None)
    all_w = w_avg + (all_w - w_avg) * truncation_psi

    # Mirror
    w_mirror = w_avg + (all_w - w_avg) * (-truncation_psi)

    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * fps), 0, num_frames - 1))
        w = all_w[frame_idx].unsqueeze(0)
        w_m = w_mirror[frame_idx].unsqueeze(0)
        dlatent = torch.cat((w, w_m), axis=0)
        # Get the images with the labels
        images = w_to_img(G, dlatent, noise_mode)
        # Generate the grid for this timestamp
        grid = create_image_grid(images, grid_size)
        # Grayscale => RGB
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2)
        return grid

    # Generate video using the respective make_frame function
    videoclip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    videoclip.set_duration(total_duration)

    # Change the video parameters (codec, bitrate) if you so desire
    final_video = os.path.join(run_dir, f'{mp4_name}.mp4')
    videoclip.write_videofile(final_video, fps=fps, codec='libx264', bitrate='16M')

    # Compress the video (lower file size, same resolution)
    if compress:
        compress_video(original_video=final_video, original_video_name=mp4_name, outdir=run_dir, ctx=ctx)


# ----------------------------------------------------------------------------


def project():
    a = np.random.RandomState(0).randn(1, 512)
    a = G.mapping(torch.from_numpy(a).to(device), None)
    b = torch.from_numpy(np.load('path')).to(device)

    proj_ab = b * torch.sum(a * b) / (b.square().sum())
    proj_perp = a - proj_ab

    img_a = w_to_img(G, a)[0]
    img_b = w_to_img(G, b)[0]
    img_projab = w_to_img(G, proj_ab)[0]
    img_projperp = w_to_img(G, proj_perp)[0]


# ----------------------------------------------------------------------------


def n_pendulum():
    pass


# ----------------------------------------------------------------------------


def circular():
    pass


# ----------------------------------------------------------------------------


if __name__ == '__main__':
    main()
