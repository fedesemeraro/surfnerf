import copy
import torch
import torchvision.transforms
import numpy as np
import skimage.io as io
def_dtype = np.float32


def trans_t(t): return torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()


def rot_phi(phi): return torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()


def rot_theta(th): return torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi) @ c2w
    c2w = rot_theta(theta) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def read_depth_map(depth_map_path, df=1):
    """
    Open one-band depth map .tif file.

    Parameters:
    depth_map_path (string): Path to depth map
    df (int): Downscaling factor for depth map (optional)

    Outputs:
    img (array[N,M]): Depth map of size NxM
    """
    img = torch.Tensor(io.imread(depth_map_path))
    if df > 1:
        img = torchvision.transforms.Resize((img.shape[0] // df, img.shape[1] // df), antialias=True)(img)
    return img


def generate_images(image_path, view_indices, downscale_factor=1):
    """

    Create a list of downsampled tensors [H, W, bands] from .tif images.

    Parameters:
    image_path (string): Path to images
    view_indices (list(string)): List of view indices to be read
    downscale_factor (int): Downscaling factor for depth map (optional)

    Outputs:
    imgs (list(array[N,M,3])): List of images of size NxMx3, last dimension is RGB

    """
    imgs = []
    for view_index in view_indices:
        image_name = f"{image_path}_{view_index}.tif"
        im = torch.Tensor(io.imread(image_name).transpose(2, 0, 1))
        H, W = im.shape[1], im.shape[2]
        H = H // downscale_factor
        W = W // downscale_factor
        im = torchvision.transforms.Resize((H, W), antialias=True)(im)
        im = torch.permute(im, [1, 2, 0])  # gdal puts bands first
        imgs.append(im)
    return imgs, H, W


def read_image_metadata(arg_dict, sep=' ', ids=None):
    """

    Convert metadata file to list of poses (c2w matrices), focals, view and light directions.

    Parameters:
    arg_dict (dict): Metadata path, sampling distance and image downscale factor
    sep (char): Separating character in metadata file
    ids (list(string)): List of image IDs to read metadata from

    Outputs:
    poses (list(array[4,4])): List of camera to world matrices
    focals (list(float)): List of focal lengths
    view_dirs(list(array[1,2])): List of viewing angles
    light_dirs(list(array[1,2])): List of light angles

    """
    poses, focals, view_dirs, light_dirs = [], [], [], []
    with open(f"{arg_dict['data.md.path']}", 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            params = line.split(sep)
            # Image ID
            im_id = params[0]
            if (ids == None) or (im_id in ids):
                # True image sampling distance
                image_sd = arg_dict['data.image.sd'] * arg_dict['data.image.df']
                # Convert radius to pixel units
                radius = def_dtype(params[1]) / image_sd
                # Viewing directions
                az = def_dtype(params[2])
                el = def_dtype(params[3])
                # Light source directions
                az_ls = def_dtype(params[4])
                el_ls = def_dtype(params[5])
                poses.append(pose_spherical(az, -el, radius))
                if arg_dict['rend.unzoom']:
                    # Special case to handle satellite images, unzoom by 1/sin(elevation) to compensate
                    # https://github.com/esa/snerf/blob/37ebd919f9a663182b5e40db7e5143705d2f9125/snerf/render.py#L72
                    focals.append(radius*np.sin(el))
                else:
                    focals.append(radius)
                view_dirs.append(torch.reshape(torch.Tensor([az, el]), [1, 2]))
                light_dirs.append(torch.reshape(torch.Tensor([az_ls, el_ls]), [1, 2]))
    return poses, focals, view_dirs, light_dirs


def generate_dataset(arg_dict):
    """
    Generate data set dictionary by splitting images and metadata into train and test sets with viewing and light parameters.

    Parameters:
    arg_dict (dict): Global configuration variables

    Outputs:
    dataset (dict): Image IDs, image data, pose, focal, viewing and light parameters split into train and test sets, and depth map.
    """
    ret = {}
    ret['train_id'] = arg_dict['data.train_id']
    ret['test_id'] = arg_dict['data.test_id']
    ret['train_imgs'], H, W = generate_images(arg_dict['data.image.path'], arg_dict['data.train_id'], arg_dict['data.image.df'])
    ret['test_imgs'], H, W = generate_images(arg_dict['data.image.path'], arg_dict['data.test_id'], arg_dict['data.image.df'])
    ret['train_poses'], ret['train_focals'], ret['train_view_dirs'], ret['train_light_dirs'] = \
        read_image_metadata(arg_dict, sep=' ', ids=arg_dict['data.train_id'])
    ret['test_poses'], ret['test_focals'], ret['test_view_dirs'], ret['test_light_dirs'] = \
        read_image_metadata(arg_dict, sep=' ', ids=arg_dict['data.test_id'])
    ret['depth_map'] = read_depth_map(arg_dict['data.depth.path'], df=arg_dict['data.depth.df'])

    train_imgs = ret['train_imgs']
    test_imgs = ret['test_imgs']

    train_imgs.extend(test_imgs)

    img_counter = 0
    train_id, test_id = [], []
    for i, _ in enumerate(ret['train_id']):
        train_id.append(i)
        img_counter += 1
    for i, _ in enumerate(ret['test_id']):
        test_id.append(img_counter)
        img_counter += 1

    focals = copy.deepcopy(ret['train_focals'])
    focals.extend(ret['test_focals'])

    all_poses = copy.deepcopy(ret['train_poses'])
    all_poses.extend(ret['test_poses'])

    return torch.stack(train_imgs), torch.stack(all_poses), torch.stack(ret['test_poses']), [H, W, focals], [train_id, test_id, test_id], \
           torch.cat([torch.cat(ret['train_light_dirs']), torch.cat(ret['test_light_dirs'])]), ret['depth_map']


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', default=True,
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    # blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=500000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=1000,
                        help='frequency of render_poses video saving')

    # parser.add_argument('--dataset_type', type=str)
    # parser.add_argument('--config', is_config_file=True, help='Config file path.')
    # Dataset arguments
    parser.add_argument('--data.image.path', type=str,
                        default='./data/', help='Path that contains all images.')
    parser.add_argument('--data.image.df', type=int,
                        default=1, help='Image downsample factor.')
    parser.add_argument('--data.image.sd', type=np.float32,
                        default=1.0, help='Image sampling distance.')
    parser.add_argument('--data.depth.path', type=str,
                        default='./data/depth.tif', help='Depth map path.')
    parser.add_argument('--data.depth.df', type=int,
                        default=1, help='Depth map downsample factor.')
    parser.add_argument('--data.md.path', type=str,
                        default='./data/md.txt', help='Metadata file path.')
    parser.add_argument('--data.train_id', type=str,
                        nargs='+', help='ID of train images.')
    parser.add_argument('--data.test_id', type=str,
                        nargs='+', help='ID of test images.')
    # Model arguments
    parser.add_argument('--model.ins.light', type=bool, default=False,
                        help='Use light directions as network inputs.')
    parser.add_argument('--model.ins.views', type=bool,
                        default=False, help='Use view directions as inputs.')
    parser.add_argument('--model.outs.shad', type=bool, default=False,
                        help='Directional light source visibility function as network output.')
    parser.add_argument('--model.outs.sky', type=bool, default=False,
                        help='Diffuse light color as network output.')
    parser.add_argument('--model.act', type=str, default='relu',
                        help='Neuron activation function [relu, sin].')
    parser.add_argument('--model.act.sin.w0', type=np.float32,
                        default=30.0, help='Initial wavelength for SIREN.')
    parser.add_argument('--model.sigma.depth', type=int, default=8,
                        help='Number of fully-connected layers for sigma function.')
    parser.add_argument('--model.sigma.width', type=int,
                        default=256, help='Width of layers for sigma function.')
    parser.add_argument('--model.sigma.skips', type=int,
                        nargs='+', default=[], help='Skip connections.')
    parser.add_argument('--model.c.depth', type=int, default=1,
                        help='Number of fully-connected layers for color function.')
    parser.add_argument('--model.c.width', type=int, default=128,
                        help='Width of layers for color function.')
    parser.add_argument('--model.shad.depth', type=int, default=4,
                        help='Number of fully-connected layers for shadow function.')
    parser.add_argument('--model.shad.width', type=int,
                        default=128, help='Width of layers for shadow function.')
    parser.add_argument('--model.emb.pos', type=int, default=0,
                        help='Length of on-axis positional encoding. 0 to disable.')
    parser.add_argument('--model.emb.dir', type=int, default=0,
                        help='Length of on-axis directional encoding. 0 to disable.')

    # Rendering arguments
    parser.add_argument('--rend.nsamples', type=int, default=64,
                        help='Number of samples for coarse rendering.')
    parser.add_argument('--rend.nimportance', type=int, default=64,
                        help='Number of samples for fine rendering, 0 to disable.')
    parser.add_argument('--rend.mode', type=str, default='nf',
                        help='Rendering mode : near-far or altitude sampling [nf, alt].')
    parser.add_argument('--rend.mode.nf.near', type=np.float32,
                        default=3.0, help='Near point (px).')
    parser.add_argument('--rend.mode.nf.far', type=np.float32,
                        default=10.0, help='Far point (px).')
    parser.add_argument('--rend.mode.alt.max', type=np.float32,
                        default=30.0, help='Max alt (px).')
    parser.add_argument('--rend.mode.alt.min', type=np.float32,
                        default=-30.0, help='Min alt(px).')
    parser.add_argument('--rend.unzoom', type=bool, default=False,
                        help='Special unzoom mode for off-nadir EO images.')
    parser.add_argument('--rend.rescale', type=np.float32, default=None,
                        help='Largest scene extent in pixel units. Calculated based on image sizes if not provided.')

    # Training arguments
    parser.add_argument('--train.n_epoch', type=int, default=200,
                        help='Number of iterations for training.')
    parser.add_argument('--train.n_rand', type=int, default=1024,
                        help='Number of random rays at each iteration.')
    parser.add_argument('--train.lr.init', type=np.float32,
                        default=1e-4, help='Initial learning rate.')
    parser.add_argument('--train.lr.decay', type=np.float32, default=0.2,
                        help='Learning rate decay over entire training.')
    parser.add_argument('--train.noise.sigma', type=np.float32, default=10.0,
                        help='Standard deviation of sigma pre-activation noise.')
    parser.add_argument('--train.noise.shad', type=np.float32, default=1.0,
                        help='Standard deviation of shadow function pre-activation noise.')
    parser.add_argument('--train.shad', type=bool,
                        default=False, help='Use solar correction rays.')
    parser.add_argument('--train.shad.lambda', type=np.float32,
                        default=0.1, help='Weight of solar correction loss.')
    parser.add_argument('--train.shad.df', type=int, default=1,
                        help='Downsample factor of solar correction rays compared to image rays, 1 to sample at same resolution.')
    parser.add_argument('--train.shad.custom', type=str, default='none',
                        help='Type of custon solar correction rays [linear, rectangle].')
    parser.add_argument('--train.shad.custom.bounds.start', type=np.float32, nargs='+', default=[160.0, 40.0],
                        help='Start point (az, el) in deg.')
    parser.add_argument('--train.shad.custom.bounds.end', type=np.float32, nargs='+', default=[100.0, 80.0],
                        help='End point (az, el) in deg.')
    parser.add_argument('--train.shad.custom.bounds.samp', type=int, nargs='+', default=[10, 1],
                        help='Sampling scheme for solar correction rays. If linear, 1st dimension is number of samples.')
    # Output arguments
    parser.add_argument('--out.iplot', type=int, default=0,
                        help='Frequency of test evaluation for output, 0 to disable.')
    parser.add_argument('--out.path', type=str,
                        default='./results/', help='Path to save outputs.')
    parser.add_argument('--savelogs', type=bool,
                        default=True, help='Save training logs')

    # Hardware options
    parser.add_argument('--gpu', type=str, default='1', help='GPU to use.')

    # turn on or off S-Nerf capability
    parser.add_argument('--snerf', type=bool,
                        default=False, help='Turn on the S-Nerf algorithm')

    return parser
