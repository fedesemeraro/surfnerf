import copy
import os
from PIL import Image
import imageio
import skimage.io as io
from skimage.metrics import structural_similarity
import time
from tqdm import tqdm, trange
from model.nerf_helpers import create_nerf, get_rays, ndc_rays, to8b, sample_pdf, get_rays_np, img2mse, mse2psnr, semi_diagonal
from model.load_satdata import generate_dataset, config_parser, pose_spherical
import numpy as np
import torch.nn.functional as F
import torch
import random

# make it reproducible
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
DEBUG = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, pose_indices=None, light_dir=None):
    """
    This rendering is only called when mode is already trained
    """
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []
    depths = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        if isinstance(focal, list) and pose_indices is not None:
            K = np.array([
                [focal[pose_indices[i]], 0, 0.5*W],
                [0, focal[pose_indices[i]], 0.5*H],
                [0, 0, 1]
            ])
        if light_dir is not None:
            rgb, _, _, disp, acc, extras = render(
                H, W, K, chunk=chunk, c2w=c2w[:3, :4], light_dir=light_dir[pose_indices[i]], **render_kwargs)
        else:
            rgb, disp, acc, extras = render(
                H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        depths.append(focal[pose_indices[i]] -
                      extras['depth_map'].cpu().numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    depths = np.stack(depths, 0)

    return rgbs, disps, depths


def calculate_altitude_rmse(depth_map, file_prefix, arg_dict, render_kwargs_test):
    SR = 0.5 * arg_dict['data.depth.df']
    radius = 617000.0 / SR
    az, el = np.pi, np.pi/2
    pose = pose_spherical(az, -el, radius)
    H, W = depth_map.size()
    K = np.array([
        [radius, 0, 0.5*W],
        [0, radius, 0.5*H],
        [0, 0, 1]
    ])
    kwargs = copy.deepcopy(render_kwargs_test)
    kwargs['far'] = arg_dict['rend.mode.alt.max'] / SR
    kwargs['near'] = arg_dict['rend.mode.alt.min'] / SR
    light_dir = torch.reshape(torch.Tensor(
        [np.deg2rad(100.0), np.deg2rad(80.0)]), [1, 2])
    with torch.no_grad():
        if arg_dict['snerf']:
            rgb, sun, shadow, disp, acc, extras = render(H, W, K, c2w=pose, chunk=arg_dict['chunk'],
                                                         light_dir=light_dir, **kwargs)
        else:
            rgb, disp, acc, extras = render(
                H, W, K, c2w=pose, chunk=arg_dict['chunk'], **kwargs)

    depth = radius * SR - extras['depth_map'] * SR
    alt_abs_diff = torch.abs(depth - depth_map)
    alt_rmse = torch.sqrt(torch.mean(alt_abs_diff ** 2))
    alt_mae = torch.mean(alt_abs_diff)

    alt_comp = torch.mean(torch.where(alt_abs_diff < 1, 1.0, 0.0))
    io.imsave(file_prefix + 'ground_truth.tiff', depth_map.cpu().detach().numpy())
    io.imsave(file_prefix + 'dem.tiff', depth.cpu().detach().numpy())
    io.imsave(file_prefix + 'disp.tiff', disp.cpu().detach().numpy())
    return alt_rmse, alt_mae, alt_comp


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True, near=0., far=1., use_viewdirs=False, c2w_staticcam=None, light_dir=None, **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
      light_dir: Sun's azimuth and elevation of sun (S-Nerf)
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None and K is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if light_dir is not None:
        update_dict = {'light_dir': light_dir}
        kwargs.update(update_dict)

    if use_viewdirs:
        if c2w_staticcam is not None and K is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        # provide ray directions as input
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * \
        torch.ones_like(rays_d[..., :1]), far * \
        torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    if light_dir is not None:
        k_extract = ['rgb_map', 'sun_map', 'shadow_map', 'disp_map', 'acc_map']
    else:
        k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}

    return ret_list + [ret_dict]


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


# S-Nerf altitude sampling
def sample_alt(rays_o, rays_d, bounds, N_samples):
    """
    Sample along the rays, using the altitude bounds.

    Parameters:
    rays_o (Tensor[H*W, 3]): rays origin.
    rays_d (Tensor[H*W, 3]): rays direction.
    bounds (float, float): minimum and maximum altitude.
    N_samples (int): number of samples along the rays.

    Returns:
    z_vals (Tensor[H*W, N_samples]): depth values along the rays.
    """
    alt_min, alt_max = bounds[0][0], bounds[1][0]
    alt_vals = torch.linspace(
        alt_max.detach()[0], alt_min.detach()[0], N_samples)
    z_vals = (alt_vals - rays_o[..., None, 2]) / rays_d[..., None, 2]
    return z_vals


def render_rays(ray_batch, network_fn, network_query_fn, N_samples, retraw=False, lindisp=False, perturb=0.,
                N_importance=0, network_fine=None, white_bkgd=False, raw_noise_std=0., verbose=False, pytest=False, alt_z=False, light_dir=None, norm=1.0):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
      alt_z: bool. If True, use altitude based sampling method for rays (S-Nerf)
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if alt_z:
        # special case for S-Nerf
        z_vals = sample_alt(rays_o, rays_d, [near, far], N_samples)
    elif not lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
        z_vals[..., :, None]  # [N_rays, N_samples, 3]

    # raw = run_network(pts)
    if light_dir is not None:
        raw = network_query_fn(pts, viewdirs, light_dir, network_fn)
        rgb_map, sun_map, shadow_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, snerf=True, alt_z=alt_z, norm=norm
        )
    else:
        raw = network_query_fn(pts, viewdirs, network_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, alt_z=alt_z, norm=norm
        )
    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0, depth_map_0 = rgb_map, disp_map, acc_map, depth_map

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        if alt_z:
            # Add first and last points to fix edge cases where weights is 1 only on the first or last element
            # Not the case for near-far rendering where the final distance is "infinity"
            z_vals_mid = torch.concat(
                [z_vals[..., 0:1], z_vals_mid, z_vals[..., -1:]], -1)
            z_samples = sample_pdf(z_vals_mid, weights, N_importance, det=(
                perturb == 0.), pytest=pytest)
        else:
            z_samples = sample_pdf(
                z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine

        if light_dir is not None:
            raw = network_query_fn(pts, viewdirs, light_dir, network_fn)
            rgb_map, sun_map, shadow_map, disp_map, acc_map, weights, depth_map = raw2outputs(
                raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, snerf=True, alt_z=alt_z, norm=norm
            )
        else:
            raw = network_query_fn(pts, viewdirs, run_fn)
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
                raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, alt_z=alt_z, norm=norm
            )

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map,
           'acc_map': acc_map, 'depth_map': depth_map}
    if light_dir is not None:
        ret['sun_map'] = sun_map
        ret['shadow_map'] = shadow_map

    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['depth0'] = depth_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")
    return ret


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False, snerf=False, alt_z=False, norm=1.0):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    def raw2alpha(raw, dists, act_fn=F.relu):
        return 1. - torch.exp(-act_fn(raw)*dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    if alt_z:
        # print(norm, dists.size(), rays_d[..., None, :].size(), torch.norm(rays_d[..., None, :], dim=-1), dists, norm)
        dists /= norm
        # Replicate last distance as distance for last point
        dists = torch.cat([dists, dists[..., -2:-1]], -
                          1)  # [N_rays, N_samples]
    else:
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(
            dists[..., :1].shape)], -1)  # [N_rays, N_samples]
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

    if snerf and raw.shape[2] == 8:  # snerf = True & use_viewdirs = False
        sun = torch.sigmoid(raw[..., 4])
        sky = torch.sigmoid(raw[..., 5:])

    elif snerf and raw.shape[2] == 9:  # snerf = True & use_viewdirs = True
        sun = torch.sigmoid(raw[..., 5])
        sky = torch.sigmoid(raw[..., 6:])

    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    trans = torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * trans
    if alt_z:
        last_weight = 1.0-torch.sum(weights[..., :-1], -1)
        last_weight = torch.reshape(last_weight, [-1, 1])
        weights = torch.concat([weights[..., :-1], last_weight], -1)

    if not snerf:
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
    else:
        # not sure about the white light source, using 1 as of now
        li = sun[..., None] + sky * (1.0-sun[..., None])
        rgb_map = torch.sum(weights[..., None] * rgb * li, -2)
        sun_map = torch.sum(
            (weights[..., None]).detach() * sun[..., np.newaxis], -2)
        shadow_map = torch.mean(torch.square(
            (trans[..., None] + weights[..., None]).detach() - sun[..., None]), -2)

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map),
                            depth_map / (torch.sum(weights, -1)+1e-10))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[..., None])

    if snerf:
        return rgb_map, sun_map, shadow_map, disp_map, acc_map, weights, depth_map
    else:
        return rgb_map, disp_map, acc_map, weights, depth_map


def train():
    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None
    sat_arg_dict = vars(args)
    images, poses, render_poses, hwf, i_split, light_dir, depth_map = generate_dataset(
        sat_arg_dict)
    i_train, i_val, i_test = i_split
    image_sd = sat_arg_dict['data.image.df'] * sat_arg_dict['data.image.sd']
    far = sat_arg_dict['rend.mode.alt.max'] / image_sd
    near = sat_arg_dict['rend.mode.alt.min'] / image_sd

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    if K is None and not isinstance(focal, list):
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(
        args)
    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    if args.dataset_type == 'satdata':
        sat_args = {
            'alt_z': True,
            # Ensure that the scene stays within the [-1, 1] bounds for the network.
            'norm': semi_diagonal(H, W),
        }
        render_kwargs_train.update(sat_args)
        render_kwargs_test.update(sat_args)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
                'test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            if args.snerf:
                rgbs, _, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images,
                                         savedir=testsavedir, render_factor=args.render_factor, pose_indices=i_test, light_dir=light_dir)
            else:
                rgbs, _, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images,
                                         savedir=testsavedir, render_factor=args.render_factor, pose_indices=i_test)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(
                testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = []
        for i in i_train:
            K = np.array([
                [focal[i], 0, 0.5*W],
                [0, focal[i], 0.5*H],
                [0, 0, 1]
            ])
            rays_o, rays_d = get_rays(H, W, K, poses[i, :3, :4])
            rays.append(torch.stack([rays_o, rays_d]))
        rays = torch.stack(rays)  # [N, ro+rd, H, W, 3]

        print('done, concats')
        # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = torch.cat([rays, images[i_train, np.newaxis, ...]], 1)
        # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = torch.permute(rays_rgb, [0, 2, 3, 1, 4])
        # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = torch.reshape(rays_rgb, [-1, 3, 3])
        print('shuffle rays')
        rand_idx = torch.randperm(rays_rgb.shape[0])
        rays_rgb = rays_rgb[rand_idx]

        print('done')
        i_batch = 0

    N_iters = sat_arg_dict['train.n_epoch'] + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    if args.savelogs:
        os.makedirs(os.path.join(basedir, expname), exist_ok=True)
        logfile = os.path.join(basedir, expname, 'training_log.txt')

    start = start + 1
    for i in trange(start, N_iters):

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand]  # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            pose = poses[img_i, :3, :4]

            if isinstance(focal, list):
                K = np.array([
                    [focal[img_i], 0, 0.5*W],
                    [0, focal[img_i], 0.5*H],
                    [0, 0, 1]
                ])
                if args.snerf and sat_arg_dict['train.shad']:
                    light_df = sat_arg_dict['train.shad.df']
                    light_angles = light_dir[img_i]
                    az, el = light_angles[0], light_angles[1]
                    focal_sc = focal[img_i] / light_df
                    pose_sc = pose_spherical(
                        az.cpu().detach().numpy(), -el.cpu().detach().numpy(), focal_sc)
                    H_sc = H // light_df
                    W_sc = W // light_df
                    K_c = np.array([
                        [focal_sc, 0, 0.5*W_sc],
                        [0, focal_sc, 0.5*H_sc],
                        [0, 0, 1]
                    ])

            if N_rand is not None:
                rays_o, rays_d = get_rays(
                    H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
                if args.snerf and sat_arg_dict['train.shad']:
                    sc_rays_o, sc_rays_d = get_rays(
                        H_sc, W_sc, K_c, pose_sc[:3, :4])  # (H_sc, W_sc, 3), (H_sc, W_sc, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                                       torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)), -1)
                    if i == start:
                        print(
                            f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
                    # TODO: support sc train rays
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(
                        0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
                    if args.snerf and sat_arg_dict['train.shad']:
                        coords_sc = torch.stack(torch.meshgrid(torch.linspace(
                            0, H_sc-1, H_sc), torch.linspace(0, W_sc-1, W_sc)), -1)  # (H_sc, W_sc, 2)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                select_inds = np.random.choice(
                    coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0],
                                select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0],
                                select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0],
                                  select_coords[:, 1]]  # (N_rand, 3)

                if args.snerf and sat_arg_dict['train.shad']:
                    coords_sc = torch.reshape(
                        coords_sc, [-1, 2])  # (H_sx * W_sx, 2)
                    select_inds_sc = np.random.choice(
                        coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                    # (N_rand, 2)
                    select_coords_sc = coords[select_inds_sc].long()
                    sc_rays_o = sc_rays_o[select_coords_sc[:, 0],
                                          select_coords_sc[:, 1]]  # (N_rand, 3)
                    sc_rays_d = sc_rays_d[select_coords_sc[:, 0],
                                          select_coords_sc[:, 1]]  # (N_rand, 3)
                    batch_sc = torch.stack([sc_rays_o, sc_rays_d], 0)

        #####  Core optimization loop  #####
        if args.snerf:
            rgb, sun, shadow, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays, verbose=i < 10,
                                                         retraw=True, light_dir=light_dir[img_i], **render_kwargs_train)
            if sat_arg_dict['train.shad']:
                _, sun, shadow, _, _, _ = render(H, W, K, chunk=args.chunk, rays=batch_sc, verbose=i < 10,
                                                 retraw=True, light_dir=light_dir[img_i], **render_kwargs_train)
        else:
            rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays, verbose=i < 10, retraw=True,
                                            **render_kwargs_train)

        optimizer.zero_grad()
        rgb_loss = img2mse(rgb, target_s)
        loss = rgb_loss

        if args.snerf:
            # Compute shadow loss
            lambda_sc = 0.05  # recommended by the SNerf paper
            shadow_loss = (torch.mean(shadow) +
                           torch.mean(1.0-sun)) * lambda_sc
            loss += shadow_loss

        # saving logs
        if i % args.i_print == 0 or i % args.i_video == 0:
            psnr = mse2psnr(rgb_loss)
            ssim = structural_similarity(rgb.cpu().detach().numpy(
            ), target_s.cpu().detach().numpy(), multichannel=True)
            if args.savelogs:
                with open(logfile, 'a') as file:
                    file.write('{}, {}, {}\n'.format(
                        i, psnr.cpu().detach().numpy()[0], ssim))

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                if args.snerf:
                    rgbs, disps, depths = render_path(
                        render_poses, hwf, K, args.chunk, render_kwargs_test, pose_indices=i_test, light_dir=light_dir)
                else:
                    rgbs, disps, depths = render_path(
                        render_poses, hwf, K, args.chunk, render_kwargs_test, pose_indices=i_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(
                basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))

            numOfImages = rgbs.shape[0]
            for j in range(numOfImages):
                target_j = images[i_test[j]]
                psnr_test = mse2psnr(img2mse(torch.Tensor(rgbs[j]), target_j))
                ssim_test = structural_similarity(
                    rgbs[j], target_j.cpu().numpy(), multichannel=True)
                print(
                    f"[TEST {j}] Iter: {i}  PSNR: {psnr_test.item()}  SSIM: {ssim_test.item()}")
                if args.savelogs:
                    with open(logfile, 'a') as file:
                        file.write('[TEST {}], {}, {}\n'.format(
                            j, psnr_test.cpu().detach().numpy()[0], ssim_test))
                img = Image.fromarray(to8b(rgbs[j, :, :, :]), 'RGB')
                img.save(moviebase + '_view_' + str(j) + '_rgb.png')
                io.imsave(moviebase + '_view_' + str(j) + '_disp.tiff', disps[j, :, :])
                io.imsave(moviebase + '_view_' + str(j) + '_depth.tiff', depths[j, :, :])

            # Calculates altitude metrics.
            file_prefix = os.path.join(basedir, expname, '{:06d}_'.format(i))
            alt_rmse, alt_mae, alt_comp = calculate_altitude_rmse(
                depth_map, file_prefix, sat_arg_dict, render_kwargs_test)
            print(f"[TRAIN] Iter: {i}  Loss: {loss.item()}  PSNR: {psnr.item()}  SSIM: {ssim.item()}  Altitude RMSE: {alt_rmse.item()}, MAE: {alt_rmse.item()}, Completion Rate: {alt_comp.item()}")
            if args.savelogs:
                with open(logfile, 'a') as file:
                    file.write('{}, {}, {}, {}, {}, {}\n'.format(
                        i, psnr.cpu().detach().numpy()[0], ssim, alt_rmse, alt_rmse, alt_comp))

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(
                basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                if args.snerf:
                    render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test,
                                gt_imgs=images[i_test], savedir=testsavedir, pose_indices=i_test, light_dir=light_dir)
                else:
                    render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test,
                                gt_imgs=images[i_test], savedir=testsavedir, pose_indices=i_test)
            print('Saved test set')

        if i % args.i_print == 0:
            tqdm.write(
                f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}  SSIM: {ssim.item()}")
        global_step += 1


if __name__ == '__main__':
    if device == torch.device("cuda"):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
