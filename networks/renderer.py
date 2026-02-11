import re
import time

import torch
from torch import nn

from utils.rays import get_rays, get_ndc_rays, sample_pdf
from networks.nerf import NeRF
from networks.tonemapping import CRF
from networks.embedding import get_embedder
from networks.pdrf.voxnerf import VoxelNeRFRayFeatures, VoxelNeRFSampleFeatures


class NeRFAll(nn.Module):
    def __init__(self, args):
        super(NeRFAll, self).__init__()
        self.args = args
        self.embed_fn, self.input_ch = get_embedder(args.multires)

        self.mode = args.mode

        self.composite_feature_coarse = False
        self.composite_feature_fine = False
        self.extract_feature = "before_linear"

        self.input_ch_views = 0
        self.embeddirs_fn = None
        if args.use_viewdirs:
            self.embeddirs_fn, self.input_ch_views = get_embedder(args.multires_views)

        self.output_ch = 5 if args.N_importance > 0 else 4

        skips = [4]
        if self.mode == 'c2f':
            self.mlp_coarse = VoxelNeRFRayFeatures(
                aabb=args.bounding_box, num_layers=args.coarse_num_layers, hidden_dim=args.coarse_hidden_dim,
                geo_feat_dim=args.coarse_geo_feat_dim, num_layers_color=args.coarse_num_layers_color,
                hidden_dim_color=args.coarse_hidden_dim_color, add_bias_color=args.rgb_add_bias,
                input_ch=args.coarse_app_dim + self.input_ch,
                input_ch_views=self.input_ch_views, render_rmnearplane=args.render_rmnearplane,
                app_dim=args.coarse_app_dim, app_n_comp=args.coarse_app_n_comp, n_voxels=args.coarse_n_voxels,
                composite_feature=self.composite_feature_coarse,
                extract_feature=self.extract_feature, app_actfn=args.coarse_app_actfn)

            self.grad_vars_vol, self.grad_vars = self.mlp_coarse.get_optparam_groups()

            if args.N_importance > 0:
                self.mlp_fine = VoxelNeRFSampleFeatures(
                    aabb=args.bounding_box, num_layers=args.fine_num_layers, hidden_dim=args.fine_hidden_dim,
                    geo_feat_dim=args.fine_geo_feat_dim, num_layers_color=args.fine_num_layers_color,
                    hidden_dim_color=args.fine_hidden_dim_color, add_bias_color=args.rgb_add_bias,
                    input_ch=args.coarse_app_dim + args.fine_app_dim + self.input_ch,
                    input_ch_views=self.input_ch_views, render_rmnearplane=args.render_rmnearplane,
                    app_dim=args.fine_app_dim, app_n_comp=args.fine_app_n_comp, n_voxels=args.fine_n_voxels,
                    composite_feature=self.composite_feature_fine,
                    extract_feature=self.extract_feature, app_actfn=args.fine_app_actfn)

                grad_vars_vol, grad_vars_net = self.mlp_fine.get_optparam_groups()
                self.grad_vars += grad_vars_net
                self.grad_vars_vol += grad_vars_vol
            else:
                self.mlp_fine = None
        elif self.mode == 'nerf':
            self.mlp_coarse = NeRF(
                D=args.netdepth, W=args.netwidth,
                input_ch=self.input_ch, output_ch=self.output_ch, skips=skips,
                input_ch_views=self.input_ch_views, use_viewdirs=args.use_viewdirs,
                rgb_activate=args.rgb_activate, rgb_add_bias=args.rgb_add_bias, sigma_activate=args.sigma_activate,
                render_rmnearplane=args.render_rmnearplane,
                extract_feature=self.extract_feature, composite_feature=self.composite_feature_coarse)

            if args.N_importance > 0:
                self.mlp_fine = NeRF(
                    D=args.netdepth_fine, W=args.netwidth_fine,
                    input_ch=self.input_ch, output_ch=self.output_ch, skips=skips,
                    input_ch_views=self.input_ch_views, use_viewdirs=args.use_viewdirs,
                    rgb_activate=args.rgb_activate, rgb_add_bias=args.rgb_add_bias, sigma_activate=args.sigma_activate,
                    render_rmnearplane=args.render_rmnearplane,
                    extract_feature=self.extract_feature, composite_feature=self.composite_feature_fine)
            else:
                self.mlp_fine = None
        else:
            raise NotImplementedError(f"{self.mode} for rendering network is not implemented")

        activate = {'relu': torch.relu, 'sigmoid': torch.sigmoid, 'exp': torch.exp, 'none': lambda x: x,
                    'sigmoid1': lambda x: 1.002 / (torch.exp(-x) + 1) - 0.001,
                    'softplus': lambda x: nn.Softplus()(x - 1)}
        self.rgb_activate = activate[args.rgb_activate]
        self.sigma_activate = activate[args.sigma_activate]

        print(self.mlp_coarse, self.mlp_fine)

    def get_parameters(self, type, match_re=None, not_match_re=None):
        def match(text, regex):
            return len(re.findall(regex, text)) > 0

        def is_vol_param(k):
            return "app_plane" in k or "app_line" in k

        params = {}
        for k, v in self.named_parameters():
            if (match_re is None or match(k, match_re)) and \
                    (not_match_re is None or not match(k, not_match_re)):
                if type == "net" and not is_vol_param(k):
                    params[k] = v
                elif type == "vol" and is_vol_param(k):
                    params[k] = v
        return list(params.values())

    def render_rays(self,
                    ray_batch,
                    N_samples,
                    retraw=False,
                    lindisp=False,
                    perturb=0.,
                    N_importance=0,
                    white_bkgd=False,
                    raw_noise_std=0.,
                    pytest=False,
                    force_naive=False,
                    inference=False):
        """Volumetric rendering.
        Args:
          ray_batch: array of shape [batch_size, ...]. All information necessary
            for sampling along a ray, including: ray origin, ray direction, min
            dist, max dist, and unit-magnitude viewing direction.
          N_samples: int. Number of different times to sample along each ray.
          retraw: bool. If True, include model's raw, unprocessed predictions.
          lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
          perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
            random points in time.
          N_importance: int. Number of additional times to sample along each ray.
            These samples are only passed to network_fine.
          white_bkgd: bool. If True, assume a white background.
          raw_noise_std: ...
          verbose: bool. If True, print more debugging info.
        """
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

        t_vals = torch.linspace(0., 1., steps=N_samples).type_as(rays_o)
        if not lindisp:
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
            t_rand = torch.rand(z_vals.shape).type_as(rays_o)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

        if self.mode == 'c2f':
            ft_coarse = self.mlp_coarse.sample(pts)

            rgb_map, depth_map, acc_map, weights, feature = self.mlp_coarse(
                pts, viewdirs, ft_coarse, self.embed_fn, self.embeddirs_fn,
                z_vals, rays_d, raw_noise_std, self.training)

            if N_importance > 0:
                if feature is not None:
                    del feature  # Will be overwritten anyway

                # self.mlp_fine is only defined if N_importance > 0
                ft_fine = self.mlp_fine.sample(pts)
                ft_comb0 = torch.cat([ft_coarse, ft_fine], -1)

                rgb_map_0, depth_map_0, acc_map_0 = rgb_map, depth_map, acc_map
                z_vals_0, weights_0 = z_vals, weights

                z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
                z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.),
                                       pytest=pytest)
                z_samples = z_samples.detach()

                z_vals, order = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
                pts1 = rays_o[..., None, :] + rays_d[..., None, :] \
                       * z_samples[..., :, None]  # [N_rays, N_samples + N_importance, 3]

                ft_coarse = self.mlp_coarse.sample(pts1)
                pts = torch.cat([pts, pts1], 1)[torch.arange(pts1.shape[0]).unsqueeze(1), order]
                ft_fine = self.mlp_fine.sample(pts1)
                ft_comb1 = torch.cat([ft_coarse, ft_fine], -1)
                ft_comb = torch.cat([ft_comb0, ft_comb1], 1)[torch.arange(pts1.shape[0]).unsqueeze(1), order]

                rgb_map, depth_map, acc_map, weights, feature = self.mlp_fine(
                    pts, viewdirs, ft_comb, self.embed_fn, self.embeddirs_fn,
                    z_vals, rays_d, raw_noise_std, self.training)
        else:
            rgb_map, depth_map, acc_map, weights, feature = self.mlp_coarse(
                pts, viewdirs, self.embed_fn, self.embeddirs_fn,
                z_vals, rays_d, raw_noise_std, white_bkgd, self.training)

            if N_importance > 0:
                if feature is not None:
                    del feature  # Will be overwritten anyway

                rgb_map_0, depth_map_0, acc_map_0 = rgb_map, depth_map, acc_map
                z_vals_0, weights_0 = z_vals, weights

                z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
                z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
                z_samples = z_samples.detach()

                z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
                pts = rays_o[..., None, :] + rays_d[..., None, :] \
                      * z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

                rgb_map, depth_map, acc_map, weights, feature = self.mlp_fine(
                    pts, viewdirs, self.embed_fn, self.embeddirs_fn,
                    z_vals, rays_d, raw_noise_std, white_bkgd, self.training)

        ret = {'rgb_map': rgb_map, 'depth_map': depth_map, 'acc_map': acc_map}
        if retraw:
            ret['z_vals'] = z_vals
            ret['weights'] = weights
        if N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['depth0'] = depth_map_0
            ret['acc0'] = acc_map_0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
            if retraw:
                ret['z_vals0'] = z_vals_0
                ret['weights0'] = weights_0

        for k in ret:
            if torch.isnan(ret[k]).any():
                print(f"! [Numerical Error] {k} contains nan.")
            if torch.isinf(ret[k]).any():
                print(f"! [Numerical Error] {k} contains inf.")
        return ret

    def forward(self, H, W, K, chunk=1024 * 32, rays=None, rays_info=None, poses=None, **kwargs):
        """
        render rays or render poses, rays and poses should atleast specify one
        calling model.train() to render rays, where rays, rays_info, should be specified
        calling model.eval() to render an image, where poses should be specified

        optional args:
        force_naive: when True, will only run the naive NeRF, even if the kernelsnet is specified
        """

        # training
        if self.training:
            assert rays is not None, "Please specify rays when in the training mode"

            force_baseline = kwargs.pop("force_naive", True)
            return_pts0_rgb = kwargs.pop("return_pts0_rgb", False)
            N_importance = kwargs.get("N_importance", 0)

            other_loss, other_tensors = {}, {}

            # kernel mode, run multiple rays to get result of one ray
            if not force_baseline:
                assert "rays_weights" in rays_info, "Please specify rays_weights when in the kernel mode"

                weights = rays_info["rays_weights"]
                weights = weights / torch.sum(weights, dim=-1, keepdim=True)

                ray_num, pt_num, _, _ = rays.shape
                assert ray_num == weights.shape[0] and pt_num == weights.shape[1], \
                    (f"rays and rays_weights should have the same first 2 dimensions, "
                     f"but got {rays.shape} and {weights.shape}")

                rgb, depth, acc, extras = self.render(H, W, K, chunk, rays.reshape(-1, 3, 2), **kwargs)

                rgb_pts = rgb.reshape(ray_num, pt_num, 3)
                rgb = torch.sum(rgb_pts * weights[..., None], dim=1)

                if N_importance > 0:  # rgb0 only defined when N_importance > 0
                    rgb1_pts = extras['rgb0'].reshape(ray_num, pt_num, 3)
                    rgb1 = torch.sum(rgb1_pts * weights[..., None], dim=1)
                else:
                    rgb1 = None

                if self.mode == 'c2f':
                    other_loss["TV"] = self.mlp_coarse.TV_loss_app()
                    if N_importance > 0:  # mlp_fine is only defined when N_importance > 0
                        other_loss["TV"] += self.mlp_fine.TV_loss_app()
                    other_loss["TV"] *= 5

                if return_pts0_rgb:
                    other_tensors["stage1_rgb_pts0"] = rgb_pts[:, 0]
                    if N_importance > 0:
                        other_tensors["stage1_rgb1_pts0"] = rgb1_pts[:, 0]

                return rgb, rgb1, other_loss, other_tensors
            else:
                if rays.ndim == 4:
                    rays = rays[:, 0]  # just take the center exposure ray
                else:
                    assert rays.ndim == 3, f"rays should have 3 or 4 dimensions, but got {rays.ndim}"

                rgb, depth, acc, extras = self.render(H, W, K, chunk, rays, **kwargs)

                if return_pts0_rgb:
                    other_tensors["stage1_rgb_pts0"] = rgb
                    if N_importance > 0:
                        other_tensors["stage1_rgb1_pts0"] = extras["rgb0"]

                if self.mode == 'c2f':
                    other_loss["TV"] = self.mlp_coarse.TV_loss_app()
                    if N_importance > 0:  # mlp_fine is only defined when N_importance > 0
                        other_loss["TV"] += self.mlp_fine.TV_loss_app()
                    other_loss["TV"] *= 5

                return rgb, extras['rgb0'] if 'rgb0' in extras else None, other_loss, other_tensors

        #  evaluation
        else:
            assert poses is not None, "Please specify poses when in the eval model"
            rgbs, depths = self.render_path(H, W, K, chunk, poses, **kwargs)
            return rgbs, depths

    def render(self, H, W, K, chunk, rays=None, c2w=None, ndc=True,
               near=0., far=1., use_viewdirs=False, c2w_staticcam=None, **kwargs):
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
            Returns:
              rgb_map: [batch_size, 3]. Predicted RGB values for rays.
              disp_map: [batch_size]. Disparity map. Inverse of depth.
              acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
              extras: dict with everything returned by render_rays().
            """
        rays_o, rays_d = rays[..., 0], rays[..., 1]

        if use_viewdirs:
            # provide ray directions as input
            viewdirs = rays_d
            if c2w_staticcam is not None:
                # special case to visualize effect of viewdirs
                rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

        sh = rays_d.shape  # [..., 3]
        if ndc:
            # for forward facing scenes
            rays_o, rays_d = get_ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()

        near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
        rays = torch.cat([rays_o, rays_d, near, far], -1)
        if use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)

        # Batchfy and Render and reshape
        all_ret = {}
        for i in range(0, max(1, rays.shape[0]), chunk):  # max(1,.) to handle empty rays
            ret = self.render_rays(rays[i:i + chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        k_extract = ['rgb_map', 'depth_map', 'acc_map']
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}

        return ret_list + [ret_dict]

    def coarse_render(self, H, W, K, chunk, rays=None, c2w=None, ndc=True,
                      near=0., far=1.,
                      use_viewdirs=False, c2w_staticcam=None,
                      **kwargs):  # the render function
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
            Returns:
              rgb_map: [batch_size, 3]. Predicted RGB values for rays.
              disp_map: [batch_size]. Disparity map. Inverse of depth.
              acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
              extras: dict with everything returned by render_rays().
            """
        rays_o, rays_d = rays[..., 0], rays[..., 1]

        if use_viewdirs:
            # provide ray directions as input
            viewdirs = rays_d
            if c2w_staticcam is not None:
                # special case to visualize effect of viewdirs
                rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

        sh = rays_d.shape  # [..., 3]
        if ndc:
            # for forward facing scenes
            rays_o, rays_d = get_ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()

        near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
        rays = torch.cat([rays_o, rays_d, near, far], -1)
        if use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)

        all_ret = {}
        # TODO You could merge the render rays coarse and fine by making the
        #   render function take in the render rays function or mode as an argument
        rgb, feat = self.coarse_render_rays(rays, **kwargs)
        return rgb, feat

    def coarse_render_rays(self,
                           ray_batch,
                           N_samples,
                           retraw=False,
                           lindisp=False,
                           perturb=0.,
                           N_importance=0,
                           white_bkgd=False,
                           raw_noise_std=0.,
                           force_naive=False,
                           inference=False):
        """Volumetric rendering.
        Args:
          ray_batch: array of shape [batch_size, ...]. All information necessary
            for sampling along a ray, including: ray origin, ray direction, min
            dist, max dist, and unit-magnitude viewing direction.
          N_samples: int. Number of different times to sample along each ray.
          retraw: bool. If True, include model's raw, unprocessed predictions.
          lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
          perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
            random points in time.
          N_importance: int. Number of additional times to sample along each ray.
            These samples are only passed to network_fine.
          white_bkgd: bool. If True, assume a white background.
          raw_noise_std: ...
          verbose: bool. If True, print more debugging info.
        """
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

        t_vals = torch.linspace(0., 1., steps=N_samples).type_as(rays_o)
        if not lindisp:
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
            t_rand = torch.rand(z_vals.shape).type_as(rays_o)

            z_vals = lower + (upper - lower) * t_rand

        pts0 = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

        if self.mode == 'c2f':
            ft_coarse = self.mlp_coarse.sample(pts0)
            rgb_map, _, _, _, feat = self.mlp_coarse(pts0, viewdirs, ft_coarse, self.embed_fn, self.embeddirs_fn,
                                                     z_vals, rays_d, raw_noise_std, self.training)
        else:
            rgb_map, _, _, _, feat = self.mlp_coarse(pts0, viewdirs, self.embed_fn, self.embeddirs_fn, z_vals, rays_d,
                                                     raw_noise_std, white_bkgd, self.training)
        return rgb_map, feat

    def render_path(self, H, W, K, chunk, render_poses, render_kwargs, render_factor=0, ):
        """
        render image specified by the render_poses
        """
        if render_factor != 0:
            # Render downsampled for speed
            H = H // render_factor
            W = W // render_factor

        rgbs = []
        depths = []

        for i, c2w in enumerate(render_poses):
            device = c2w.device
            c2w = c2w.cuda()  # This could be on cpu if the caller decided to run in "memory efficient" mode
            
            rays = get_rays(H, W, K, c2w)
            rays = torch.stack(rays, dim=-1)
            rgb, depth, acc, extras = self.render(H, W, K, chunk=chunk, rays=rays, c2w=c2w[:3, :4], **render_kwargs)

            rgbs.append(rgb.to(device))
            depths.append(depth.to(device))

        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)

        return rgbs, depths
