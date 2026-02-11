import os
from options import config_parser

import cv2
import time
import imageio
import numpy as np
from glob import glob

import torch
from torch import nn
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from data.sampler_image_batch import ImageBatchSampler

from networks.renderer import NeRFAll
from networks.tonemapping import TonemappingTransform
from networks.embedding import ViewEmbedding, ViewEmbeddingMLP, get_embedder
from poses.timestamp_sampler import TimestampSampler, LearnedTimestampSampler, HybridTimestampSampler
from poses import RefineInterpolator, SlerpInterpolator, PoseNet
from poses.testtime_refiner import TestTimePoseRefiner

from utils.logger import Logger
from utils.grads import grads_norm
from utils.metrics import compute_img_metric, img2mse, mse2psnr
from utils.poses import evo_align, evo_solution, trajectory_errors
from utils.events import egm_loss

from data.loader import LLFFDataset, endless
from data.loader_events import LLFFEventsDataset
from utils.data import render_path_interpolated
from utils.misc import seed_everything, to8b, smart_load_state_dict, \
    annealing_interpolator, get_optimizer, get_seed_worker_fn
from utils.rays import get_rays_pix


def train():
    parser = config_parser()
    args = parser.parse_args()

    if args.events_threshold_pos is None or args.events_threshold_neg is None:
        print(f"WARNING: overriding events_threshold_pos and events_threshold_neg "
              f"to events_threshold={args.events_threshold}")
        args.events_threshold_pos = args.events_threshold
        args.events_threshold_neg = args.events_threshold

    if len(args.torch_hub_dir) > 0:
        print(f"Change torch hub cache to {args.torch_hub_dir}")
        torch.hub.set_dir(args.torch_hub_dir)

    # Load data
    print(args)
    print('RANDOM SEED', args.seed)
    seed_everything(args.seed, warn_only=True)

    llff_dataset = LLFFDataset(args, args.datadir, args.factor,
                               recenter=True, bd_factor=args.bd_factor,
                               spherify=args.spherify,
                               path_epi=args.render_epi,
                               pose_transform_all_known_poses=args.transform_all_known_poses,
                               device="cpu")

    if args.ray_sampling_mode == "random":
        sampler = BatchSampler(RandomSampler(llff_dataset, generator=torch.Generator(device='cuda')),
                               batch_size=args.N_rand, drop_last=True)
    elif args.ray_sampling_mode in ["same", "images"]:
        sampler = ImageBatchSampler(llff_dataset, same_imgs_size=args.ray_sampling_images_num,
                                    batch_size=args.N_rand, num_imgs=llff_dataset.n_imgs,
                                    image_resolution=(llff_dataset.w, llff_dataset.h),
                                    generator=torch.Generator(device='cpu'))
    else:
        raise ValueError(f"Unknown ray_sampling_mode: {args.ray_sampling_mode}")

    # These will be used to interpolate poses from
    gt_seenim_poses = llff_dataset.test_poses_seen  # gt poses of train images (seen)
    gt_seenim_tms = llff_dataset.images_mid_tms
    gt_allknown_poses = gt_seenim_poses.new_empty([0, 3, 4])  # extra in-between images gt poses (allknown)
    gt_allknown_tms = gt_seenim_poses.new_empty([0])

    ev_allknown_poses = llff_dataset.poses.new_empty([0, 3, 4])
    ev_allknown_poses_tms = llff_dataset.images_mid_tms.new_empty([0])

    if args.use_events:
        llffev_dataset = LLFFEventsDataset(args, args.datadir, llff_dataset.h, llff_dataset.w, llff_dataset.K,
                                           args.factor, recenter=True,
                                           bd_factor=args.bd_factor,
                                           bd_scale=llff_dataset.scale,
                                           closest_bds=llff_dataset.closest_bds,
                                           furthest_bds=llff_dataset.furthest_bds,
                                           spherify=args.spherify,
                                           recenter_partial=llff_dataset.recenter_partial,
                                           spherify_partial=llff_dataset.spherify_partial,
                                           events_tms_unit=args.events_tms_unit,
                                           events_tms_files_unit=args.events_tms_files_unit,
                                           color_events=args.event_egm_use_colorevents,
                                           device="cpu")
        g = torch.Generator(device='cuda')
        g.manual_seed(args.seed)
        train_ev_loader = DataLoader(
            llffev_dataset,
            # Use a batch sampler as the sampler so that __getitem__ is called with a list of indices
            sampler=BatchSampler(RandomSampler(llffev_dataset, generator=g),
                                 batch_size=args.events_N_rand, drop_last=True),
            # Use batch size None to disable auto-batching, but still use multiple workers to prefetch
            batch_size=None, num_workers=8, pin_memory=True, prefetch_factor=16,
            worker_init_fn=get_seed_worker_fn(args.seed))

        events_threshold_negpos = torch.tensor([[args.events_threshold_neg, args.events_threshold_pos]],
                                               dtype=torch.float32, device="cuda")

        if args.use_prior_images == "edi":
            llff_dataset.set_prior(llffev_dataset.compute_edi_prior(
                llff_dataset.i_train, llff_dataset.images, args.prior_edi_steps,
                args.events_threshold_pos, args.events_threshold_neg))


        gt_allknown_poses = llffev_dataset.allknown_test_poses  # extra in-between images gt poses (allknown)
        gt_allknown_tms = llffev_dataset.allknown_test_timestamps

        ev_allknown_poses_tms = llffev_dataset.allknown_poses_timestamps  # extra in-between images event poses (allknown)
        ev_allknown_poses = llffev_dataset.allknown_poses
    else:
        llffev_dataset, train_ev_loader = None, None
        events_threshold_negpos = None

    g = torch.Generator(device='cuda')
    g.manual_seed(args.seed)
    train_loader = DataLoader(
        llff_dataset, sampler=sampler,
        # Use batch size None to disable auto-batching, but still use multiple workers to prefetch
        batch_size=None, num_workers=8, pin_memory=True, prefetch_factor=8,
        worker_init_fn=get_seed_worker_fn(args.seed), generator=g)

    train_iterator = iter(endless(train_loader))
    train_ev_iterator = iter(endless(train_ev_loader))

    args.bounding_box = llff_dataset.bounding_box
    near, far = llff_dataset.near, llff_dataset.far
    H, W = int(llff_dataset.h), int(llff_dataset.w)
    K = llff_dataset.K

    w_events_egm = lambda x: None
    if args.use_events:
        w_events_egm = annealing_interpolator(args.event_egm_weight,
                                              args.event_egm_weight_end,
                                              args.event_egm_weight_steps,
                                              args.event_egm_weight_scheduler)

    w_prior = lambda x: None
    if args.use_prior_images:
        w_prior = annealing_interpolator(args.prior_weight,
                                         args.prior_weight_end,
                                         args.prior_weight_steps,
                                         args.prior_weight_scheduler)

    w_kernel = lambda x: 1.0
    kernel_end_warmup_iter = -1
    if args.kernel_start_warmup_mode != "step":
        kernel_end_warmup_iter = args.kernel_start_iter + args.kernel_start_warmup_iters
        w_kernel = annealing_interpolator(0.0, 1.0,
                                          kernel_end_warmup_iter,
                                          args.kernel_start_warmup_mode,
                                          start_step=args.kernel_start_iter)

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    wandb_id = None
    test_metric_file = os.path.join(basedir, expname, 'test_metrics.txt')
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None and not args.render_only:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

        with open(test_metric_file, 'a') as file:
            file.write(open(args.config, 'r').read())
            file.write("\n============================\n"
                       "||\n"
                       "\\/\n")

    # Compute the allknown poses and timestamps for interpolation. With allknown here we mean the full trajectory (not
    # just the poses at seen image timestamps), which could be available thanks to events, imu, mocap etc.
    if args.use_events:
        # Merge poses in-order and without repetitions
        new_tms = ~torch.isin(llff_dataset.images_mid_tms, ev_allknown_poses_tms)  # mask of new timestamps
        allknown_poses = torch.cat([ev_allknown_poses, llff_dataset.poses[new_tms]], dim=0)  # ad w/o repetition
        allknown_tms = torch.cat([ev_allknown_poses_tms, llff_dataset.images_mid_tms[new_tms]], dim=0)

        allknown_sort = torch.argsort(allknown_tms)  # sort by timestamps
        allknown_poses = allknown_poses[allknown_sort]
        allknown_tms = allknown_tms[allknown_sort]
    else:
        # Remove event poses
        allknown_tms = llff_dataset.images_mid_tms
        allknown_poses = llff_dataset.poses

    if args.pose_interpolator_type == "slerp":
        pose_interpolator = SlerpInterpolator(
            ts_us=allknown_tms, poses_rot=allknown_poses[:, :3, :3], poses_tran=allknown_poses[:, :3, 3],
            clip_boundary=True, warn_boundary=False)
        refiner = None
    else:
        time_embedder, time_emb_cnl = get_embedder(args.pose_interpolator_time_emb_freq,
                                                   i=1, input_dim=1, include_input=True, scale_pi=True)

        if args.pose_interpolator_type == "posenet":
            refiner = PoseNet(
                layers_feat=args.pose_interpolator_posenet_layers, skip=args.pose_interpolator_posenet_skip,
                activ=args.pose_interpolator_posenet_activ, embed_size=time_emb_cnl
            )
        else:
            raise ValueError(f"Unknown pose_interpolator_type: {args.pose_interpolator_type}")

        pose_interpolator = RefineInterpolator(
            refiner=refiner, embedder=time_embedder,
            min_time=allknown_tms.min(), max_time=allknown_tms.max(),
            embed_schedule=args.pose_interpolator_time_emb_schedule,
            ts_us=allknown_tms, poses_rot=allknown_poses[:, :3, :3],
            poses_tran=allknown_poses[:, :3, 3],
            init_identity=args.pose_interpolator_init_identity,
            clip_boundary=True, warn_boundary=False)

        if args.pose_interpolator_init_ckpt is not None:
            ckpt = torch.load(args.pose_interpolator_init_ckpt)
            smart_load_state_dict(pose_interpolator, ckpt, network_key="pose_interpolator_state_dict")

    if args.view_embedder_type == 'param':
        view_embed = ViewEmbedding(num_embed=llff_dataset.n_imgs, embed_dim=args.view_embedder_embed,
                                   init_params=args.view_embedder_embed_init)
    elif args.view_embedder_type == 'param_mlp':
        view_embed = ViewEmbeddingMLP(num_embed=llff_dataset.n_imgs, embed_dim=args.view_embedder_embed,
                                      init_params=args.view_embedder_embed_init,
                                      D=args.view_embedder_mlp_depth, W=args.view_embedder_mlp_embed,
                                      skips=[args.view_embedder_mlp_skips])
    else:
        raise ValueError(f"Unknown view_embedder_type: {args.view_embedder_type}")

    timestamps_sampler_exposure_time_us = args.timestamps_sampler_exposure_time_us
    if timestamps_sampler_exposure_time_us is None:
        exposure_times_us = llff_dataset.images_end_tms - llff_dataset.images_start_tms
        timestamps_sampler_exposure_time_us = (exposure_times_us).to(torch.float64).max().item()
        print("Using (max) exposure time from dataset:", timestamps_sampler_exposure_time_us)
        print("Unique exposure times:", torch.unique(exposure_times_us))

    timestamps_sampler = None
    if args.timestamps_sampler_type == "uniform":
        timestamps_sampler = TimestampSampler(num_samples=args.timestamps_sampler_num_samples,
                                              exposure_time_us=timestamps_sampler_exposure_time_us)
    elif args.timestamps_sampler_type == "learned":
        timestamps_sampler = LearnedTimestampSampler(num_samples=args.timestamps_sampler_num_samples,
                                                     exposure_time_us=timestamps_sampler_exposure_time_us,
                                                     num_images=llff_dataset.n_imgs,
                                                     view_embed=view_embed,
                                                     num_layers=args.timestamps_sampler_learned_num_layers,
                                                     hidden_dim=args.timestamps_sampler_learned_hidden_dim,
                                                     predict_weigths=args.timestamps_sampler_learned_predict_weights,
                                                     use_anchors=args.timestamps_sampler_learned_use_anchors)
    elif args.timestamps_sampler_type == "hybrid":
        timestamps_sampler =  HybridTimestampSampler(num_samples=args.timestamps_sampler_num_samples,
                                                     exposure_time_us=timestamps_sampler_exposure_time_us,
                                                     num_images=llff_dataset.n_imgs,
                                                     view_embed=view_embed,
                                                     num_layers=args.timestamps_sampler_learned_num_layers,
                                                     hidden_dim=args.timestamps_sampler_learned_hidden_dim,
                                                     predict_weigths=args.timestamps_sampler_learned_predict_weights,
                                                     use_anchors=args.timestamps_sampler_learned_use_anchors)
    elif args.timestamps_sampler_type is not None:
        raise ValueError(f"Unknown timestamps_sampler_type: {args.timestamps_sampler_type}")

    # Create camera(s) response function
    extra_features_event = 0 if args.tone_mapping_events_add_bii == "none" else 2
    if args.tone_mapping_events_type == 'rgb_learn' and args.tone_mapping_events_add_bii == 'color-pos-neg':
        extra_features_event *= 3  # either 6 or 0
    crf = TonemappingTransform(map_type_rgb=args.tone_mapping_type,
                               map_type_event=args.tone_mapping_events_type,
                               extra_features_event=extra_features_event,
                               gamma=args.tone_mapping_gamma,
                               init_learn_identity=args.tone_mapping_learn_init_identity)

    # Create nerf model
    nerf = NeRFAll(args)
    if args.mode == 'c2f':
        if args.colornet_weightdecay:
            optim_params = [
                {'name': 'nerf_color', 'params': nerf.get_parameters("net", match_re=r"\.color_net\.[0-9]+\.weight"),
                 'lr': args.lrate, 'weight_decay': args.colornet_weightdecay},
                {'name': 'nerf', 'params': nerf.get_parameters("net", not_match_re=r"\.color_net\.[0-9]+\.weight"),
                 'lr': args.lrate},
                {'name': 'nerf_volume', 'params': nerf.grad_vars_vol, 'lr': args.lrate}]
        else:
            optim_params = [
                {'name': 'nerf', 'params': nerf.grad_vars, 'lr': args.lrate},
                {'name': 'nerf_volume','params': nerf.grad_vars_vol, 'lr': args.lrate}]
    elif args.mode == 'nerf':
        optim_params = [
            {'name': 'nerf', 'params': nerf.parameters(), 'lr': args.lrate}]
    else:
        raise NotImplementedError(f"{args.mode} for rendering network is not implemented")

    if timestamps_sampler is not None:
        optim_params += [{'name': 'timestamps_sampler', 'params': timestamps_sampler.parameters(), 'lr': args.lrate}]

    optim_params += [{'name': 'crf', 'params': crf.parameters(), 'lr': args.lrate}]
    interp_rot_lrate = args.pose_interpolator_rot_lrate or args.pose_interpolator_lrate
    interp_tran_lrate = args.pose_interpolator_tran_lrate or args.pose_interpolator_lrate

    interp_params = pose_interpolator.get_param_groups()
    assert len(interp_params["rest"]) == 0
    optim_params += [
        {'name': 'interpolator_rot', 'params': interp_params["rotation"], 'lr': interp_rot_lrate}]
    optim_params += [
        {'name': 'interpolator_tra', 'params': interp_params["translation"], 'lr': interp_tran_lrate}]

    # Stores the initial lr to remember it for later
    for group in optim_params:
        group.setdefault('initial_lr', group['lr'])

    # Scales the lr by the warmup factor
    for group in optim_params:
        if args.pose_interpolator_warmup_iters > 0 and group["name"] in ["interpolator_rot", "interpolator_tra"]:
            group['lr'] = group['lr'] * args.pose_interpolator_warmup_factor
        elif args.lrate_warmup_iters > 0:
            group['lr'] = group['lr'] * args.lrate_warmup_factor

    optimizer = torch.optim.Adam(params=optim_params,
                                 lr=args.lrate,
                                 betas=(0.9, 0.999))

    start = 0

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 '.tar' in f and 'testtime' not in f]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        if llffev_dataset is not None:
            llffev_dataset.global_step = start
        wandb_id = ckpt['wandb_id'] if 'wandb_id' in ckpt else None

        # Load model
        smart_load_state_dict(nerf, ckpt, network_key="network_state_dict")
        smart_load_state_dict(crf, ckpt, network_key="crf_state_dict")
        smart_load_state_dict(pose_interpolator, ckpt, network_key="pose_interpolator_state_dict")
        if timestamps_sampler is not None:
            smart_load_state_dict(timestamps_sampler, ckpt, network_key="timestamps_sampler_state_dict")
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    logger = Logger(log_dir=args.tbdir, expname=args.expname,
                    use_wandb=not args.no_wandb and not args.render_only,
                    use_tensorboard=args.use_tensorboard,
                    wandb_id=wandb_id,
                    args=args)

    # figuring out the train/test configuration
    render_kwargs_train = {
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'N_samples': args.N_samples,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
        'inference': False,
    }
    # NDC only good for LLFF-style forward facing data
    if args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp
    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['inference'] = True
    render_kwargs_test['raw_noise_std'] = 0.

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    global_step = start
    # Move testing data to GPU
    nerf = nerf.cuda()
    crf = crf.cuda()
    pose_interpolator = pose_interpolator.cuda()
    if timestamps_sampler is not None:
        timestamps_sampler = timestamps_sampler.cuda()
    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            render_poses = llff_dataset.poses if args.render_test else llff_dataset.render_poses
            testsavedir = os.path.join(basedir, expname,
                                       f"renderonly"
                                       f"_{'test' if args.render_test else 'path'}"
                                       f"_{start:06d}")

            if os.path.exists(testsavedir):
                all_versions = sorted(glob(testsavedir + "_ver*"))
                if len(all_versions) == 0:
                    ver = 0
                else:
                    ver = max([int(p.split("_ver")[1]) for p in all_versions]) + 1
                testsavedir = testsavedir + f"_ver{ver}"

            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)
            np.save(os.path.join(testsavedir, 'render_poses.npy'), render_poses.cpu().numpy())

            dummy_num = ((len(render_poses) - 1) // args.num_gpu + 1) * args.num_gpu - len(render_poses)
            dummy_poses = torch.eye(3, 4).unsqueeze(0).expand(dummy_num, 3, 4).type_as(render_poses)
            print(f"Append {dummy_num} # of poses to fill all the GPUs")
            torch.cuda.empty_cache()

            # measure rendering speed
            torch.cuda.synchronize()
            time0 = time.time()
            with torch.no_grad():
                nerf.eval()
                crf.eval()
                pose_interpolator.eval()
                if timestamps_sampler is not None:
                    timestamps_sampler.eval()
                rgbshdr, disps = nerf(
                    H, W, K, args.chunk // 2,
                    poses=torch.cat([render_poses, dummy_poses], dim=0),
                    render_kwargs=render_kwargs_test,
                    render_factor=args.render_factor,
                             )
            rgbshdr = crf(rgbshdr, mode="encode_rgb", chunk=8)
            torch.cuda.synchronize()
            time1 = time.time()
            print(f"Time for rendering {len(render_poses)} views: {time1 - time0} sec,"
                  f" avg {(time1 - time0) / len(render_poses)} sec")

            rgbshdr = rgbshdr[:len(rgbshdr) - dummy_num]
            disps = (1. - disps)
            disps = disps[:len(disps) - dummy_num].cpu().numpy()
            rgbs = rgbshdr
            rgbs = rgbs.cpu().numpy()

            for rgb_idx, rgb in enumerate(rgbs):
                rgb8 = to8b(rgb)
                np.save(os.path.join(testsavedir, f'{rgb_idx:03d}_disp.npy'), disps[rgb_idx])
                curr_disp = to8b(disps[rgb_idx] / disps[rgb_idx].max())
                imageio.imwrite(os.path.join(testsavedir, f'{rgb_idx:03d}.png'), rgb8)
                imageio.imwrite(os.path.join(testsavedir, f'{rgb_idx:03d}_disp.png'),
                                cv2.applyColorMap(255 - curr_disp, cv2.COLORMAP_TWILIGHT_SHIFTED))

            prefix = 'epi_' if args.render_epi else ''
            imageio.mimwrite(os.path.join(testsavedir, f'{prefix}video.mp4'), rgbs, fps=30, quality=9)
            disps = to8b(disps / disps.max())
            imageio.mimwrite(os.path.join(testsavedir, f'{prefix}video_disp.mp4'), disps, fps=30, quality=9)

    N_iters = args.N_iters + 1
    print('Begin')

    def detach_pose_interpolator_till(value, i):
        if args.pose_interpolator_refine_detach_till is not None and i < args.pose_interpolator_refine_detach_till:
            return value.detach()
        else:
            return value

    start = start + 1
    for i in trange(start, N_iters):
        is_last_iter = i == N_iters - 1
        if not args.skip_train:
            time0 = time.time()
            #####  Core optimization loop  #####
            nerf.train()
            crf.train()
            pose_interpolator.train()
            if timestamps_sampler is not None:
                timestamps_sampler.train()

            if i == args.kernel_start_iter:
                torch.cuda.empty_cache()

            batch_data = next(train_iterator)
            batch_data = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v
                          for k, v in batch_data.items()}

            loss_bck = 0.0
            use_prior_loss = args.use_prior_images is not None and args.prior_start_iter <= i < args.prior_end_iter

            if timestamps_sampler is not None:
                force_naive = False

                exposure_tms, exposure_w = timestamps_sampler(batch_data["images_tms"], batch_data["images_idx"]) # [N, Nexp]
                n_rays, n_exposures = exposure_tms.shape
                exposure_poses = detach_pose_interpolator_till(pose_interpolator(exposure_tms.reshape(-1), progress=i/N_iters,
                                                                                 refine=i >= args.pose_interpolator_refine_start_iter), i)
                exposure_rays_x = batch_data['rays_x'][:, None].repeat(1, n_exposures, 1).reshape(-1, 1)
                exposure_rays_y = batch_data['rays_y'][:, None].repeat(1, n_exposures, 1).reshape(-1, 1)
                batch_data['rays_weights'] = exposure_w

                exposure_rays = torch.stack(get_rays_pix(
                    torch.stack([exposure_rays_x, exposure_rays_y], dim=-1), llff_dataset.K, exposure_poses,
                    add_halfpix=False), dim=-1)
                exposure_rays = exposure_rays.reshape(n_rays, n_exposures, 3, -1)
            else:
                force_naive = True

                exposure_rays = torch.stack(get_rays_pix(
                    torch.cat([batch_data['rays_x'], batch_data['rays_y']], dim=-1), llff_dataset.K,
                    detach_pose_interpolator_till(pose_interpolator(batch_data["images_tms"], progress=i / N_iters,
                                      refine=i >= args.pose_interpolator_refine_start_iter), i),
                    add_halfpix=False), dim=-1)

            rgb, rgb0, extra_loss, extra_tensor = nerf(H, W, K, chunk=args.chunk,
                                                       rays=exposure_rays, rays_info=batch_data, retraw=True,
                                                       force_naive=force_naive or (i < args.kernel_start_iter),
                                                       return_pts0_rgb=global_step < kernel_end_warmup_iter or
                                                                       use_prior_loss,
                                                       **render_kwargs_train)
            rgb = crf(rgb, mode="encode_rgb", skip_learn_crf=i<args.tone_mapping_start_learn_iter)
            rgb0 = crf(rgb0, mode="encode_rgb", skip_learn_crf=i<args.tone_mapping_start_learn_iter)

            # Compute Losses
            # =====================
            loss = 0.0
            target_rgb = batch_data['rgbsf'].squeeze(-2)

            if i > args.blur_loss_after:
                img_loss = img2mse(rgb, target_rgb)
                psnr = mse2psnr(img_loss)

                if rgb0 is not None:
                    img_loss0 = img2mse(rgb0, target_rgb)
                    img_loss = img_loss + img_loss0
                loss += img_loss
            else:
                img_loss = torch.tensor(0.0)
                psnr = torch.tensor(0.0)

            if (args.kernel_start_warmup_mode != "step" and
                args.kernel_start_iter <= global_step < kernel_end_warmup_iter) or use_prior_loss:
                prior_loss = 0.0
                target_rgb_pts0 = target_rgb if not use_prior_loss else batch_data['rgbsf_prior'].squeeze(-2)
                # Directly apply the loss between the mid-exposure ray and the blur color, as done before kernel start
                for outname in ["stage0_rgb_pts0", "stage1_rgb_pts0", "stage1_rgb1_pts0"]:
                    if outname in extra_tensor:
                        prior_loss += img2mse(crf(extra_tensor[outname], mode="encode_rgb",
                                                 skip_learn_crf=i<args.tone_mapping_start_learn_iter),
                                             target_rgb_pts0)

                extra_loss[f"prior_{args.use_prior_images}_loss"] = prior_loss
                w_prior_override = None
                if i <= args.blur_loss_after:  # print this psnr
                    psnr = mse2psnr(extra_loss[f"prior_{args.use_prior_images}_loss"])
                    w_prior_override = 1.0

                if use_prior_loss:
                    w_prior_ = w_prior_override if w_prior_override is not None else w_prior(global_step)
                    loss = loss + extra_loss[f"prior_{args.use_prior_images}_loss"] * w_prior_
                else:
                    # Interpolate between before-kernel-start mode and after-kernel-start mode
                    loss = w_kernel(global_step) * loss + (1 - w_kernel(global_step)) * prior_loss

            extra_loss.update({k: torch.mean(v) for k, v in extra_loss.items()})
            if "TV" in extra_loss:
                loss = loss + extra_loss["TV"] * args.kernel_tv_loss_weight

            if args.add_event_egm and (args.add_event_egm_startiter is None or i >= args.add_event_egm_startiter):
                ev_batch_data = next(train_ev_iterator)
                ev_batch_data = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v
                                 for k, v in ev_batch_data.items()}

                events_rays_start = torch.stack(get_rays_pix(
                    ev_batch_data['events_coords'], llffev_dataset.K,
                    detach_pose_interpolator_till(pose_interpolator(ev_batch_data["events_tms_start"], progress=i/N_iters,
                                      refine=i >= args.pose_interpolator_refine_start_iter), i),
                    add_halfpix=llffev_dataset.integer_coords), dim=-1)
                events_rays_end = torch.stack(get_rays_pix(
                    ev_batch_data['events_coords'], llffev_dataset.K,
                    detach_pose_interpolator_till(pose_interpolator(ev_batch_data["events_tms_end"], progress=i/N_iters,
                                      refine=i >= args.pose_interpolator_refine_start_iter), i),
                    add_halfpix=llffev_dataset.integer_coords), dim=-1)

                n_exp, n_exp_start, n_exp_end = 0, 0, 0
                events_coords_ids = ev_batch_data["events_coords_ids"]
                events_neg_pol_cumsum = ev_batch_data["events_neg_pol_cumsum"]
                events_pos_pol_cumsum = ev_batch_data["events_pos_pol_cumsum"]
                events_color_map = ev_batch_data["events_color_map"]

                cumsum_pols = torch.stack([events_neg_pol_cumsum, events_pos_pol_cumsum], dim=-1)
                bii = (events_threshold_negpos * cumsum_pols).sum(-1)  # [N,2] -> [N]

                ev_crf_kwargs = {"tonemap_only": True} if args.event_egm_use_colorevents else {}
                if args.tone_mapping_events_add_bii == 'pos-neg':
                    ev_crf_extra_feat = torch.stack([events_neg_pol_cumsum, events_pos_pol_cumsum], dim=-1)
                elif args.tone_mapping_events_add_bii == 'color-pos-neg':
                    color_events_neg_pol_cumsum = events_neg_pol_cumsum.new_zeros([events_color_map.shape[0], 3])
                    color_events_pos_pol_cumsum = events_pos_pol_cumsum.new_zeros([events_color_map.shape[0], 3])
                    color_events_neg_pol_cumsum[events_color_map] = events_neg_pol_cumsum
                    color_events_pos_pol_cumsum[events_color_map] = events_pos_pol_cumsum
                    ev_crf_extra_feat = torch.stack([color_events_neg_pol_cumsum, color_events_pos_pol_cumsum], dim=-1)
                else:
                    ev_crf_extra_feat = None

                all_start_rgb, all_start_rgb0, start_extra_loss, start_extra_tensor = nerf(
                    H, W, K, chunk=args.chunk,
                    rays=events_rays_start, rays_info=None,
                    retraw=True, force_naive=True,  # Does not use the kernel network
                    **render_kwargs_train)
                ev_start_luma = crf(all_start_rgb, mode="encode_luma",
                                     skip_learn_crf=i<args.tone_mapping_start_learn_iter,
                                     ev_extra_feat=ev_crf_extra_feat, **ev_crf_kwargs)
                ev_start_luma0 = crf(all_start_rgb0, mode="encode_luma",
                                      skip_learn_crf=i<args.tone_mapping_start_learn_iter,
                                      ev_extra_feat=ev_crf_extra_feat,
                                      **ev_crf_kwargs)

                all_end_rgb, all_end_rgb0, end_extra_loss, end_extra_tensor = nerf(
                    H, W, K, chunk=args.chunk,
                    rays=events_rays_end, rays_info=None,
                    retraw=True, force_naive=True,  # Does not use the kernel network
                    **render_kwargs_train)
                ev_end_luma = crf(all_end_rgb, mode="encode_luma",
                                   skip_learn_crf=i<args.tone_mapping_start_learn_iter,
                                   ev_extra_feat=ev_crf_extra_feat, **ev_crf_kwargs)
                ev_end_luma0 = crf(all_end_rgb0, mode="encode_luma",
                                    skip_learn_crf=i<args.tone_mapping_start_learn_iter,
                                    ev_extra_feat=ev_crf_extra_feat, **ev_crf_kwargs)

                if args.add_event_egm:
                    event_egm_parts = []
                    if all_start_rgb0 is not None and all_end_rgb0 is not None:
                        if "stage0" in args.add_event_egm_stages:
                            event_egm_parts.append(egm_loss(ev_start_luma0, ev_end_luma0, bii,
                                                            color_mask=events_color_map,
                                                            color_weight=args.event_egm_use_color_weights
                                                            if i > args.event_egm_color_weights_start_iter else None))
                    if "stage1" in args.add_event_egm_stages:
                        event_egm_parts.append(egm_loss(ev_start_luma, ev_end_luma, bii,
                                                        color_mask=events_color_map,
                                                        color_weight=args.event_egm_use_color_weights
                                                        if i > args.event_egm_color_weights_start_iter else None))

                    extra_loss["event_egm"] = sum(event_egm_parts)

                    loss += extra_loss["event_egm"] * w_events_egm(global_step)

            optimizer.zero_grad()
            loss.backward()

            if args.clip_grads_norm is not None:
                nn.utils.clip_grad_norm_(nerf.parameters(),
                                         max_norm=args.clip_grads_norm,
                                         norm_type=2)

            optimizer.step()

            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            for param_group in optimizer.param_groups:
                if args.pose_interpolator_warmup_iters > 0 and param_group["name"] in ["interpolator_rot", "interpolator_tra"] \
                        and (global_step - args.pose_interpolator_refine_start_iter) < args.pose_interpolator_warmup_iters:
                    refine_iter_step = global_step - args.pose_interpolator_refine_start_iter
                    if refine_iter_step > 0:
                        scale = (1 - args.pose_interpolator_warmup_factor) * refine_iter_step / args.pose_interpolator_warmup_iters + args.pose_interpolator_warmup_factor
                        param_group['lr'] = param_group['initial_lr'] * scale
                elif args.lrate_warmup_iters > 0 and global_step < args.lrate_warmup_iters:
                    scale = (1 - args.lrate_warmup_factor) * global_step / args.lrate_warmup_iters + args.lrate_warmup_factor
                    param_group['lr'] = param_group['initial_lr'] * scale
                else:
                    new_lrate = param_group['initial_lr'] * (decay_rate ** (global_step / decay_steps))
                    param_group['lr'] = new_lrate
            ################################

            # Rest is logging
            if (i % args.i_weights == 0 and i > 0) or is_last_iter:
                path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
                if os.path.exists(path):
                    # Encapsulates '[' in brackets to escape, otherwise it will be interpreted as a character set
                    ver_path = sorted(glob(os.path.join(basedir, expname, '{:06d}_ver*.tar'.format(i)).replace('[', '[[]')))
                    latest_ver = max([int(os.path.basename(p).split('_ver')[-1].split('.')[0]) for p in ver_path]) \
                        if len(ver_path) > 0 else 0
                    path = os.path.join(basedir, expname, '{:06d}_ver{:02d}.tar'.format(i, latest_ver + 1))

                if not os.path.exists(path):
                    torch.save({
                        'wandb_id': wandb_id,
                        'global_step': global_step,
                        'crf_state_dict': crf.state_dict(),
                        'network_state_dict': nerf.state_dict(),
                        'pose_interpolator_state_dict': pose_interpolator.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        **({'timestamps_sampler_state_dict': timestamps_sampler.state_dict()} if timestamps_sampler is not None else {}),
                    }, path)
                    print('Saved checkpoints at', path)
                else:
                    # Versioning did not work for some reason, we avoid overwriting the checkpoint
                    print('Checkpoint already exists at', path)

        if is_last_iter or args.skip_train:  # Run test evaluation when training finishes
            torch.cuda.empty_cache()
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}_testtime_ver00'.format(i))
            poses_path  = os.path.join(basedir, expname,  "poses_{:06d}_testtime_ver00".format(i), "laptop")
            if os.path.exists(testsavedir):
                # Encapsulates '[' in brackets to escape, otherwise it will be interpreted as a character set
                ver_path = sorted(glob(os.path.join(basedir, expname, 'testset_{:06d}_testtime_ver*'.format(i)).replace('[', '[[]')))
                latest_ver = max([int(os.path.basename(p).split('_ver')[-1]) for p in ver_path]) \
                    if len(ver_path) > 0 else 0
                testsavedir = os.path.join(basedir, expname, 'testset_{:06d}_testtime_ver{:02d}'.format(i, latest_ver + 1))
                poses_path  = os.path.join(basedir, expname,  "poses_{:06d}_testtime_ver{:02d}".format(i, latest_ver + 1), "laptop")
            os.makedirs(testsavedir, exist_ok=True)
            os.makedirs(poses_path, exist_ok=True)

            # Dump refined image poses vs GT, and refined full poses vs GT (if allknown exists)
            poses_to_dump_dict = {}

            # Refined image poses vs GT
            with torch.no_grad():
                pose_interpolator.eval()
                dump_es_poses = pose_interpolator(gt_seenim_tms.cuda(), progress=i / N_iters, refine=True)
                pose_interpolator.train()
            # Put (<gt_tms>, <gt_poses>, <est_tms>, <est_poses>) tuples to later save as txt and npy
            poses_to_dump_dict["refined/images"] = (gt_seenim_tms, gt_seenim_poses, gt_seenim_tms, dump_es_poses)

            # Refined full poses vs GT
            if gt_allknown_tms.shape[0] > 0:
                with torch.no_grad():
                    pose_interpolator.eval()
                    dump_es_poses = pose_interpolator(gt_allknown_tms.cuda(), progress=i / N_iters, refine=True)
                    pose_interpolator.train()
                # Put (<gt_tms>, <gt_poses>, <est_tms>, <est_poses>) tuples to later save as txt and npy
                poses_to_dump_dict["refined/allknown"] = (gt_allknown_tms, gt_allknown_poses, gt_allknown_tms, dump_es_poses)

            for dump_name, (dump_gt_tms, dump_gt_poses, dump_es_tms, dump_es_poses) in poses_to_dump_dict.items():
                dump_path = os.path.join(poses_path, dump_name)
                os.makedirs(dump_path, exist_ok=True)

                dump_gt_ts_tran_quat = np.concatenate([
                    dump_gt_tms.reshape(-1, 1) * 1e-6,  # usec to sec
                    llff_dataset.poses_to_original(dump_gt_poses.cpu().numpy(), "mocap")],
                    axis=-1)
                dump_est_ts_tran_quat = np.concatenate([
                    dump_es_tms.reshape(-1, 1) * 1e-6,  # usec to sec
                    llff_dataset.poses_to_original(dump_es_poses.cpu().numpy(), "mocap")],
                    axis=-1)
                np.savetxt(os.path.join(dump_path, "stamped_groundtruth.txt"), dump_gt_ts_tran_quat, fmt='%f')
                np.savetxt(os.path.join(dump_path, "stamped_traj_estimate.txt"), dump_est_ts_tran_quat, fmt='%f')

                dump_gt_pose_bounds = llff_dataset.poses_to_original(dump_gt_poses.cpu().numpy(), "poses_bounds")
                dump_es_pose_bounds = llff_dataset.poses_to_original(dump_es_poses.cpu().numpy(), "poses_bounds")
                # save as npy
                np.save(os.path.join(dump_path, "stamped_groundtruth_poses_bounds.npy"), dump_gt_pose_bounds)
                np.save(os.path.join(dump_path, "stamped_traj_estimate_poses_bounds.npy"), dump_es_pose_bounds)

            poses = llff_dataset.test_poses
            if args.test_pose_alignment:
                with torch.no_grad():
                    pose_interpolator.eval()
                    # timestamps of training images (USLAM) as input; output: refined/trained traj
                    refined_poses = pose_interpolator(gt_seenim_tms.cuda(), progress=i / N_iters, refine=True)
                    pose_interpolator.train()
                try:
                    evo_result = evo_solution(gt_seenim_poses[..., :3, :4],  # transformation from GT (mocap) traj to refined traj (trained)
                                              refined_poses[..., :3, :4].to(gt_seenim_poses.dtype))
                    poses = evo_align(poses, evo_result=evo_result)  # transform test poses GT to test poses refined/trained
                except:
                    print("EVO failed, using the original poses")
                    poses = poses

            target_rgb_ldr = llff_dataset.test_images
            target_rgb_ldr = target_rgb_ldr.cuda()

            if args.test_pose_refinement is not None:
                print("Starting test-time pose refinement at iter", i)
                testtime_N_imgs = poses.shape[0]
                testtime_N_rays_per_img = args.test_pose_refinement_rays_per_img
                testtime_lrate_decay = args.test_pose_refinement_lrate_decay
                testtime_lrate = args.test_pose_refinement_lrate
                testtime_startiter = 0

                testtime_refiner = TestTimePoseRefiner(poses, mode=args.test_pose_refinement) # refine the test poses in the trained world
                testtime_refiner = testtime_refiner.cuda()

                testtime_optimizers = {
                    0: get_optimizer(args.test_pose_refinement_optim)(
                        params=testtime_refiner.parameters(), lr=testtime_lrate),
                    1: get_optimizer(args.test_pose_refinement_2stage_optim)(
                        params=testtime_refiner.parameters(), lr=testtime_lrate),
                }

                # 0. Load the latest checkpoint, if testtime was already done
                testtime_ckpts = [os.path.join(basedir, expname, f)
                                  for f in sorted(os.listdir(os.path.join(basedir, expname)))
                                  if '.tar' in f and 'testtime' in f]
                print('Found testtime ckpts', ckpts)

                if len(testtime_ckpts) > 0 and not args.no_testtime_reload:
                    testtime_ckpt_path = testtime_ckpts[-1]
                    print('Reloading testtime_poses from', testtime_ckpt_path)
                    testtime_ckpt = torch.load(testtime_ckpt_path)

                    smart_load_state_dict(testtime_refiner, testtime_ckpt, network_key="testtime_refiner_state_dict")
                    testtime_startiter = testtime_ckpt['testtime_global_step']

                pixmse, img_psnrs = None, torch.zeros([testtime_N_imgs])

                pbar = tqdm(range(testtime_startiter, args.test_pose_refinement_iters), desc="[TEST ALIGNMENT]")
                for j in pbar:
                    stage = int(j > args.test_pose_refinement_2stage_perc * args.test_pose_refinement_iters)
                    testtime_refiner.fix_independent(j < args.test_pose_refinement_independent_start)
                    testtime_refiner.fix_shared(stage == 1)

                    testtime_optimizer = testtime_optimizers[stage]

                    # 1. Sample pixels
                    img_id = (torch.arange(testtime_N_imgs).reshape(testtime_N_imgs, 1)
                              .repeat(1, testtime_N_rays_per_img).reshape(-1).cuda().long())
                    ray_x = torch.randint(0, W, (testtime_N_imgs * testtime_N_rays_per_img,)).cuda().float()
                    ray_x = ray_x.reshape(testtime_N_imgs, testtime_N_rays_per_img)
                    ray_y = torch.randint(0, H, (testtime_N_imgs * testtime_N_rays_per_img,)).cuda().float()
                    ray_y = ray_y.reshape(testtime_N_imgs, testtime_N_rays_per_img)

                    imgs_under = torch.tensor(img_psnrs) < args.test_pose_refinement_errorbased_underpsnr
                    if bool(imgs_under.any()) and pixmse is not None:
                        ray_xy = torch.multinomial(pixmse[imgs_under].reshape(-1, W * H), testtime_N_rays_per_img,
                                                   replacement=False)
                        ray_x[imgs_under] = (ray_xy % W).float()
                        ray_y[imgs_under] = (ray_xy // W).float()
                    ray_x = ray_x.reshape(-1)
                    ray_y = ray_y.reshape(-1)

                    # 2. Get updated poses
                    refined_poses = testtime_refiner(img_id)

                    # 3. Cast rays from optimized poses
                    rays_o, rays_d = get_rays_pix(torch.stack([ray_x, ray_y], dim=-1), K, refined_poses)
                    rays = torch.stack([rays_o, rays_d], dim=-2).permute(0, 2, 1)  # [N_rays, 3, 2]

                    # 4. Render
                    rgb, rgb0, extra_loss, extra_tensor = nerf(H, W, K, chunk=args.chunk,
                                                               force_naive=True, rays=rays, retraw=True,
                                                               **render_kwargs_test)
                    rgb = crf(rgb, mode="encode_rgb", skip_learn_crf=False)
                    rgb0 = crf(rgb0, mode="encode_rgb", skip_learn_crf=False)

                    # 5. Get gt color
                    target_s_novel = target_rgb_ldr[img_id, ray_y.long(), ray_x.long()]

                    # 6. Compute loss
                    loss_sharp = img2mse(rgb, target_s_novel)
                    psnr_sharp = mse2psnr(loss_sharp)
                    if rgb0 is not None:
                        img_loss0 = img2mse(rgb0, target_s_novel)
                        loss_sharp = loss_sharp + img_loss0

                    pbar.set_postfix(loss=f"{loss_sharp.item():.4f}", psnr=f"{psnr_sharp.item():.2f}")

                    testtime_optimizer.zero_grad()
                    loss_sharp.backward()
                    testtime_optimizer.step()

                    if testtime_lrate_decay > 0:
                        decay_rate_sharp = 0.01
                        decay_steps_sharp = testtime_lrate_decay * 100
                        new_lrate_novel = testtime_lrate * (decay_rate_sharp ** (j / decay_steps_sharp))
                        for param_group in testtime_optimizer.param_groups:
                            if (j / decay_steps_sharp) <= 1.:
                                param_group['lr'] = new_lrate_novel

                    if (j % args.test_pose_refinement_valid_every == 0 and testtime_refiner.can_update_best_independent) or \
                        j == args.test_pose_refinement_iters - 1:  # track the best poses for each image individually
                        # Render full images for visualization
                        poses = testtime_refiner(torch.arange(testtime_N_imgs).cuda().long())[:, :3, :4]
                        dummy_num = ((len(poses) - 1) // args.num_gpu + 1) * args.num_gpu - len(poses)
                        dummy_poses = torch.eye(3, 4).unsqueeze(0).expand(dummy_num, 3, 4).type_as(poses)
                        with torch.no_grad():
                            nerf.eval()
                            crf.eval()
                            rgbs, disps = nerf(H, W, K, args.chunk // 2, poses=torch.cat([poses, dummy_poses], dim=0),
                                               render_kwargs=render_kwargs_test)
                            rgbs = crf(rgbs, mode="encode_rgb", chunk=8)
                            img_psnrs = compute_img_metric(rgbs, target_rgb_ldr, 'psnr', avg=False)

                            pixmse = ((rgbs - target_rgb_ldr) ** 2).mean(-1)
                            min_vals = pixmse.amin(dim=(1, 2), keepdim=True)  # Shape [B, 1, 1]
                            max_vals = pixmse.amax(dim=(1, 2), keepdim=True)  # Shape [B, 1, 1]
                            pixmse = (pixmse - min_vals) / (max_vals - min_vals + 1e-8)

                            testtime_refiner.update_best_independent(img_psnrs)
                            nerf.train()
                            crf.train()
                pbar.close()

                # 7. Override the poses with the refined ones
                testtime_refiner.set_best_independent()
                poses = testtime_refiner(torch.arange(testtime_N_imgs).cuda().long())[:, :3, :4]
                poses_to_save = llff_dataset.poses_to_original(poses.detach().cpu().numpy(), "poses_bounds")
                np.save(os.path.join(testsavedir, "testtime_refiner_poses.npy"), poses_to_save)
                # 8. Save the refined poses
                testtime_path = os.path.join(basedir, expname, '{:06d}_testtime_ver00.tar'.format(i))
                if os.path.exists(testtime_path):
                    # Encapsulates '[' in brackets to escape, otherwise it will be interpreted as a character set
                    ver_path = sorted(glob(os.path.join(
                        basedir, expname, '{:06d}_testtime_ver*.tar'.format(i)).replace('[', '[[]')))
                    latest_ver = max([int(os.path.basename(p).split('_ver')[-1].split('.')[0]) for p in ver_path]) \
                        if len(ver_path) > 0 else 0
                    testtime_path = os.path.join(
                        basedir, expname, '{:06d}_testtime_ver{:02d}.tar'.format(i, latest_ver + 1))

                if not os.path.exists(testtime_path):
                    torch.save({
                        'testtime_global_step': j,
                        'testtime_refiner_state_dict': testtime_refiner.state_dict(),
                    }, testtime_path)
                    print('Saved testime checkpoints at', testtime_path)

            print('test poses shape', poses.shape)
            dummy_num = ((len(poses) - 1) // args.num_gpu + 1) * args.num_gpu - len(poses)
            dummy_poses = torch.eye(3, 4).unsqueeze(0).expand(dummy_num, 3, 4).type_as(poses)
            print(f"Append {dummy_num} # of poses to fill all the GPUs")
            with torch.no_grad():
                nerf.eval()
                crf.eval()
                pose_interpolator.eval()
                if timestamps_sampler is not None:
                    timestamps_sampler.eval()

                rgbs, disps = nerf(H, W, K, args.chunk // 2, poses=torch.cat([poses, dummy_poses], dim=0),
                                   render_kwargs=render_kwargs_test)
                rgbs = crf(rgbs, mode="encode_rgb", chunk=8)
                rgbs = rgbs[:len(rgbs) - dummy_num]
                rgbs_save = rgbs  # (rgbs - rgbs.min()) / (rgbs.max() - rgbs.min())
                disps = (1. - disps)

                for j, (rgb, gtrgb, disp) in enumerate(zip(rgbs, target_rgb_ldr, disps)):
                    assert rgb.shape == gtrgb.shape and len(rgb.shape) == 3
                    rgb = rgb.cpu().numpy()
                    disp = disp.cpu().numpy()
                    gtrgb = gtrgb.cpu().numpy()
                    pixmse = ((rgb - gtrgb) ** 2).mean(-1)

                    logger.image(f"images/test_groundtruth_{j}", to8b(gtrgb), step=global_step)
                    logger.image(f"images/test_prediction_{j}", to8b(rgb), step=global_step)
                    logger.image(f"images/test_depth_{j}",
                                 cv2.applyColorMap(255 - to8b(disp / float(disps.max())),
                                                   cv2.COLORMAP_TWILIGHT_SHIFTED),
                                 step=global_step)
                    logger.image(f"images/test_errmap_{j}",
                                 cv2.applyColorMap(255 - to8b(pixmse / float(pixmse.max())),
                                                   cv2.COLORMAP_TWILIGHT_SHIFTED),
                                 step=global_step)

                metrics_str = ""
                # evaluation
                test_mse = compute_img_metric(rgbs, target_rgb_ldr, 'mse')
                test_psnr = compute_img_metric(rgbs, target_rgb_ldr, 'psnr')
                test_ssim = compute_img_metric(rgbs, target_rgb_ldr, 'ssim')
                test_lpips = compute_img_metric(rgbs, target_rgb_ldr, 'lpips')
                if isinstance(test_lpips, torch.Tensor):
                    test_lpips = test_lpips.item()

                logger.scalar("test/mse", test_mse, step=global_step)
                logger.scalar("test/psnr", test_psnr, step=global_step)
                logger.scalar("test/ssim", test_ssim, step=global_step)
                logger.scalar("test/lpips", test_lpips, step=global_step)
                metrics_str += f"MSE:{test_mse:.8f} PSNR:{test_psnr:.8f} " \
                               f"SSIM:{test_ssim:.8f} LPIPS:{test_lpips:.8f}"

                with open(test_metric_file, 'a') as outfile:
                    outfile.write(f"iter{i}/globalstep{global_step}: {metrics_str}\n")
                print(f"[TEST]  Iter: {i} {metrics_str}")

                for rgb_idx, rgb in enumerate(rgbs_save):
                    rgb8 = to8b(rgb.cpu().numpy())
                    filename = os.path.join(testsavedir, f'{rgb_idx:03d}.png')
                    imageio.imwrite(filename, rgb8)

            torch.cuda.empty_cache()
            print('Saved test set')

        if (i % args.i_video == 0 and i > 0) or is_last_iter:
            torch.cuda.empty_cache()
            # Turn on testing mode
            torch.cuda.empty_cache()
            # Turn on testing mode
            with torch.no_grad():
                nerf.eval()
                crf.eval()
                pose_interpolator.eval()
                if timestamps_sampler is not None:
                    timestamps_sampler.eval()

                if args.render_test:
                    render_poses = llff_dataset.test_poses
                elif args.render_train_interp:
                    render_poses, render_tms = render_path_interpolated(allknown_poses, allknown_tms, num_poses=120)
                else:
                    render_poses = llff_dataset.render_poses

                if args.test_pose_alignment:
                    pose_interpolator.eval()
                    known_poses = allknown_poses.cuda()
                    known_tms = allknown_tms.cuda()
                    refined_poses = pose_interpolator(known_tms, progress=i / N_iters, refine=True)
                    pose_interpolator.train()

                    # from GT to predicted
                    try:
                        evo_result = evo_solution(known_poses[..., :3, :4],
                                                  refined_poses[..., :3, :4].to(known_poses.dtype))
                        render_poses = evo_align(render_poses, evo_result=evo_result)
                    except:
                        print("EVO failed, using the original poses")
                        render_poses = render_poses

                rgbs, disps = nerf(H, W, K, args.chunk // 2, poses=render_poses, render_kwargs=render_kwargs_test)
                lumas = crf(rgbs, mode="encode_luma", chunk=8)  # Zero-pad the CRF if learned with extra bii features
                rgbs = crf(rgbs, mode="encode_rgb", chunk=8)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))

            rgbs = (rgbs - rgbs.min()) / (rgbs.max() - rgbs.min())
            rgbs = rgbs.cpu().numpy()
            disps = disps.cpu().numpy()

            logger.video(f"test/spiral_rgb", moviebase + 'rgb.mp4', to8b(rgbs),
                         fps=30, step=global_step)
            logger.video(f"test/spiral_disp", moviebase + 'disp.mp4', to8b(disps / disps.max()),
                         fps=30, step=global_step)
            torch.cuda.empty_cache()

            rendersavedir = os.path.join(basedir, expname, 'render_{:06d}'.format(i))
            os.makedirs(os.path.join(rendersavedir, "disps"), exist_ok=True)
            os.makedirs(os.path.join(rendersavedir, "disps_raw"), exist_ok=True)
            os.makedirs(os.path.join(rendersavedir, "rgbs"), exist_ok=True)
            for rgb_idx, (rgb, disp) in enumerate(zip(rgbs, disps)):
                rgb8 = to8b(rgb)
                disp8 = cv2.applyColorMap(to8b(disp / float(disps.max())),
                                  cv2.COLORMAP_TWILIGHT_SHIFTED)

                imageio.imwrite(os.path.join(rendersavedir, f'rgbs/{rgb_idx:03d}.png'), rgb8)
                imageio.imwrite(os.path.join(rendersavedir, f'disps/{rgb_idx:03d}.png'), disp8)
                np.save(os.path.join(rendersavedir, f'disps_raw/{rgb_idx:03d}.npy'), disp)

        if (i % args.i_tensorboard == 0 and i > 0) or is_last_iter:
            if not args.no_log_grads_norm:
                for k, v in grads_norm(nerf).items():
                    logger.scalar(f"gradients/{k}", float(v), global_step)
                for k, v in grads_norm(crf).items():
                    logger.scalar(f"gradients/crf_{k}", float(v), global_step)
                if timestamps_sampler is not None:
                    for k, v in grads_norm(timestamps_sampler).items():
                        logger.scalar(f"gradients/timestamps_sampler_{k}", float(v), global_step)
                if refiner is not None:
                    for k, v in grads_norm(refiner).items():
                        logger.scalar(f"gradients/refiner_{k}", float(v), global_step)

            if refiner is not None and i >= args.pose_interpolator_refine_start_iter \
                    and i % (args.i_tensorboard_img_mult * args.i_tensorboard) == 0:
                with torch.no_grad():
                    pose_interpolator.eval()

                    # Plot refined image poses vs GT
                    with torch.no_grad():
                        es_seenim_poses = pose_interpolator(gt_seenim_tms.cuda(), progress=i / N_iters, refine=True)
                        try:
                            es_seenim_poses = evo_align(es_seenim_poses, gt_seenim_poses)
                        except:
                            print("EVO failed, using the original poses")
                    ate, mpe, figs = trajectory_errors(gt_seenim_poses, gt_seenim_tms, es_seenim_poses, gt_seenim_tms,
                                                 plot=True, est_name="refined(seen_images)",
                                                 save_folder=os.path.join(basedir, expname, "trj_plots"),
                                                 save_prefix=f"gt_seenimg_poses_{global_step}")
                    logger.scalar("test/gt_seenim_poses_ate", ate, step=global_step)
                    logger.scalar("test/gt_seenim_poses_mpe", mpe, step=global_step)
                    for fig_path in figs:
                        name = os.path.splitext(os.path.basename(fig_path))[0].replace(f"gt_seenimg_poses_{global_step}_", "")
                        logger.image(f"test/gt_seenimg_poses_{name}", cv2.imread(fig_path), step=global_step)

                    # If events, plot refined full poses vs GT (if allknown exists)
                    if gt_allknown_tms.shape[0] > 0:
                        with torch.no_grad():
                            es_allknown_poses = pose_interpolator(gt_allknown_tms.cuda(), progress=i / N_iters, refine=True)
                            try:
                                es_allknown_poses = evo_align(es_allknown_poses, gt_allknown_poses)
                            except:
                                print("EVO failed, using the original poses")
                                # es_allknown_poses = es_allknown_poses  
                        ate, mpe, figs = trajectory_errors(gt_allknown_poses, gt_allknown_tms, es_allknown_poses, gt_allknown_tms,
                                                 plot=True, est_name="refined(gt_allknown)",
                                                 save_folder=os.path.join(basedir, expname, "trj_plots"),
                                                 save_prefix=f"gt_allknown_poses_{global_step}")
                        logger.scalar("test/gt_allknown_poses_ate", ate, step=global_step)
                        logger.scalar("test/gt_allknown_poses_mpe", mpe, step=global_step)
                        for fig_path in figs:
                            name = os.path.splitext(os.path.basename(fig_path))[0].replace(f"gt_allknown_poses_{global_step}_", "")
                            logger.image(f"test/gt_allknown_poses_{name}", cv2.imread(fig_path), step=global_step)
                pose_interpolator.train()

            for pg_i, param_group in enumerate(optimizer.param_groups):
                pname = "param_group_" + param_group.get('name', str(pg_i))
                logger.scalar(f"learning_rates/{pname}", param_group['lr'], global_step)

            if not args.skip_train:
                logger.scalar("train/loss_img", img_loss.item(), global_step)
                logger.scalar("train/psnr", psnr.item(), global_step)

                for k, v in extra_loss.items():
                    logger.scalar(f"train/{k}", v.item(), global_step)

                if args.kernel_start_warmup_mode != "step":
                    logger.scalar(f"train/w_kernel", w_kernel(global_step), global_step)

                if args.use_prior_images:
                    logger.scalar(f"train/weight_prior", w_prior(global_step), global_step)

                if args.use_events:
                    if args.event_accumulate_step_scheduler != "constant":
                        # Reads the internal dataset global step to make sure
                        # the value is the one actually applied by the loader
                        dataset_global_step = llffev_dataset.global_step
                        logger.scalar(f"train/dataset_global_step", dataset_global_step, global_step)
                        logger.scalar(f"train/event_accum_min", llffev_dataset.event_accum_min_step(
                            dataset_global_step), global_step)
                        logger.scalar(f"train/event_accum_max", llffev_dataset.event_accum_max_step(
                            dataset_global_step), global_step)
                    if w_events_egm is not None:
                        logger.scalar(f"train/w_events_egm", w_events_egm(global_step), global_step)
                    if events_threshold_negpos is not None:
                        pix_ths_neg, pix_ths_pos = events_threshold_negpos[..., 0].mean(), events_threshold_negpos[..., 1].mean()
                        logger.scalar(f"train/pix_ths_neg", pix_ths_neg.float().item(), global_step)
                        logger.scalar(f"train/pix_ths_pos", pix_ths_pos.float().item(), global_step)

        if (i % args.i_print == 0 and i > 0) or is_last_iter:
            print(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
