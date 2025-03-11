"""Main training loop."""

import os
import csv
import time
import copy
import json
import pickle
import numpy as np
import torch
import dnnlib
import random
from torch import autocast
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from models.ldm.util import instantiate_from_config
from torch_utils.download_util import check_file_by_key

#----------------------------------------------------------------------------
# Load pre-trained models from the LDM codebase (https://github.com/CompVis/latent-diffusion) 
# and Stable Diffusion codebase (https://github.com/CompVis/stable-diffusion)

def load_ldm_model(config, ckpt, verbose=False):
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        dist.print0(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model

#----------------------------------------------------------------------------

def create_model(dataset_name=None, guidance_type=None, guidance_rate=None, device=None):
    model_path, classifier_path = check_file_by_key(dataset_name)
    dist.print0(f'Loading the pre-trained guided diffusion model from "{model_path}"...')

    # Load the guided diffusion UNet (ADM)
    from models.guided_diffusion.cg_model_loader import load_cg_model
    from models.networks_edm import CGPrecond, CFGPrecond

    if guidance_type == 'cg':  # Classifier guidance
        assert classifier_path is not None, "Classifier path required for classifier guidance"
        net, classifier = load_cg_model(model_path, classifier_path)
        net = CGPrecond(net, classifier, guidance_rate=guidance_rate).to(device)
    elif guidance_type in ['uncond', 'cfg']:  # Unconditional or classifier-free guidance
        with dnnlib.util.open_url(model_path, verbose=(dist.get_rank() == 0)) as f:
            net = pickle.load(f)['ema'].to(device)  # Load pre-trained UNet
        net = CFGPrecond(
            net,
            img_resolution=64 if dataset_name == 'imagenet64' else 256,  # Adjust based on dataset
            img_channels=3,
            guidance_rate=guidance_rate if guidance_type == 'cfg' else 1.0,
            guidance_type='classifier-free' if guidance_type == 'cfg' else 'uncond',
            label_dim=1000 if dataset_name in ['imagenet64', 'imagenet256'] else 0  # ImageNet has 1000 classes
        ).to(device)
    else:
        raise ValueError("guidance_type must be 'cg', 'cfg', or 'uncond' for guided diffusion")

    net.eval()
    return net

#----------------------------------------------------------------------------

def training_loop(
    run_dir='.',
    AMED_kwargs={},
    loss_kwargs={},
    optimizer_kwargs={},
    seed=0,
    batch_size=None,
    batch_gpu=None,
    total_kimg=20,
    kimg_per_tick=1,
    snapshot_ticks=1,
    state_dump_ticks=20,
    cudnn_benchmark=True,
    dataset_name=None,
    prompt_path=None,
    guidance_type=None,
    guidance_rate=0.,
    device=torch.device('cuda'),
    **kwargs,
):
    # Initialize
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark

    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load prompts for MS-COCO if applicable
    sample_captions = []
    if dataset_name == 'ms_coco' and guidance_type == 'cfg':
        prompt_path, _ = check_file_by_key('prompts')
        with open(prompt_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                sample_captions.append(row['text'])

    # Load guided diffusion model
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    net = create_model(dataset_name, guidance_type, guidance_rate, device)
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Construct AMED predictor
    dist.print0('Constructing AMED predictor...')
    AMED_kwargs.update(img_resolution=net.img_resolution)
    AMED_predictor = dnnlib.util.construct_class_by_name(**AMED_kwargs).train().requires_grad_(True).to(device)

    # Setup optimizer and loss
    loss_kwargs.update(
        num_steps=AMED_kwargs['num_steps'], sampler_stu=AMED_kwargs['sampler_stu'],
        sampler_tea=AMED_kwargs['sampler_tea'], M=AMED_kwargs['M'],
        schedule_type=AMED_kwargs['schedule_type'], schedule_rho=AMED_kwargs['schedule_rho'],
        afs=AMED_kwargs['afs'], max_order=AMED_kwargs['max_order'],
        sigma_min=net.sigma_min, sigma_max=net.sigma_max,
        predict_x0=AMED_kwargs['predict_x0'], lower_order_final=AMED_kwargs['lower_order_final']
    )
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=AMED_predictor.parameters(), **optimizer_kwargs)
    ddp = torch.nn.parallel.DistributedDataParallel(AMED_predictor, device_ids=[device], broadcast_buffers=False)

    # Training loop
    dist.print0(f'Training for {total_kimg} kimg...')
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    while True:
        # Generate latents and conditions
        latents = loss_fn.sigma_max * torch.randn([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        labels = c = uc = None
        if guidance_type == 'cg' and net.label_dim:  # Classifier guidance
            labels = torch.randint(net.label_dim, size=(batch_gpu,), device=device)
        elif guidance_type == 'cfg':
            if dataset_name == 'ms_coco':  # Text prompts
                prompts = random.sample(sample_captions, batch_gpu)
                c = net.model.get_learned_conditioning(prompts)
                uc = net.model.get_learned_conditioning(batch_gpu * [""]) if guidance_rate != 1.0 else None
            elif net.label_dim:  # Class labels (e.g., ImageNet)
                labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_gpu], device=device)]
        # Unconditional: no labels or conditions unless dataset requires it

        # Generate teacher trajectories
        with torch.no_grad():
            teacher_traj = loss_fn.get_teacher_traj(net=net, tensor_in=latents, labels=labels)
        # Training steps
        for step_idx in range(loss_fn.num_steps - 1):
            optimizer.zero_grad(set_to_none=True)
            for round_idx in range(num_accumulation_rounds):
                with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                    loss, stu_out = loss_fn(
                        AMED_predictor=ddp, net=net, tensor_in=latents, labels=labels,
                        step_idx=step_idx, teacher_out=teacher_traj[step_idx],
                        condition=c, unconditional_condition=uc
                    )
                    training_stats.report('Loss/loss', loss)
                    loss.sum().mul(1 / batch_gpu_total).backward()

            for param in AMED_predictor.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
            optimizer.step()

            if AMED_predictor.sampler_stu in ['euler', 'dpm', 'amed']:
                latents = teacher_traj[step_idx]
            else:
                latents = stu_out

        # Maintenance tasks (unchanged from original)
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # [Rest of the maintenance code remains unchanged]
        if done:
            break

    dist.print0('Exiting...')