import os
import json
import re
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import training_loop

@click.command()
@click.option('--dataset_name', help='Dataset name', type=click.Choice(['imagenet64', 'imagenet256']), required=True)
@click.option('--outdir', help='Where to save the results', type=str, default='./exps')
@click.option('--total_kimg', help='Number of images (k) for training', type=int, default=10)
@click.option('--num_steps', help='Number of time steps', type=click.IntRange(min=1), default=4)
@click.option('--sampler_stu', help='Student solver', type=click.Choice(['amed', 'dpm', 'dpmpp', 'euler', 'ipndm']), default='amed')
@click.option('--sampler_tea', help='Teacher solver', type=click.Choice(['heun', 'dpm', 'dpmpp', 'euler', 'ipndm']), default='heun')
@click.option('--M', help='Steps to insert between two adjacent steps', type=click.IntRange(min=1), default=1)
@click.option('--guidance_type', help='Guidance type', type=click.Choice(['cg', 'uncond']), default='uncond')
@click.option('--guidance_rate', help='Guidance rate', type=float, default=0.)
@click.option('--schedule_type', help='Time discretization schedule', type=click.Choice(['polynomial', 'logsnr', 'time_uniform', 'discrete']), default='polynomial')
@click.option('--schedule_rho', help='Time step exponent', type=click.FloatRange(min=0), default=7)
@click.option('--afs', help='Whether to use afs', type=bool, default=True)
@click.option('--scale_dir', help='Scale the gradient', type=click.FloatRange(min=0), default=0.01)
@click.option('--scale_time', help='Scale the gradient', type=click.FloatRange(min=0), default=0)
@click.option('--max_order', help='Max order for solvers', type=click.IntRange(min=1), default=3)
@click.option('--predict_x0', help='Whether to use data prediction mode', type=bool, default=True)
@click.option('--lower_order_final', help='Lower the order at final stages', type=bool, default=True)
@click.option('--batch', help='Total batch size', type=click.IntRange(min=1), default=512)
@click.option('--batch-gpu', help='Limit batch size per GPU', type=click.IntRange(min=1))
@click.option('--lr', help='Learning rate', type=click.FloatRange(min=0, min_open=True), default=5e-3)
@click.option('--bench', help='Enable cuDNN benchmarking', type=bool, default=True)
@click.option('--desc', help='String to include in result dir name', type=str)
@click.option('--nosubdir', help='Do not create a subdirectory for results', is_flag=True)
@click.option('--tick', help='How often to print progress', type=click.IntRange(min=1), default=10)
@click.option('--snap', help='How often to save snapshots', type=click.IntRange(min=1), default=10)
@click.option('--dump', help='How often to dump state', type=click.IntRange(min=1), default=50)
@click.option('--seed', help='Random seed', type=int)
@click.option('-n', '--dry-run', help='Print training options and exit', is_flag=True)

def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    c = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.AMED_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9, 0.999], eps=1e-8)

    c.AMED_kwargs.class_name = 'training.networks.AMED_predictor'
    c.AMED_kwargs.update(
        num_steps=opts.num_steps, sampler_stu=opts.sampler_stu, sampler_tea=opts.sampler_tea,
        M=opts.M, guidance_type=opts.guidance_type, guidance_rate=opts.guidance_rate,
        schedule_rho=opts.schedule_rho, schedule_type=opts.schedule_type, afs=opts.afs,
        dataset_name=opts.dataset_name, scale_dir=opts.scale_dir, scale_time=opts.scale_time,
        max_order=opts.max_order, predict_x0=opts.predict_x0, lower_order_final=opts.lower_order_final
    )
    c.loss_kwargs.class_name = 'training.loss.AMED_loss'

    c.total_kimg = opts.total_kimg
    c.kimg_per_tick = 1
    c.snapshot_ticks = c.total_kimg
    c.state_dump_ticks = c.total_kimg
    c.update(dataset_name=opts.dataset_name, batch_size=opts.batch, batch_gpu=opts.batch_gpu, gpus=dist.get_world_size(), cudnn_benchmark=opts.bench)
    c.update(guidance_type=opts.guidance_type, guidance_rate=opts.guidance_rate)

    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    schedule_str = 'poly' + str(opts.schedule_rho) if opts.schedule_type == 'polynomial' else opts.schedule_type
    nfe = 2 * (opts.num_steps - 1) - 1 if opts.afs else 2 * (opts.num_steps - 1)
    desc = f'{opts.dataset_name}-{opts.num_steps}-{nfe}-{opts.sampler_stu}-{opts.sampler_tea}-{opts.M}-{schedule_str}-afs' if opts.afs else \
           f'{opts.dataset_name}-{opts.num_steps}-{nfe}-{opts.sampler_stu}-{opts.sampler_tea}-{opts.M}-{schedule_str}'
    if opts.desc:
        desc += f'-{opts.desc}'

    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))] if os.path.isdir(opts.outdir) else []
        prev_run_ids = [int(re.match(r'^\d+', x).group()) for x in prev_run_dirs if re.match(r'^\d+', x)]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0(f'Output directory: {c.run_dir}')
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    training_loop.training_loop(**c)

if __name__ == "__main__":
    main()