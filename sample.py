import os
import pickle
import re
import PIL
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from torchvision.utils import save_image
from torch_utils.download_util import check_file_by_key
import solvers_amed

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]
    def randn(self, size, **kwargs):
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

def parse_int_list(s):
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        ranges.extend(range(int(m.group(1)), int(m.group(2))+1)) if m else [int(p)]
    return ranges

def create_model(dataset_name=None, guidance_type=None, guidance_rate=None, device=None):
    model_path, classifier_path = check_file_by_key(dataset_name)
    dist.print0(f'Loading the pre-trained ADM model from "{model_path}"...')
    if guidance_type == 'cg':  # Classifier guidance
        assert classifier_path is not None
        from models.guided_diffusion.cg_model_loader import load_cg_model
        from models.networks_edm import CGPrecond
        net, classifier = load_cg_model(model_path, classifier_path)
        net = CGPrecond(net, classifier, guidance_rate=guidance_rate).to(device)
    elif guidance_type == 'uncond':  # Unconditional
        with dnnlib.util.open_url(model_path, verbose=(dist.get_rank() == 0)) as f:
            net = pickle.load(f).to(device)  # Assuming unconditional ADM model is a simple pickle load
    else:
        raise ValueError("Guidance type must be 'cg' or 'uncond' for ADM model")
    net.eval()
    return net, 'adm'

@click.command()
@click.option('--predictor_path', help='Path to trained AMED instructor', type=str, required=True)
@click.option('--model_path', help='Network filepath', type=str)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', type=click.IntRange(min=1), default=64)
@click.option('--seeds', help='Random seeds (e.g. 1,2,5-10)', type=parse_int_list, default='0-63')
@click.option('--outdir', help='Where to save the output images', type=str)
@click.option('--grid', help='Whether to make grid', type=bool, default=False)
@click.option('--subdirs', help='Create subdirectory for every 1000 seeds', type=bool, default=True)

def main(predictor_path, max_batch_size, seeds, grid, outdir, subdirs, device=torch.device('cuda')):
    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load AMED predictor
    dist.print0(f'Loading AMED predictor from "{predictor_path}"...')
    with dnnlib.util.open_url(predictor_path, verbose=(dist.get_rank() == 0)) as f:
        AMED_predictor = pickle.load(f)['model'].to(device)

    # Update settings (assuming ADM-specific defaults)
    solver_kwargs = {
        'AMED_predictor': AMED_predictor,
        'solver': AMED_predictor.sampler_stu,
        'num_steps': AMED_predictor.num_steps,
        'guidance_type': AMED_predictor.guidance_type,
        'guidance_rate': AMED_predictor.guidance_rate,
        'afs': AMED_predictor.afs,
        'denoise_to_zero': False,
        'max_order': AMED_predictor.max_order,
        'predict_x0': AMED_predictor.predict_x0,
        'lower_order_final': AMED_predictor.lower_order_final,
        'schedule_type': AMED_predictor.schedule_type,
        'schedule_rho': AMED_predictor.schedule_rho,
        'dataset_name': AMED_predictor.dataset_name
    }

    net, model_source = create_model(solver_kwargs['dataset_name'], solver_kwargs['guidance_type'], solver_kwargs['guidance_rate'], device)
    solver_kwargs['sigma_min'] = 0.002  # Hardcoded for ADM
    solver_kwargs['sigma_max'] = 80.0   # Hardcoded for ADM
    nfe = 2 * (solver_kwargs['num_steps'] - 1) - 1 if solver_kwargs["afs"] else 2 * (solver_kwargs['num_steps'] - 1)
    solver_kwargs['nfe'] = nfe

    if dist.get_rank() == 0:
        torch.distributed.barrier()

    if solver_kwargs['solver'] == 'amed':
        sampler_fn = solvers_amed.amed_sampler
    elif solver_kwargs['solver'] == 'euler':
        sampler_fn = solvers_amed.euler_sampler
    elif solver_kwargs['solver'] == 'dpm':
        sampler_fn = solvers_amed.dpm_2_sampler
    elif solver_kwargs['solver'] == 'ipndm':
        sampler_fn = solvers_amed.ipndm_sampler
    elif solver_kwargs['solver'] == 'dpmpp':
        sampler_fn = solvers_amed.dpm_pp_sampler
    else:
        raise ValueError(f"Unsupported solver: {solver_kwargs['solver']}")

    dist.print0("Solver settings:")
    for key, value in solver_kwargs.items():
        if value is not None and key != 'AMED_predictor':
            dist.print0(f"\t{key}: {value}")

    if outdir is None:
        outdir = os.path.join(f"./samples/{solver_kwargs['dataset_name']}", f"{solver_kwargs['solver']}_nfe{nfe}")
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in rank_batches:
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = rnd.randint(net.label_dim, size=(batch_size,), device=device) if net.label_dim else None

        with torch.no_grad():
            images = sampler_fn(net, latents, class_labels=class_labels, **solver_kwargs)

        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            PIL.Image.fromarray(image_np, 'RGB').save(os.path.join(image_dir, f'{seed:06d}.png'))

    torch.distributed.barrier()
    dist.print0('Done.')

if __name__ == "__main__":
    main()