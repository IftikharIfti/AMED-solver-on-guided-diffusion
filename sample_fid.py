import os
import pickle
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils.download_util import check_file_by_key
import solvers_amed
from models.guided_diffusion.cg_model_loader import load_cg_model
from models.networks_edm import CGPrecond
import PIL.Image

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]
    def randn(self, size, **kwargs):
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

def create_model(dataset_name, guidance_type, guidance_rate, device):
    model_path, classifier_path = check_file_by_key(dataset_name)
    dist.print0(f'Loading the pre-trained ADM model from "{model_path}"...')
    if guidance_type == 'cg':
        assert classifier_path is not None
        net, classifier = load_cg_model(model_path, classifier_path)
        net = CGPrecond(net, classifier, guidance_rate=guidance_rate).to(device)
    elif guidance_type == 'uncond':
        with dnnlib.util.open_url(model_path, verbose=(dist.get_rank() == 0)) as f:
            net = torch.load(f).to(device)
    else:
        raise ValueError("Guidance type must be 'cg' or 'uncond' for ADM model")
    net.eval()
    return net, 'adm'

@click.command()
@click.option('--predictor_path', help='Path to trained AMED predictor', type=str, required=True)
@click.option('--dataset_name', help='Dataset name (e.g., imagenet256)', type=str, default='imagenet256')
@click.option('--guidance_type', help='Guidance type', type=click.Choice(['cg', 'uncond']), default='uncond')
@click.option('--guidance_rate', help='Guidance rate for cg', type=float, default=1.0)
@click.option('--batch_size', help='Batch size for sampling', type=click.IntRange(min=1), default=64)
@click.option('--num_samples', help='Total number of samples to generate', type=click.IntRange(min=1), default=50000)
@click.option('--outdir', help='Where to save the output images', type=str, default='./fid_samples')
@click.option('--seed', help='Random seed', type=int, default=0)

def main(predictor_path, dataset_name, guidance_type, guidance_rate, batch_size, num_samples, outdir, seed, device=torch.device('cuda')):
    dist.init()
    os.makedirs(outdir, exist_ok=True)

    dist.print0(f'Loading AMED predictor from "{predictor_path}"...')
    with dnnlib.util.open_url(predictor_path, verbose=(dist.get_rank() == 0)) as f:
        AMED_predictor = pickle.load(f)['model'].to(device)

    solver_kwargs = {
        'AMED_predictor': AMED_predictor,
        'solver': AMED_predictor.sampler_stu,
        'num_steps': AMED_predictor.num_steps,
        'guidance_type': guidance_type,
        'guidance_rate': guidance_rate,
        'afs': AMED_predictor.afs,
        'denoise_to_zero': False,
        'max_order': AMED_predictor.max_order,
        'predict_x0': AMED_predictor.predict_x0,
        'lower_order_final': AMED_predictor.lower_order_final,
        'schedule_type': AMED_predictor.schedule_type,
        'schedule_rho': AMED_predictor.schedule_rho,
        'sigma_min': 0.002,
        'sigma_max': 80.0,
    }
    nfe = 2 * (solver_kwargs['num_steps'] - 1) - 1 if solver_kwargs["afs"] else 2 * (solver_kwargs['num_steps'] - 1)
    solver_kwargs['nfe'] = nfe

    net, _ = create_model(dataset_name, guidance_type, guidance_rate, device)

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
        raise ValueError(f"Unsupported solver: {solver_kwargs['solver']} for FID evaluation")

    dist.print0(f"Generating {num_samples} samples with {solver_kwargs['solver']} (NFE: {nfe})...")
    samples_generated = 0
    torch.manual_seed(seed)

    with torch.no_grad():
        while samples_generated < num_samples:
            batch_seeds = range(seed + samples_generated, seed + samples_generated + batch_size)
            batch_size_actual = min(batch_size, num_samples - samples_generated)
            rnd = StackedRandomGenerator(device, batch_seeds)
            latents = rnd.randn([batch_size_actual, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            class_labels = rnd.randint(net.label_dim, size=(batch_size_actual,), device=device) if net.label_dim else None

            images = sampler_fn(net, latents, class_labels=class_labels, **solver_kwargs)
            images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            
            for seed_idx, image_np in zip(batch_seeds, images_np):
                image_path = os.path.join(outdir, f'{seed_idx:06d}.png')
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)
            samples_generated += batch_size_actual
            dist.print0(f"Generated {samples_generated}/{num_samples} samples...")

    dist.print0(f"Saved {num_samples} samples to {outdir}")

if __name__ == "__main__":
    main()