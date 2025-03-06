import torch
from solver_utils import get_schedule

def init_hook(net, class_labels=None):
    unet_enc_out = []
    def hook_fn(module, input, output):
        unet_enc_out.append(output.detach())
    if net.img_resolution == 256:  # Assuming ADM-specific resolution
        hook = net.model.middle_block.register_forward_hook(hook_fn)
    else:
        module_name = '8x8_block2' if class_labels is not None else '8x8_block3'
        hook = net.model.enc[module_name].register_forward_hook(hook_fn)
    return unet_enc_out, hook

def get_amed_prediction(AMED_predictor, t_cur, t_next, net, unet_enc_out, use_afs, batch_size):
    unet_enc = torch.mean(unet_enc_out[-1], dim=1) if not use_afs else torch.zeros((batch_size, 8, 8), device=t_cur.device)
    output = AMED_predictor(unet_enc, t_cur, t_next)
    output_list = [*output]
    
    if len(output_list) == 2:
        r, scale_dir = output_list
        r = r.reshape(-1, 1, 1, 1)
        scale_dir = scale_dir.reshape(-1, 1, 1, 1)
        scale_time = torch.ones_like(scale_dir)
    elif len(output_list) == 3:
        r, scale_dir, scale_time = output_list
        r = r.reshape(-1, 1, 1, 1)
        scale_dir = scale_dir.reshape(-1, 1, 1, 1)
        scale_time = scale_time.reshape(-1, 1, 1, 1)
    else:
        r = output.reshape(-1, 1, 1, 1)
        scale_dir = torch.ones_like(r)
        scale_time = torch.ones_like(r)
    return r, scale_dir, scale_time

def get_denoised(net, x, t, class_labels=None):
    return net(x, t, class_labels=class_labels)

def amed_sampler(net, latents, class_labels=None, num_steps=None, sigma_min=0.002, sigma_max=80, 
                 schedule_type='polynomial', schedule_rho=7, afs=False, denoise_to_zero=False, 
                 return_inters=False, AMED_predictor=None, step_idx=None, train=False, **kwargs):
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho)
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        unet_enc_out, hook = init_hook(net, class_labels)
        use_afs = afs and (((not train) and i == 0) or (train and step_idx == 0))
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels)
            d_cur = (x_cur - denoised) / t_cur
        hook.remove()
        t_cur = t_cur.reshape(-1, 1, 1, 1)
        t_next = t_next.reshape(-1, 1, 1, 1)
        r, scale_dir, scale_time = get_amed_prediction(AMED_predictor, t_cur, t_next, net, unet_enc_out, use_afs, latents.shape[0])
        t_mid = (t_next ** r) * (t_cur ** (1 - r))
        x_next = x_cur + (t_mid - t_cur) * d_cur
        denoised = get_denoised(net, x_next, scale_time * t_mid, class_labels=class_labels)
        d_mid = (x_next - denoised) / t_mid
        x_next = x_cur + scale_dir * (t_next - t_cur) * d_mid
        if return_inters:
            inters.append(x_next.unsqueeze(0))
    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels)
        if return_inters:
            inters.append(x_next.unsqueeze(0))
    return torch.cat(inters, dim=0).to(latents.device) if return_inters else x_next

def euler_sampler(net, latents, class_labels=None, num_steps=None, sigma_min=0.002, sigma_max=80, 
                  schedule_type='polynomial', schedule_rho=7, afs=False, denoise_to_zero=False, 
                  return_inters=False, AMED_predictor=None, step_idx=None, train=False, **kwargs):
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho)
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        unet_enc_out, hook = init_hook(net, class_labels)
        use_afs = afs and (((not train) and i == 0) or (train and step_idx == 0))
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels)
            d_cur = (x_cur - denoised) / t_cur
        hook.remove()
        t_cur = t_cur.reshape(-1, 1, 1, 1)
        t_next = t_next.reshape(-1, 1, 1, 1)
        r, scale_dir, scale_time = get_amed_prediction(AMED_predictor, t_cur, t_next, net, unet_enc_out, use_afs, latents.shape[0])
        t_mid = (t_next ** r) * (t_cur ** (1 - r))
        x_next = x_cur + (t_mid - t_cur) * d_cur
        denoised = get_denoised(net, x_next, scale_time * t_mid, class_labels=class_labels)
        d_mid = (x_next - denoised) / t_mid
        x_next = x_next + scale_dir * (t_next - t_mid) * d_mid
        if return_inters:
            inters.append(x_next.unsqueeze(0))
    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels)
        if return_inters:
            inters.append(x_next.unsqueeze(0))
    return torch.cat(inters, dim=0).to(latents.device) if return_inters else x_next

def ipndm_sampler(net, latents, class_labels=None, num_steps=None, sigma_min=0.002, sigma_max=80, 
                  schedule_type='polynomial', schedule_rho=7, afs=False, denoise_to_zero=False, 
                  return_inters=False, AMED_predictor=None, train=False, max_order=4, buffer_model=[], **kwargs):
    assert max_order >= 1 and max_order <= 4
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho)
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    buffer_model = buffer_model if train else []
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        unet_enc_out, hook = init_hook(net, class_labels)
        use_afs = afs and len(buffer_model) == 0
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels)
            d_cur = (x_cur - denoised) / t_cur
        
        hook.remove()
        t_cur = t_cur.reshape(-1, 1, 1, 1)
        t_next = t_next.reshape(-1, 1, 1, 1)
        r, scale_dir, scale_time = get_amed_prediction(AMED_predictor, t_cur, t_next, net, unet_enc_out, use_afs, latents.shape[0])
        t_mid = (t_next ** r) * (t_cur ** (1 - r))
        
        order = min(max_order, len(buffer_model) + 1)
        if order == 1:
            x_next = x_cur + (t_mid - t_cur) * d_cur
        elif order == 2:
            x_next = x_cur + (t_mid - t_cur) * (3 * d_cur - buffer_model[-1]) / 2
        elif order == 3:
            x_next = x_cur + (t_mid - t_cur) * (23 * d_cur - 16 * buffer_model[-1] + 5 * buffer_model[-2]) / 12
        elif order == 4:
            x_next = x_cur + (t_mid - t_cur) * (55 * d_cur - 59 * buffer_model[-1] + 37 * buffer_model[-2] - 9 * buffer_model[-3]) / 24
        
        denoised = get_denoised(net, x_next, scale_time * t_mid, class_labels=class_labels)
        d_mid = (x_next - denoised) / t_mid
        order = min(max_order, len(buffer_model) + 1)
        if order == 1:
            x_next = x_next + scale_dir * (t_next - t_mid) * d_mid
        elif order == 2:
            x_next = x_next + scale_dir * (t_next - t_mid) * (3 * d_mid - buffer_model[-1]) / 2
        elif order == 3:
            x_next = x_next + scale_dir * (t_next - t_mid) * (23 * d_mid - 16 * buffer_model[-1] + 5 * buffer_model[-2]) / 12
        elif order == 4:
            x_next = x_next + scale_dir * (t_next - t_mid) * (55 * d_mid - 59 * buffer_model[-1] + 37 * buffer_model[-2] - 9 * buffer_model[-3]) / 24
        
        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k + 1]
            buffer_model[-1] = d_cur.detach()
        else:
            buffer_model.append(d_cur.detach())
        
        if return_inters:
            inters.append(x_next.unsqueeze(0))
    
    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels)
        if return_inters:
            inters.append(x_next.unsqueeze(0))
    
    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    if train:
        return x_next, buffer_model, [], r, scale_dir, scale_time
    return x_next

def dpm_2_sampler(net, latents, class_labels=None, num_steps=None, sigma_min=0.002, sigma_max=80, 
                  schedule_type='polynomial', schedule_rho=7, afs=False, denoise_to_zero=False, 
                  return_inters=False, AMED_predictor=None, step_idx=None, train=False, **kwargs):
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho)
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        unet_enc_out, hook = init_hook(net, class_labels)
        use_afs = afs and (((not train) and i == 0) or (train and step_idx == 0))
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels)
            d_cur = (x_cur - denoised) / t_cur
        
        hook.remove()
        t_cur = t_cur.reshape(-1, 1, 1, 1)
        t_next = t_next.reshape(-1, 1, 1, 1)
        r, scale_dir, scale_time = get_amed_prediction(AMED_predictor, t_cur, t_next, net, unet_enc_out, use_afs, latents.shape[0])
        t_mid = (t_next ** r) * (t_cur ** (1 - r))
        x_next = x_cur + (t_mid - t_cur) * d_cur
        
        denoised = get_denoised(net, x_next, scale_time * t_mid, class_labels=class_labels)
        d_mid = (x_next - denoised) / t_mid
        x_next = x_cur + scale_dir * (t_next - t_cur) * ((1 / (2 * r)) * d_mid + (1 - 1 / (2 * r)) * d_cur)
        
        if return_inters:
            inters.append(x_next.unsqueeze(0))
    
    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels)
        if return_inters:
            inters.append(x_next.unsqueeze(0))
    
    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    if train:
        return x_next, [], [], r, scale_dir, scale_time
    return x_next

def dpm_pp_sampler(net, latents, class_labels=None, num_steps=None, sigma_min=0.002, sigma_max=80, 
                   schedule_type='polynomial', schedule_rho=7, afs=False, denoise_to_zero=False, 
                   return_inters=False, AMED_predictor=None, step_idx=None, train=False, 
                   buffer_model=[], buffer_t=[], max_order=3, predict_x0=True, lower_order_final=True, **kwargs):
    assert max_order >= 1 and max_order <= 3
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho)
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    buffer_model = buffer_model if train else []
    buffer_t = buffer_t if train else []
    num_steps_adjusted = 2 * num_steps - 1 if not train else num_steps
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        step_cur = (2 * i + 1) if not train else (2 * step_idx + 1 if train and step_idx is not None else i + 1)
        unet_enc_out, hook = init_hook(net, class_labels)
        use_afs = afs and len(buffer_model) == 0
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
            denoised = x_cur - t_cur * d_cur
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels)
            d_cur = (x_cur - denoised) / t_cur
        
        buffer_model.append(denoised if predict_x0 else d_cur)
        hook.remove()
        t_cur = t_cur.reshape(-1, 1, 1, 1)
        t_next = t_next.reshape(-1, 1, 1, 1)
        r, scale_dir, scale_time = get_amed_prediction(AMED_predictor, t_cur, t_next, net, unet_enc_out, use_afs, latents.shape[0])
        t_mid = (t_next ** r) * (t_cur ** (1 - r))
        buffer_t.append(t_cur)
        
        order = min(max_order, step_cur) if not lower_order_final else (step_cur if step_cur < max_order else min(max_order, num_steps_adjusted - step_cur))
        x_next = dpm_pp_update(x_cur, buffer_model, buffer_t, t_mid, order, predict_x0=predict_x0)
        
        denoised = get_denoised(net, x_next, scale_time * t_mid, class_labels=class_labels)
        d_mid = (x_next - denoised) / t_mid if not predict_x0 else denoised
        buffer_model.append(d_mid)
        buffer_t.append(t_mid)
        
        step_cur += 1
        order = min(max_order, step_cur) if not lower_order_final else (step_cur if step_cur < max_order else min(max_order, num_steps_adjusted - step_cur))
        x_next = dpm_pp_update(x_next, buffer_model, buffer_t, t_next, order, predict_x0=predict_x0, scale=scale_dir)
        
        if len(buffer_model) >= 3:
            buffer_model = [a.detach() for a in buffer_model[-3:]]
            buffer_t = [a.detach() for a in buffer_t[-3:]]
        else:
            buffer_model = [a.detach() for a in buffer_model]
            buffer_t = [a.detach() for a in buffer_t]
        
        if return_inters:
            inters.append(x_next.unsqueeze(0))
    
    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels)
        if return_inters:
            inters.append(x_next.unsqueeze(0))
    
    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    if train:
        return x_next, buffer_model, buffer_t, r, scale_dir, scale_time
    return x_next

def dpm_pp_update(x, model_prev_list, t_prev_list, t, order, predict_x0=True, scale=1):
    if order == 1:
        return dpm_solver_first_update(x, t_prev_list[-1], t, model_prev_list[-1], predict_x0, scale)
    elif order == 2:
        return multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, predict_x0, scale)
    elif order == 3:
        return multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, predict_x0, scale)
    else:
        raise ValueError("Solver order must be 1, 2, or 3")

def dpm_solver_first_update(x, s, t, model_s, predict_x0=True, scale=1):
    s, t = s.reshape(-1, 1, 1, 1), t.reshape(-1, 1, 1, 1)
    lambda_s, lambda_t = -s.log(), -t.log()
    h = lambda_t - lambda_s
    phi_1 = torch.expm1(-h) if predict_x0 else torch.expm1(h)
    return (t / s) * x - scale * phi_1 * model_s if predict_x0 else x - scale * t * phi_1 * model_s

def multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, predict_x0=True, scale=1):
    t = t.reshape(-1, 1, 1, 1)
    model_prev_0, model_prev_1 = model_prev_list[-1], model_prev_list[-2]
    t_prev_0, t_prev_1 = t_prev_list[-1].reshape(-1, 1, 1, 1), t_prev_list[-2].reshape(-1, 1, 1, 1)
    lambda_prev_0, lambda_prev_1, lambda_t = -t_prev_0.log(), -t_prev_1.log(), -t.log()
    h_0, h = lambda_prev_0 - lambda_prev_1, lambda_t - lambda_prev_0
    r0 = h_0 / h
    D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
    phi_1 = torch.expm1(-h) if predict_x0 else torch.expm1(h)
    return (t / t_prev_0) * x - scale * (phi_1 * model_prev_0 + 0.5 * phi_1 * D1_0) if predict_x0 else x - scale * (t * phi_1 * model_prev_0 + 0.5 * t * phi_1 * D1_0)

def multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, predict_x0=True, scale=1):
    t = t.reshape(-1, 1, 1, 1)
    model_prev_0, model_prev_1, model_prev_2 = model_prev_list[-1], model_prev_list[-2], model_prev_list[-3]
    t_prev_0, t_prev_1, t_prev_2 = t_prev_list[-1].reshape(-1, 1, 1, 1), t_prev_list[-2].reshape(-1, 1, 1, 1), t_prev_list[-3].reshape(-1, 1, 1, 1)
    lambda_prev_0, lambda_prev_1, lambda_prev_2, lambda_t = -t_prev_0.log(), -t_prev_1.log(), -t_prev_2.log(), -t.log()
    h_0, h_1, h = lambda_prev_0 - lambda_prev_1, lambda_prev_1 - lambda_prev_2, lambda_t - lambda_prev_0
    r0, r1 = h_0 / h, h_1 / h
    D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
    D1_1 = (1. / r1) * (model_prev_1 - model_prev_2)
    D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
    D2 = (1. / (r0 + r1)) * (D1_0 - D1_1)
    phi_1 = torch.expm1(-h) if predict_x0 else torch.expm1(h)
    phi_2 = phi_1 / h + 1. if predict_x0 else phi_1 / h - 1.
    phi_3 = phi_2 / h - 0.5
    return (t / t_prev_0) * x - scale * (phi_1 * model_prev_0 - phi_2 * D1 + phi_3 * D2) if predict_x0 else x - scale * (t * phi_1 * model_prev_0 + t * phi_2 * D1 + t * phi_3 * D2)