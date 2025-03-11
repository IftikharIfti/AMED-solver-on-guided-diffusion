import torch
from torch_utils import persistence
from torch_utils import distributed as dist
import solvers_amed
from solver_utils import get_schedule

def get_solver_fn(solver_name):
    if solver_name == 'amed':
        return solvers_amed.amed_sampler
    elif solver_name == 'euler':
        return solvers_amed.euler_sampler
    elif solver_name == 'heun':
        return solvers_amed.heun_sampler
    elif solver_name == 'dpmpp':
        solver_fn = solvers_amed.dpm_pp_sampler
    else:
        raise ValueError(f"Unsupported solver: {solver_name}")
    return solver_fn

@persistence.persistent_class
class AMED_loss:
    def __init__(
        self,
        num_steps=4,
        sampler_stu='amed',
        sampler_tea='heun',
        schedule_type='polynomial',
        schedule_rho=7,
        dataset_name= None,
        img_resolution= None,
        M= None,
        afs=True,
        max_order=3,
        sigma_min=0.002,
        sigma_max=80.0,
        predict_x0=True,
        lower_order_final=True,
    ):
        self.num_steps = num_steps
        self.solver_stu = get_solver_fn(sampler_stu)
        self.solver_tea = get_solver_fn(sampler_tea)
        self.schedule_type = schedule_type
        self.schedule_rho = schedule_rho
        self.dataset_name = dataset_name
        self.img_resolution = img_resolution
        self.M = M
        self.afs = afs
        self.max_order = max_order
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.predict_x0 = predict_x0
        self.lower_order_final = lower_order_final
        self.t_steps = None
        self.buffer_model = []
        self.buffer_t = []

    def __call__(self, AMED_predictor, net, tensor_in, labels=None, step_idx=None, teacher_out=None):
        step_idx = torch.tensor([step_idx]).reshape(1,)
        t_cur = self.t_steps[step_idx].to(tensor_in.device)
        t_next = self.t_steps[step_idx + 1].to(tensor_in.device)
        if step_idx == 0:
            self.buffer_model = []
            self.buffer_t = []

        student_out, buffer_model, buffer_t, r, scale_dir, scale_time = self.solver_stu(
            net, tensor_in / t_cur, class_labels=labels,
            num_steps=2, sigma_min=t_next, sigma_max=t_cur, schedule_type=self.schedule_type, schedule_rho=self.schedule_rho,
            afs=self.afs, denoise_to_zero=False, return_inters=False, AMED_predictor=AMED_predictor, step_idx=step_idx,
            train=True, predict_x0=self.predict_x0, lower_order_final=self.lower_order_final, max_order=self.max_order,
            buffer_model=self.buffer_model, buffer_t=self.buffer_t,
        )
        self.buffer_model = buffer_model
        self.buffer_t = buffer_t

        loss = (student_out - teacher_out) ** 2
        dist.print0(f"Step: {step_idx.item()} | Loss: {torch.mean(torch.norm(loss, p=2, dim=(1, 2, 3))).item():8.4f}")
        return loss, student_out.detach()

    def get_teacher_traj(self, net, tensor_in, labels=None):
        if self.t_steps is None:
            self.t_steps = get_schedule(self.num_steps, self.sigma_min, self.sigma_max, schedule_type=self.schedule_type, schedule_rho=self.schedule_rho, device=tensor_in.device)
        num_steps_teacher = 2 * (self.num_steps - 1) + 1  # Simplified for FID
        tea_slice = [i * 2 for i in range(1, self.num_steps)]

        teacher_traj = self.solver_tea(
            net, tensor_in / self.t_steps[0], class_labels=labels,
            num_steps=num_steps_teacher, sigma_min=self.sigma_min, sigma_max=self.sigma_max,
            schedule_type=self.schedule_type, schedule_rho=self.schedule_rho, afs=False,
            denoise_to_zero=False, return_inters=True,
        )
        return teacher_traj[tea_slice]