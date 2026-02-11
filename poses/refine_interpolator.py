import math
import torch
from tqdm import tqdm

from poses.slerp_interpolator import SlerpInterpolator


class RefineInterpolator(SlerpInterpolator):
    def __init__(self, refiner, embedder, min_time, max_time, embed_schedule=None, init_identity=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embed = embedder
        self.refine = refiner

        self.min_time = min_time
        self.max_time = max_time
        self.embed_schedule = embed_schedule

        self.embed_freqs = embedder.n_freqs
        self.embed_periodic = embedder.n_periodic
        self.embed_include_input = embedder.include_input

        self._bounds_warn_count = 0

        if init_identity:
            self.init_identity()

    def get_param_groups(self):
        return self.refine.get_param_groups()
            
    def init_identity(self, batch_size=32, iterations=1000):
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)
        pbar = tqdm(range(iterations), desc='Pose interpolator init identity')
        gen = torch.Generator(device="cuda").manual_seed(42)
        target = torch.eye(4, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1, -1)[:, :3, :4]
        progress = 0.0 if self.embed_schedule is not None else None

        # Trains the network for 1000 iterations
        for _ in pbar:
            ts = torch.rand(size=(batch_size, 1), dtype=torch.float32, generator=gen)
            ts = self.min_time + ts * (self.max_time - self.min_time)
            ts_norm, ts_emb = self.embed_ts(ts, progress)
            refine_tr = self.refine(ts_norm, ts_emb)

            loss = torch.nn.functional.mse_loss(refine_tr, target)
            pbar.set_postfix({'loss': loss.item()})

            optim.zero_grad()
            loss.backward()
            optim.step()
        optim.zero_grad()

    def embed_ts(self, time, progress=None):
        if torch.any(time < self.min_time) or torch.any(time > self.max_time):
            if self.warn_boundary and self._bounds_warn_count < 10:
                self._bounds_warn_count += 1
                failed = torch.unique(time[(time < self.min_time) | (time > self.max_time)]).detach().cpu().numpy().tolist()
                print(f"Warning: time ({failed}) out of range [{self.min_time}, {self.max_time}]")

        time = 2 * (time - self.min_time) / (self.max_time - self.min_time) - 1
        time = time.reshape(-1, 1).to(torch.float32)

        time_enc = self.embed(time)

        if self.embed_schedule is not None:
            shape = time_enc.shape
            device = time_enc.device

            start, end = self.embed_schedule
            alpha = (progress - start) / (end - start) * self.embed_freqs
            k = torch.arange(self.embed_freqs, dtype=torch.float32, device=device)
            weight = (1 - (alpha - k).clamp_(min=0, max=1).mul_(math.pi).cos_()) / 2  # L
            # Apply weight on all periodic functions (sin, cos)
            weight = torch.stack([weight] * self.embed_periodic, dim=-1).reshape(-1)  # L,2 -> L*2

            if self.embed_include_input:
                weight = torch.cat([torch.ones(1, device=device), weight], dim=0)  # 1+L*2

            time_enc = (time_enc * weight.unsqueeze(0)).view(*shape)

        return time, time_enc

    def forward(self, ts, progress=None, refine=True, *args, **kwargs):
        if not refine:  # Just return the slerp interpolation
            return super().forward(ts, progress, *args, **kwargs)

        if ts.shape[0] == 0:
            return torch.zeros(0, 4, 4, device=ts.device)

        if ts.ndim == 2:
            ts = ts.squeeze(-1)
            assert ts.ndim == 1, "ts must be either a 1D or 2D tensor with unitary last dimension"

        rots, trans = self.interpolate(ts)
        ts_norm, ts_emb = self.embed_ts(ts, progress)
        refine_tr = self.refine(ts_norm, ts_emb)
        refine_rots, refine_trans = refine_tr[:, :3, :3], refine_tr[:, :3, 3]

        if self.fix_timestamps is not None:
            # Replace fixed timestamps with identity refinement transformation
            id_rot = torch.eye(3, device=rots.device, dtype=rots.dtype)
            id_tran = torch.zeros([3], device=rots.device, dtype=rots.dtype)
            refine_rots, refine_trans = self.replace_with_known(
                refine_rots, refine_trans, ts, id_rot, id_tran, self.fix_timestamps)

        poses = self.to_homo(rots, trans).float()
        refine_poses = self.to_homo(refine_rots, refine_trans).float()
        poses = torch.bmm(poses, refine_poses)
        return poses
