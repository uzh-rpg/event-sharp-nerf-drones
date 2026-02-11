import torch
import torch.nn as nn

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp as Slerp_scipy

from utils.interpolate import Interp1D, Slerp


class SlerpInterpolator(nn.Module):
    def __init__(self, ts_us, poses_rot, poses_tran, clip_boundary=True, warn_boundary=False, interp_known=False,
                 fix_timestamps=[], tran_kwargs={"kind": "cubic"}, slerp_kwargs={}):
        """
        Class to interpolate at unknown images_mid_tms given known poses

        :param ts_us: np.ndarray of the known camera poses, [N]
        :param poses_rot: np.ndarray of the poses rotations, [N, 3, 3]
        :param poses_trans: np.ndarray of the poses translation vectors [N, 3, 1]
        :param tran_kwargs: a dict of additional argument  to pass to scipy interp1d
            (default: "kind": "cubic", "bounds_error": True)
        :param slerp_kwargs: a dict of additional arguments to pass to scipy Slerp (default: none)
        """
        super().__init__()

        self._warned = 0
        self.clip_boundary = clip_boundary
        self.warn_boundary = warn_boundary

        self.interp_known = interp_known

        if fix_timestamps is not None and len(fix_timestamps) != 0:
            self.register_buffer("fix_timestamps", torch.sort(torch.tensor(fix_timestamps)).values)
            idx = torch.searchsorted(ts_us, self.fix_timestamps)
            self.register_buffer("fix_rot", poses_rot[idx])
            self.register_buffer("fix_tran", poses_tran[idx])
        else:
            self.fix_timestamps, self.fix_rot, self.fix_tran = None, None, None

        self.register_buffer("t", ts_us)
        self.register_buffer("poses_rot", poses_rot)
        self.register_buffer("poses_tran", poses_tran)

        self.max_t, self.min_t = float(max(ts_us)), float(min(ts_us))
        # Setup Rot interpolator
        # self.rot_interpolator = Slerp(ts_us, poses_rot, **slerp_kwargs)
        self.rot_interpolator = Slerp_scipy(ts_us.cpu().numpy(), R.from_matrix(poses_rot.cpu().numpy()))
        # Setup trans interpolator
        self.tran_interpolator = Interp1D(x=ts_us, y=poses_tran, **tran_kwargs)

    def get_param_groups(self):
        return {
            'rotation': [],
            'translation': [],
            'rest': []
        }

    def replace_with_known(self, rots, trans, ts, known_rot, known_trans, known_ts):
        if known_rot.ndim == 2 and known_trans.ndim == 1:
            # If only one rotation and translation is given, assume it is the same for all timestamps
            known_rot = known_rot.unsqueeze(0).expand(known_ts.shape[0], -1, -1).clone()
            known_trans = known_trans.unsqueeze(0).expand(known_ts.shape[0], -1).clone()
        else:
            assert known_rot.shape[0] == known_trans.shape[0] == known_ts.shape[0]

        # Overrides known timestamps' values with known poses
        is_known = torch.isin(ts, known_ts)
        idx = torch.searchsorted(known_ts, ts[is_known])
        assert torch.allclose(known_ts[idx], ts[is_known].to(known_ts.dtype))

        rots[is_known] = known_rot[idx].to(rots.dtype)
        trans[is_known] = known_trans[idx].to(trans.dtype)

        return rots, trans

    @staticmethod
    def to_homo(rots, trans):
        poses = torch.eye(4, device=rots.device, dtype=rots.dtype).unsqueeze(0).expand(rots.shape[0], -1, -1).clone()
        poses[:, :3, :3] = rots
        poses[:, :3, 3] = trans
        return poses

    def interpolate(self, ts):
        """
        Interpolate at unknown images_mid_tms given known poses
        :param ts: np.ndarray of the unknown camera poses, [N]
        :return: rots: np.ndarray of the interpolated rotations, [N, 3, 3],
                 trans: np.ndarray of the interpolated translations, [N, 3]
        """

        if ts.shape[0] == 0:
            return torch.zeros(0, 3, 3, device=ts.device), \
                   torch.zeros(0, 3, device=ts.device)

        if ts.max() > self.max_t or ts.min() < self.min_t:
            if self.warn_boundary and self._warned < 10:
                print(f"Warning: requested images_mid_tms [{ts.min()}, {ts.max()}] is outside of the "
                      f"known images_mid_tms [{self.min_t}, {self.max_t}]")
                self._warned += 1
            if self.clip_boundary:
                ts = torch.clamp(ts, self.min_t, self.max_t)

        if ts.ndim == 2:
            ts = ts.squeeze(-1)
            assert ts.ndim == 1, "ts must be either a 1D or 2D tensor with unitary last dimension"

        # Query rot interpolator
        rots = torch.from_numpy(self.rot_interpolator(ts.detach().cpu().numpy()).as_matrix()).to(ts.device)
        # Query trans interpolator
        trans = self.tran_interpolator(ts)

        if self.interp_known is False:
            rots, trans = self.replace_with_known(rots, trans, ts, self.poses_rot, self.poses_tran, self.t)

        return rots, trans

    def forward(self, ts, *args, **kwargs):
        if ts.ndim == 2:
            ts = ts.squeeze(-1)
            assert ts.ndim == 1, "ts must be either a 1D or 2D tensor with unitary last dimension"

        rots, trans = self.interpolate(ts)

        if self.fix_timestamps is not None:
            rots, trans = self.replace_with_known(rots, trans, ts, self.fix_rot, self.fix_tran, self.fix_timestamps)

        poses = self.to_homo(rots, trans).float()
        return poses