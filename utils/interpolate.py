import roma
import torch
from torch import nn
from scipy.interpolate import interp1d


class Slerp(nn.Module):
    """Spherical Linear Interpolation of Rotations. This class reimplements scipy.spatial.transform.Slerp.
    """

    def __init__(self, times, rotations):
        """
        :param times: [N] torch.tensor, the timestamps of the rotations
        :param rotations: [N, 3, 3] torch.tensor, the rotations
        """
        super().__init__()

        if rotations.shape[0] < 2:
            raise ValueError("`rotations` must be a sequence of at least 2 rotations.")
        if times.ndim != 1:
            raise ValueError("Expected times to be specified in a 1 "
                             "dimensional array, got {} "
                             "dimensions.".format(times.ndim))

        if times.shape[0] != rotations.shape[0]:
            raise ValueError("Expected number of rotations to be equal to "
                             "number of timestamps given, got {} rotations "
                             "and {} timestamps.".format(
                rotations.shape[0], times.shape[0]))

        uniquats = roma.rotmat_to_unitquat(rotations)

        self.register_buffer("times", times)
        self.register_buffer("uniquats", uniquats[:-1])
        self.register_buffer("rotvecs", roma.unitquat_to_rotvec(
            roma.quat_product(roma.quat_inverse(uniquats[:-1]),  uniquats[1:])
        ))

    def forward(self, x):
        single_time = x.ndim == 0
        if single_time:
            x = x.reshape(1, 1)

        # side = 'left' (default) excludes t_min.
        ind = torch.searchsorted(self.times, x) - 1
        # Include t_min. Without this step, index for t_min equals -1
        ind[x == self.times[0]] = 0

        if torch.any(torch.logical_or(ind < 0, ind > self.uniquats.shape[0] - 1)):
            raise ValueError("Interpolation times must be within the range "
                             "[{}, {}], both inclusive.".format(
                self.times[0], self.times[-1]))

        alpha = (x - self.times[ind]) / (self.times[ind+1] - self.times[ind])
        alpha = alpha.to(self.uniquats.dtype)

        result =  roma.unitquat_to_rotmat(roma.quat_product(
            self.uniquats[ind], roma.rotvec_to_unitquat(self.rotvecs[ind] * alpha[:, None])))

        if single_time:
            result = result[0]

        return result


class Interp1D(nn.Module):
    def __init__(self, x, y, kind="cubic"):
        """
        :param x: [N] torch.tensor, the timestamps of the rotations
        :param y: [N, D] torch.tensor, the rotations
        """
        super().__init__()

        if x.ndim != 1:
            raise ValueError("Expected times to be specified in a 1 "
                             "dimensional array, got {} "
                             "dimensions.".format(x.ndim))

        if x.shape[0] != y.shape[0]:
            raise ValueError("Expected number of rotations to be equal to "
                             "number of timestamps given, got {} rotations "
                             "and {} timestamps.".format(
                y.shape[0], x.shape[0]))

        self.kind = kind
        self.register_buffer("x", x)
        self.register_buffer("y", y)

    def interp_linear(self, x_new):
        idx = torch.searchsorted(self.x, x_new)
        idx = torch.clamp(idx, 1, len(self.x) - 1).long()

        lo = idx - 1
        hi = idx

        x_lo = self.x[lo]
        x_hi = self.x[hi]
        y_lo = self.y[lo]
        y_hi = self.y[hi]

        slope = (y_hi - y_lo) / (x_hi - x_lo)[:, None]
        y_new = slope * (x_new - x_lo)[:, None] + y_lo
        return y_new

    # def interp_quadratic(self, x_new):
    #     idx = torch.searchsorted(self.x, x_new)
    #     idx = torch.clamp(idx, 1, len(self.x) - 2).long()
    #
    #     x0, x1, x2 = self.x[idx - 1:idx + 2]
    #     y0, y1, y2 = self.y[idx - 1:idx + 2]
    #
    #     t = (x_new - x1) / (x2 - x1)
    #     t2 = t * t
    #
    #     a = 0.5 * (y2 - 2 * y1 + y0)
    #     b = 0.5 * (y2 - y0)
    #     c = y1
    #
    #     return a * t2 + b * t + c
    #
    # def interp_cubic(self, x_new):
    #     idx = torch.searchsorted(self.x, x_new)
    #     idx = torch.clamp(idx, 1, len(self.x) - 2).long()
    #
    #     x0, x1, x2, x3 = self.x[idx - 1:idx + 3]
    #     y0, y1, y2, y3 = self.y[idx - 1:idx + 3]
    #
    #     t = (x_new - x1) / (x2 - x1)
    #     t2 = t * t
    #     t3 = t2 * t
    #
    #     a0 = y3 - y2 - y0 + y1
    #     a1 = y0 - y1 - a0
    #     a2 = y2 - y0
    #     a3 = y1
    #
    #     return a0 * t3 + a1 * t2 + a2 * t + a3

    def forward(self, x):
        if self.kind == "linear":
            return self.interp_linear(x)
        elif self.kind in ["quadratic", "cubic"]:
            interp = interp1d(x=self.x.cpu().numpy(), y=self.y.cpu().numpy(), axis=0, kind=self.kind, bounds_error=True)
            return torch.from_numpy(interp(x.detach().cpu().numpy())).to(x.device)
        else:
            raise ValueError(f"Unknown kind {self.kind}")
