import torch
from torch import nn


class TestTimePoseRefiner(nn.Module):
    def __init__(self, test_poses, mode="independent_rel"):
        super().__init__()
        # Save test poses
        self.register_buffer("test_poses_SE3", self.to_homo(test_poses))

        # shared: the transformation is a single learned matrix, shared among all test poses, which transforms all poses
        # independent: N se3 params are initialized with the test poses and directly optimized
        # both: a shared transformation is first applied, followed by N other independent transformations, which
        #   multiply the original poses (both initialized as the identity)
        self.mode = mode
        if mode in ["shared", "both"]:
            self.shared_params = self.init_shared_params(torch.eye(4))

        if mode in ["both", "independent_rel"]:
            self.independent_params = self.init_independent_params(
                torch.eye(4)[None].repeat(test_poses.shape[0], 1, 1)
            )
        elif mode == "independent_abs":
            self.independent_params = self.init_independent_params(
                test_poses
            )

        if mode in ["both", "independent_rel", "independent_abs"]:
            self.register_buffer("refined_best_independent", self.independent_params.detach().clone())
            self.register_buffer("refined_best_scores", torch.zeros(self.refined_best_independent.shape[0])-1)

    @staticmethod
    def fix_param(param, fix):
        if isinstance(param, nn.Parameter):
            param.requires_grad = not fix
        else:
            param.weight.requires_grad = not fix

    def is_fixed(self, param):
        if hasattr(self, param):
            param = getattr(self, param)
            if isinstance(param, nn.Parameter):
                return not param.requires_grad
            else:
                return not param.weight.requires_grad
        return True  # If the param does not exist, it is fixed

    def has_param(self, type):
        if "independent" in type:
            return hasattr(self, "independent_params")
        elif "shared" in type:
            return hasattr(self, "shared_params")
        return False

    def fix_independent(self, fix):
        if hasattr(self, "independent_params"):
            self.fix_param(self.independent_params, fix)

    def fix_shared(self, fix):
        if hasattr(self, "shared_params"):
            self.fix_param(self.shared_params, fix)

    @property
    def can_update_best_independent(self):
        return self.is_fixed("shared_params") and self.has_param("independent")

    def update_best_independent(self, scores):
        # in "both" mode, if shared are not fixed, independent might not make sense
        if self.can_update_best_independent:
            scores = torch.tensor(scores).float()
            is_better = scores > self.refined_best_scores
            self.refined_best_scores[is_better] = scores[is_better]
            self.refined_best_independent[is_better] = self.independent_params[is_better].detach().clone()

    def set_best_independent(self):
        if self.has_param("independent"):
            saved_best = self.refined_best_scores != -1
            self.independent_params.data[saved_best] = self.refined_best_independent[saved_best].clone()

    def init_shared_params(self, init_SE3):
        return nn.Parameter(self.SE3_to_se3(init_SE3[:3, :4]), requires_grad=True)

    def init_independent_params(self, init_SE3s):
        return nn.Parameter(self.SE3_to_se3_N(init_SE3s[:, :3, :4]), requires_grad=True)
        #return nn.Embedding.from_pretrained(self.SE3_to_se3_N(init_SE3s[:, :3, :4]), freeze=False)

    @staticmethod
    def to_homo(poses):
        nobatch = poses.ndim == 2
        if nobatch:
            poses = poses[None]
        assert poses.ndim == 3

        if poses.shape[-2:] == (4,4):
            return poses

        homo_poses = (torch.eye(4, device=poses.device, dtype=poses.dtype)
                      .unsqueeze(0).repeat(poses.shape[0], 1, 1).clone())
        homo_poses[:, :3, :4] = poses

        if nobatch:
            homo_poses = homo_poses[0]
        return homo_poses

    @classmethod
    def se3_to_SE3(cls, wu):  # [...,3]
        w, u = wu.split([3, 3], dim=-1)
        wx = cls.skew_symmetric(w)  # wx=[0 -w(2) w(1);w(2) 0 -w(0);-w(1) w(0) 0]
        theta = w.norm(dim=-1)[..., None, None]  # theta=sqrt(w'*w)
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = cls.taylor_A(theta)
        B = cls.taylor_B(theta)
        C = cls.taylor_C(theta)
        R = I + A * wx + B * wx @ wx
        V = I + B * wx + C * wx @ wx
        Rt = torch.cat([R, (V @ u[..., None])], dim=-1)
        return Rt

    @classmethod
    def SE3_to_se3_N(cls, poses_rt):
        poses_se3_list = []
        for i in range(poses_rt.shape[0]):
            pose_se3 = cls.SE3_to_se3(poses_rt[i])
            poses_se3_list.append(pose_se3)
        poses = torch.stack(poses_se3_list, 0)
        return poses

    @classmethod
    def SE3_to_se3(cls, Rt, eps=1e-8):  # [...,3,4]
        R, t = Rt.split([3, 1], dim=-1)
        w = cls.SO3_to_so3(R)
        wx = cls.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = cls.taylor_A(theta)
        B = cls.taylor_B(theta)
        invV = I - 0.5 * wx + (1 - A / (2 * B)) / (theta ** 2 + eps) * wx @ wx
        u = (invV @ t)[..., 0]
        wu = torch.cat([w, u], dim=-1)
        return wu

    @classmethod
    def SO3_to_so3(cls, R, eps=1e-7):  # [...,3,3]
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        theta = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()[
                    ..., None, None
                ] % torch.pi  # ln(R) will explode if theta==pi
        lnR = (
                1 / (2 * cls.taylor_A(theta) + 1e-8) * (R - R.transpose(-2, -1))
        )
        w0, w1, w2 = lnR[..., 2, 1], lnR[..., 0, 2], lnR[..., 1, 0]
        w = torch.stack([w0, w1, w2], dim=-1)
        return w

    @classmethod
    def taylor_A(cls, x, nth=10):
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.0
        for i in range(nth + 1):
            if i > 0:
                denom *= (2 * i) * (2 * i + 1)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans

    @classmethod
    def taylor_B(cls, x, nth=10):
        # Taylor expansion of (1-cos(x))/x**2
        ans = torch.zeros_like(x)
        denom = 1.0
        for i in range(nth + 1):
            denom *= (2 * i + 1) * (2 * i + 2)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans

    @classmethod
    def taylor_C(cls, x, nth=10):
        # Taylor expansion of (x-sin(x))/x**3
        ans = torch.zeros_like(x)
        denom = 1.0
        for i in range(nth + 1):
            denom *= (2 * i + 2) * (2 * i + 3)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans

    @classmethod
    def skew_symmetric(cls, w):
        w0, w1, w2 = w.unbind(dim=-1)
        O = torch.zeros_like(w0)
        wx = torch.stack(
            [
                torch.stack([O, -w2, w1], dim=-1),
                torch.stack([w2, O, -w0], dim=-1),
                torch.stack([-w1, w0, O], dim=-1),
            ],
            dim=-2,
        )
        return wx

    def forward(self, pose_idx):
        shared_tr, indep_tr = None, None

        if self.mode in ["shared", "both"]:
            shared_tr = self.to_homo(self.se3_to_SE3(self.shared_params[None]))
            shared_tr = shared_tr.repeat(pose_idx.shape[0], 1, 1)

        if self.mode in ["independent_rel", "independent_abs", "both"]:
            indep_tr = self.to_homo(self.se3_to_SE3(self.independent_params[pose_idx]))

        if self.mode == "independent_abs":
            return indep_tr  # This is already the refined transform
        elif self.mode in "independent_rel":
            return torch.bmm(self.test_poses_SE3[pose_idx], indep_tr)
        elif self.mode == "shared":
            return torch.bmm(self.test_poses_SE3[pose_idx], shared_tr)
        elif self.mode == "both":
            return torch.bmm(torch.bmm(self.test_poses_SE3[pose_idx], shared_tr), indep_tr)
