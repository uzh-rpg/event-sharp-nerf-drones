import torch
import numpy as np

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from collections import namedtuple

import evo.main_ape as main_ape
import evo.core.geometry as geometry
from evo.core import lie_algebra as lie
from evo.core import trajectory, sync, metrics

def sanity_check_npy(npy1_, npy2_):
    """
    Check if two numpy arrays are equal.
    :param npy1_: first numpy array
    :param npy2_: second numpy array
    :return: True if the arrays are equal, False otherwise
    """
    # load the poses bounds
    npy1 = np.load(npy1_)
    npy2 = np.load(npy2_)
    if npy1.shape != npy2.shape:
        print(npy1_ + " and " + npy2_ + " have different shapes.")
        return False
    if not np.allclose(npy1, npy2, atol=1e-5):
        print(npy1_ + " and " + npy2_ + " are not equal.")
        return False
    print(npy1_ + " and " + npy2_ + " are the same.")
    return True

def save_perturbed_poses(original_poses_bounds_npy, perturbed_poses, save_name, llffhold=None, llffhold_end=None,
                         data_type="synthetic"):
    """
    Save perturbed poses to a numpy file, preserving test poses.
    :param original_poses_bounds_npy: path to original poses_bounds.npy
    :param perturbed_poses: numpy array, shape (N_train, 3, 4)
    :param save_name: suffix to save the new file
    :param data_type: 'synthetic' or 'real'
    :param llff_hold: test image frequency or count
    """
    assert data_type in ["synthetic", "real", "events"], "data_type must be 'synthetic' or 'real' or 'events'"
    if data_type != "events":
        if llffhold == 8 and not llffhold_end:
            data_type = "synthetic"
        elif llffhold == 5 and llffhold_end:
            data_type = "real"
        else:
            raise ValueError("Check llffhold and llffhold_end values")
    # Load original poses bounds
    original_poses_bounds = np.load(original_poses_bounds_npy)  # shape [N, 17]
    N_total = original_poses_bounds.shape[0]
    poses_intrinsic = original_poses_bounds[:, :-2].reshape([N_total, 3, 5])
    bounds = original_poses_bounds[:, -2:]  # shape [N, 2]

    if data_type == "synthetic":
        train_indices = np.array([i for i in range(N_total) if i % llffhold != 0])
    elif data_type == "real":
        train_indices = np.arange(N_total - llffhold)
    elif data_type == "events":
        train_indices = np.arange(N_total)
    assert len(train_indices) == perturbed_poses.shape[0], \
        f"Mismatch: {len(train_indices)} training indices vs {perturbed_poses.shape[0]} perturbed poses"

    poses_intrinsic[train_indices, :3, :4] = perturbed_poses

    poses_arr_new = np.concatenate([
        poses_intrinsic.reshape([N_total, 15]),  # flatten [3, 5] matrices
        bounds  # [N, 2]
    ], axis=1)  # Final shape: [N, 17]

    # Save the new poses bounds
    save_path = original_poses_bounds_npy.replace(".npy", f"_{save_name}.npy")
    np.save(save_path, poses_arr_new)

    # Check if the new poses bounds are equal to the original ones
    s = sanity_check_npy(save_path, original_poses_bounds_npy)
    if 'random_rot0.0_0.0_tran0.0_0.0' in save_path:
        assert s, "New poses bounds (random_rot0.0_0.0_tran0.0_0.0) are not equal to the original ones"
    elif 'relative_rot0.0_tran0.0' in save_path:
        assert s, "New poses bounds (relative_rot0.0_tran0.0) are not equal to the original ones"
    else:
        assert not s, "New poses bounds are equal to the original ones, check the perturbation process."



def _is_pure_rotation_matrix(M):
    """
    Check if a given matrix is a pure rotation matrix.
    :param M: a numpy ndarray of shape (N, 3, 3)
    :return: a boolean ndarray of shape (N,) indicating whether each matrix is a pure rotation matrix
    """
    # Check if each matrix in the list is square
    if M.shape[1] != M.shape[2]:
        return False

    # Check if the determinant of each matrix is +1
    det = np.linalg.det(M)
    is_det_close_to_one = np.isclose(det, 1.0)
    if not np.all(is_det_close_to_one):
        return False

    # Check if the transpose of each matrix is its inverse
    MT = np.transpose(M, (0, 2, 1))
    M_inv = np.linalg.inv(M)
    is_MT_close_to_M_inv = np.allclose(MT, M_inv, atol=5e-7)
    if not np.all(is_MT_close_to_M_inv):
        return False
    return True


def _get_slerp_interpolator(tss_poses_us, poses_rots, poses_trans):
    """
    Input
    :tss_poses_ns list of known tss
    :poses_rots list of 3x3 np.arrays
    :poses_trans list of 3x1 np.arrays
    :tss_query_ns list of query tss

    Returns:
    :rots list of rots at tss_query_ns
    :trans list of translations at tss_query_ns
    """
    # Setup Rot interpolator
    rot_interpolator = Slerp(tss_poses_us, R.from_matrix(poses_rots))
    # Setup trans interpolator
    trans_interpolator = interp1d(x=tss_poses_us, y=poses_trans, axis=0, kind="cubic", bounds_error=True)

    # Create interpolator as a closure to avoid creating the
    # trans_interpolator and rot_interpolator each time
    def interpolator(tss_query_ns):
        tss_query_ns = np.clip(tss_query_ns, tss_poses_us[0], tss_poses_us[-1])
        # Query rot interpolator
        rots = rot_interpolator(tss_query_ns).as_matrix()
        # Query trans interpolator
        trans = trans_interpolator(tss_query_ns)

        return rots, trans
    return interpolator


def _get_slerp_interpolator_exposure(tss_poses_us, poses_rots, poses_trans, tms_start, tms_end):
    """
    Input
    :tss_poses_ns list of known tss
    :poses_rots list of 3x3 np.arrays
    :poses_trans list of 3x1 np.arrays
    :tss_query_ns list of query tss

    Returns:
    :rots list of rots at tss_query_ns
    :trans list of translations at tss_query_ns
    """

    iterpolators = []
    for s, e in zip(tms_start, tms_end):
        idx_inside = np.logical_and(tss_poses_us >= s, tss_poses_us <= e)
        iterpolators.append(
            _get_slerp_interpolator(tss_poses_us[idx_inside], poses_rots[idx_inside], poses_trans[idx_inside])
        )

    # Create interpolator as a closure to avoid creating the interpolators each time
    def interpolator(tss_query_ns):
        # Sort for faster splitting
        idx_sort = np.argsort(tss_query_ns)
        tss_query_ns = tss_query_ns[idx_sort]
        # Split by exposure
        interp_idx = np.searchsorted(tms_start, tss_query_ns, side="right") - 1
        iterp_id, unique_idx = np.unique(interp_idx, return_index=True)
        grouped_tss_query_ns = np.split(tss_query_ns, unique_idx[1:])

        all_rots, all_trans = [], []
        for i, g_tss_query_ns in zip(iterp_id, grouped_tss_query_ns):
            rots, trans = iterpolators[i](g_tss_query_ns)
            all_rots.append(rots)
            all_trans.append(trans)
        all_rots = np.concatenate(all_rots, axis=0)
        all_trans = np.concatenate(all_trans, axis=0)

        # Restore original order
        idx_unsort = np.argsort(idx_sort)
        all_rots = all_rots[idx_unsort]
        all_trans = all_trans[idx_unsort]

        return all_rots, all_trans
    return interpolator


def procrustes_analysis(X0, X1):
    """
    Compute the Procrustes analysis between two sets of translation vectors.
    To align P1 to P0: P1to0 = (P1-t1)/s1@R.t()*s0+t0

    Source: https://github.dev/chenhsuanlin/bundle-adjusting-NeRF
    :param tr0: translation vectors of shape (N, 3)
    :param tr1: translation vectors of shape (N, 3)
    :return:
    """
    solution = namedtuple('ProcrustesSolution', ['t0', 't1', 's0', 's1', 'R'])

    # translation
    t0 = X0.mean(dim=0,keepdim=True)
    t1 = X1.mean(dim=0,keepdim=True)
    X0c = X0-t0
    X1c = X1-t1
    # scale
    s0 = (X0c**2).sum(dim=-1).mean().sqrt()
    s1 = (X1c**2).sum(dim=-1).mean().sqrt()
    X0cs = X0c/s0
    X1cs = X1c/s1
    # rotation (use double for SVD, float loses precision)
    U,S,V = (X0cs.t()@X1cs).double().svd(some=True)
    R = (U@V.t()).float()
    if R.det()<0: R[2] *= -1

    return solution(t0, t1, s0, s1, R)


def procrustes_align(from_poses, to_poses=None, procrustes_result=None):
    """
    Align two sets of poses using the Procrustes analysis.
    :param from_poses: torch poses [N, 4, 4]
    :param to_poses: torch poses [N, 4, 4]
    :return:
    """

    device = from_poses
    if procrustes_result is None:
        assert to_poses is not None, "to_poses must be provided if procrustes_result is None"
        procrustes_result = procrustes_analysis(to_poses[..., :3, 3], from_poses[..., :3, 3])

    t0, t1 = procrustes_result.t0.to(device), procrustes_result.t1.to(device)
    s0, s1 = procrustes_result.s0.to(device), procrustes_result.s1.to(device)
    R = procrustes_result.R.to(device)

    center_aligned = (from_poses[..., :3, 3] - t1) / s1 @ R.t() * s0 + t0
    R_aligned = from_poses[..., :3, :3] @ R.t()
    t_aligned = (R_aligned @ center_aligned[..., None])[..., 0]

    aligned_poses = torch.cat([R_aligned, t_aligned[..., None]], axis=-1)
    return aligned_poses


def umeyama_solution(Xt, Yt, estimate_scale=False, allow_reflection=False, eps=1e-9):
    """
    Finds a similarity transformation (rotation `R`, translation `T`
    and optionally scale `s`)  between two given sets of corresponding
    `d`-dimensional points `X` and `Y` such that:

    `s[i] X[i] R[i] + T[i] = Y[i]`,

    for all batch indexes `i` in the least squares sense.

    The algorithm is also known as Umeyama [1].

    Args:
        **X**: Batch of `d`-dimensional points of shape `(minibatch, num_point, d)`
            or a `Pointclouds` object.
        **Y**: Batch of `d`-dimensional points of shape `(minibatch, num_point, d)`
            or a `Pointclouds` object.
        **weights**: Batch of non-negative weights of
            shape `(minibatch, num_point)` or list of `minibatch` 1-dimensional
            tensors that may have different shapes; in that case, the length of
            i-th tensor should be equal to the number of points in X_i and Y_i.
            Passing `None` means uniform weights.
        **estimate_scale**: If `True`, also estimates a scaling component `s`
            of the transformation. Otherwise assumes an identity
            scale and returns a tensor of ones.
        **allow_reflection**: If `True`, allows the algorithm to return `R`
            which is orthonormal but has determinant==-1.
        **eps**: A scalar for clamping to avoid dividing by zero. Active for the
            code that estimates the output scale `s`.

    Returns:
        3-element named tuple `SimilarityTransform` containing
        - **R**: Batch of orthonormal matrices of shape `(minibatch, d, d)`.
        - **T**: Batch of translations of shape `(minibatch, d)`.
        - **s**: batch of scaling factors of shape `(minibatch, )`.

    References:
        [1] Shinji Umeyama: Least-Suqares Estimation of
        Transformation Parameters Between Two Point Patterns
    """

    solution = namedtuple('UmeyamaSolution', ['R', 'T', 's'])

    # make sure we convert input Pointclouds structures to tensors
    num_points = torch.tensor([Xt.shape[1]]).repeat(Xt.shape[0])
    num_points_Y = torch.tensor([Yt.shape[1]]).repeat(Yt.shape[0])

    if (Xt.shape != Yt.shape) or (num_points != num_points_Y).any():
        raise ValueError(
            "Point sets X and Y have to have the same \
            number of batches, points and dimensions."
        )

    b, n, dim = Xt.shape

    if (num_points < Xt.shape[1]).any() or (num_points < Yt.shape[1]).any():
        # in case we got Pointclouds as input, mask the unused entries in Xc, Yc
        mask = (
                torch.arange(n, dtype=torch.int64, device=Xt.device)[None]
                < num_points[:, None]
        ).type_as(Xt)
        weights = mask

    # compute the centroids of the point sets
    Xmu = Xt.mean(dim=-2, keepdims=True)
    Ymu = Yt.mean(dim=-2, keepdims=True)

    # mean-center the point sets
    Xc = Xt - Xmu
    Yc = Yt - Ymu

    total_weight = torch.clamp(num_points, 1)

    if (num_points < (dim + 1)).any():
        print(
            "WARNING: The size of one of the point clouds is <= dim+1. "
            + "corresponding_points_alignment cannot return a unique rotation."
        )

    # compute the covariance XYcov between the point sets Xc, Yc
    XYcov = torch.bmm(Xc.transpose(2, 1), Yc)
    XYcov = XYcov / total_weight[:, None, None]

    # decompose the covariance matrix XYcov
    U, S, V = torch.svd(XYcov)

    # catch ambiguous rotation by checking the magnitude of singular values
    if (S.abs() <= 1e-15).any() and not (
            num_points < (dim + 1)
    ).any():
        print(
            "WARNING: Excessively low rank of "
            + "cross-correlation between aligned point clouds. "
            + "corresponding_points_alignment cannot return a unique rotation."
        )

    # identity matrix used for fixing reflections
    E = torch.eye(dim, dtype=XYcov.dtype, device=XYcov.device)[None].repeat(b, 1, 1)

    if not allow_reflection:
        # reflection test:
        #   checks whether the estimated rotation has det==1,
        #   if not, finds the nearest rotation s.t. det==1 by
        #   flipping the sign of the last singular vector U
        R_test = torch.bmm(U, V.transpose(2, 1))
        E[:, -1, -1] = torch.det(R_test)

    # find the rotation matrix by composing U and V again
    R = torch.bmm(torch.bmm(U, E), V.transpose(2, 1))

    if estimate_scale:
        # estimate the scaling component of the transformation
        trace_ES = (torch.diagonal(E, dim1=1, dim2=2) * S).sum(1)
        Xcov = (Xc * Xc).sum((1, 2)) / total_weight

        # the scaling component
        s = trace_ES / torch.clamp(Xcov, eps)

        # translation component
        T = Ymu[:, 0, :] - s[:, None] * torch.bmm(Xmu, R)[:, 0, :]
    else:
        # translation component
        T = Ymu[:, 0, :] - torch.bmm(Xmu, R)[:, 0, :]

        # unit scaling since we do not estimate scale
        s = T.new_ones(b)

    return solution(R, T, s)


def umeyama_align(from_poses, to_poses=None, umeyama_result=None):
    """
    Align two sets of poses using the Procrustes analysis.
    :param from_poses: torch poses [N, 4, 4]
    :param to_poses: torch poses [N, 4, 4]
    :return:
    """

    device = from_poses.device
    if umeyama_result is None:
        assert to_poses is not None, "to_poses must be provided if umeyama_result is None"
        umeyama_result = umeyama_solution(from_poses[..., :3, 3][None], to_poses[..., :3, 3][None], estimate_scale=True)
    # Remove temporary batch dimension and move to same device
    R, T, s = umeyama_result.R[0].to(device), umeyama_result.T[0].to(device), umeyama_result.s[0].to(device)

    # s[i] X[i] R[i] + T[i] = Y[i]

    center_aligned = s * (from_poses[..., :3, 3] @ R) + T
    R_aligned = from_poses[..., :3, :3] @ R
    t_aligned = (R_aligned @ center_aligned[..., None])[..., 0]

    aligned_poses = torch.cat([R_aligned, t_aligned[..., None]], axis=-1)
    return aligned_poses


def to_evo_trajectory(poses, timestamps_us=None):
    """
    Convert a set of poses to an evo trajectory.
    :param poses: a numpy ndarray of shape (N, 4, 4) representing the set of poses
    :param timestamps_us: a numpy ndarray of shape (N,) representing the timestamps in microseconds
    :return: an evo trajectory representing the set of poses
    """

    if isinstance(poses, trajectory.PosePath3D):
        return poses
    if isinstance(poses, torch.Tensor):
        poses = poses.cpu().numpy()

    quat = R.from_matrix(poses[..., :3, :3]).as_quat()
    if timestamps_us is not None:
        trj = trajectory.PoseTrajectory3D(
            positions_xyz=poses[..., :3, 3],
            orientations_quat_wxyz=quat[..., [3, 0, 1, 2]],
            timestamps=timestamps_us / 1e6
        )
    else:
        trj = trajectory.PosePath3D(
            positions_xyz=poses[..., :3, 3],
            orientations_quat_wxyz=quat[..., [3, 0, 1, 2]]
        )
    return trj


def from_evo_trajectory(trj, return_timestamps_us=False):
    """
    Convert an evo trajectory to a set of poses.
    :param trj: an evo trajectory representing the set of poses
    :param return_timestamps_us: whether to return the timestamps in microseconds
    :return: a numpy ndarray of shape (N, 4, 4) representing the set of poses. If return_timestamps is True, also
    return a numpy ndarray of shape (N,) representing the timestamps in microseconds
    """
    pos = trj.positions_xyz
    quat = trj.orientations_quat_wxyz[..., [1, 2, 3, 0]]
    rot = R.from_quat(quat).as_matrix()
    poses = np.concatenate([rot, pos[..., None]], axis=-1)

    if return_timestamps_us:
        if isinstance(trj, trajectory.PoseTrajectory3D):
            return poses, trj.timestamps * 1e6
        return poses, None
    return poses


def evo_solution(from_poses, to_poses=None):
    from_trj = to_evo_trajectory(from_poses)
    to_trj = to_evo_trajectory(to_poses)

    r_a, t_a, s = geometry.umeyama_alignment(from_trj.positions_xyz.T,
                                             to_trj.positions_xyz.T,
                                             True)
    return r_a, t_a, s


def evo_align(from_poses, to_poses=None, evo_result=None):
    device, dtype = from_poses.device, from_poses.dtype

    from_trj = to_evo_trajectory(from_poses)

    if evo_result is None:
        to_trj = to_evo_trajectory(to_poses)
        evo_result = evo_solution(from_trj, to_trj)

    r_a, t_a, s = evo_result
    from_trj.scale(s)
    from_trj.transform(lie.se3(r_a, t_a))

    aligned_poses = from_evo_trajectory(from_trj)
    aligned_poses = torch.tensor(aligned_poses, device=device, dtype=dtype)
    return aligned_poses


def translation_error(poses1, poses2):
    """
    Compute the translation error between two sets of poses.
    :param poses1: a numpy ndarray of shape (N, 4, 4) representing the first set of poses
    :param poses2: a numpy ndarray of shape (N, 4, 4) representing the second set of poses
    :return: a numpy ndarray of shape (N,) representing the translation error between the two sets of poses
    """
    trans1 = poses1[:, :3, 3]
    trans2 = poses2[:, :3, 3]
    return np.linalg.norm(trans1 - trans2, axis=1).mean()


def rotation_error(poses1, poses2):
    """
    Compute the rotation error between two sets of poses in degrees.
    :param poses1: a numpy ndarray of shape (N, 4, 4) representing the first set of poses
    :param poses2: a numpy ndarray of shape (N, 4, 4) representing the second set of poses
    :return: a numpy ndarray of shape (N,) representing the rotation error between the two sets of poses
    """
    rot1 = poses1[:, :3, :3].transpose(0,2,1)
    rot2 = poses2[:, :3, :3]
    # Clip is necessary to avoid NaNs in the arccos computation. Should not be necessary in theory
    tr = np.clip((np.trace(rot1 @ rot2, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0)
    return np.rad2deg(np.arccos(tr)).mean()


def trajectory_errors(traj_gt, tss_gt_us, traj_est, tss_est_us,
                      plot=False, est_name=None, save_folder=None, save_prefix=""):

    evo_traj_gt = to_evo_trajectory(traj_gt, tss_gt_us)
    evo_traj_est = to_evo_trajectory(traj_est, tss_est_us)
    gt_traj_len = evo_traj_gt.get_infos()["path length (m)"]

    # if traj_gt.shape != traj_est.shape or not bool(torch.all(tss_gt_us == tss_est_us)):
    if len(traj_gt) != len(traj_est) or not bool(torch.all(tss_gt_us == tss_est_us)):
        evo_traj_gt, evo_traj_est = sync.associate_trajectories(evo_traj_gt, evo_traj_est, max_diff=1)

    try:
        ape_trans = main_ape.ape(evo_traj_gt, evo_traj_est,
                                pose_relation=metrics.PoseRelation.translation_part,
                                align=True, correct_scale=True)
    except:
        ape_trans = main_ape.ape(evo_traj_gt, evo_traj_est,
                                pose_relation=metrics.PoseRelation.translation_part,
                                align_origin=True)
    ATE = ape_trans.stats["rmse"] * 100
    MPE = ape_trans.stats["mean"] / gt_traj_len * 100

    if plot:
        from utils.viz import plot_evo_trajectories

        figs = plot_evo_trajectories(
            [evo_traj_gt, evo_traj_est], ["gt", est_name],
            styles=["-", "--"], colors=["black", "dodgerblue"],
            save_folder=save_folder, save_prefix=save_prefix
        )
        return ATE, MPE, figs
    return ATE, MPE
