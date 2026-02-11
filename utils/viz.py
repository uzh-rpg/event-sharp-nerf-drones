import os
import tempfile
import matplotlib.pyplot as plt

from utils.poses import to_evo_trajectory
from evo.tools import plot


def plot_evo_trajectories(trajectories, names, styles=None, colors=None,
                          modes=(plot.PlotMode.xz, plot.PlotMode.xy, plot.PlotMode.yz),
                          align_ref=None, save_folder=None, save_prefix="", viz=False):
    if save_folder is None:
        save_folder = tempfile.mkdtemp()
    else:
        os.makedirs(save_folder, exist_ok=True)

    # Move trajectories to EVO format
    trajectories = [to_evo_trajectory(traj) for traj in trajectories]
    if align_ref is not None:
        trj_ref = trajectories[align_ref]
        for i, traj in enumerate(trajectories):
            if i != align_ref:
                traj.align(trj_ref, correct_scale=True)

    fig_paths = []
    for mode in modes:
        fig = plt.figure(figsize=(8, 8))
        ax = plot.prepare_axis(fig, mode)
        for i, (traj, name) in enumerate(zip(trajectories, names)):
            style = styles[i] if styles is not None else '-'
            color = colors[i] if colors is not None else 'black'
            plot.traj(ax, mode, traj, style, color, label=name)
        ax.legend()
        fig.tight_layout()

        if viz:
            plt.show()

        file_path = os.path.join(save_folder, f'{save_prefix}_{mode.name}_align_trj.png')
        plt.savefig(file_path, dpi=100)
        fig_paths.append(file_path)

    return fig_paths