import argparse
import os
import math
import random

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
import numpy as np

from processed_dataloader import ProcessedH5Dataset


def angle_to_vec(deg: float, scale: float = 0.05):
    rad = math.radians(deg)
    return math.cos(rad) * scale, math.sin(rad) * scale


def get_team_color(team_idx: float) -> str:
    # Simple two-team palette with fallback
    if int(team_idx) == 0:
        return "#1f77b4"  # blue
    if int(team_idx) == 1:
        return "#d62728"  # red
    return "#7f7f7f"      # gray


def visualize_sample(h5_path: str, index: int, fps: int = 6, dpi: int = 120, auto_close: bool = True):
    ds = ProcessedH5Dataset(h5_path, group_name="processed", return_numpy=True)
    if index < 0:
        index = random.randint(0, len(ds) - 1)
    temporal, spatial, actions = ds[index]

    # Extract ego temporal features
    # 0: team_index, 1: rel_pos_x, 2: rel_pos_y, 3: rotation (normalized)
    team_seq = temporal[:, 0]
    x_seq = temporal[:, 1]
    y_seq = temporal[:, 2]
    rot_seq_deg = temporal[:, 3] * 360.0
    # Per-timestep deltas from temporal features
    dx_seq = temporal[:, 10]
    dy_seq = temporal[:, 11]

    # Spatial snapshot (T-2) for 10 agents
    teams_sp = spatial[:, 0]
    xs_sp = spatial[:, 1]
    ys_sp = spatial[:, 2]
    rots_sp_deg = spatial[:, 3] * 360.0

    # Action vector (movement to next step)
    dx, dy, drot = float(actions[0]), float(actions[1]), float(actions[2])

    # Plot setup: single window with two subplots (map + time series)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax, ax_t) = plt.subplots(1, 2, figsize=(11, 6.2), dpi=dpi, gridspec_kw={"width_ratios": [3, 2]})
    fig.suptitle(f"Agent Timelapse (sample {index})", fontsize=14)
    ax.set_xlabel("rel_pos_x")
    ax.set_ylabel("rel_pos_y")
    ax.set_aspect("equal", adjustable="box")

    # Bounds with margins
    all_x = np.concatenate([x_seq, xs_sp])
    all_y = np.concatenate([y_seq, ys_sp])
    if all_x.size == 0:
        all_x = np.array([0.0])
    if all_y.size == 0:
        all_y = np.array([0.0])
    min_x, max_x = float(np.min(all_x)), float(np.max(all_x))
    min_y, max_y = float(np.min(all_y)), float(np.max(all_y))
    pad_x = max(0.05, 0.1 * (max_x - min_x + 1e-6))
    pad_y = max(0.05, 0.1 * (max_y - min_y + 1e-6))
    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)

    # Static plot for other agents at T-2
    for i in range(spatial.shape[0]):
        if np.all(spatial[i] == 0):
            continue
        color = get_team_color(teams_sp[i])
        ax.scatter(xs_sp[i], ys_sp[i], s=40, c=color, alpha=0.7, edgecolors="k", linewidths=0.5, zorder=2)
        vx, vy = angle_to_vec(rots_sp_deg[i], scale=0.04)
        ax.arrow(xs_sp[i], ys_sp[i], vx, vy, head_width=0.015, head_length=0.02, fc=color, ec=color, alpha=0.8, zorder=2)

    # Ego path (time-colored line) and current point
    # Build colored segments for path
    if len(x_seq) > 1:
        points = np.array([x_seq, y_seq]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap="viridis", norm=plt.Normalize(0, len(x_seq) - 1))
        lc.set_array(np.arange(len(x_seq) - 1))
        lc.set_linewidth(2)
        lc.set_alpha(0.9)
        ax.add_collection(lc)
    path_line, = ax.plot([], [], "o-", ms=3, lw=0.5, c="#2ca02c", alpha=0.7, label="ego path")
    ego_pt = ax.scatter([], [], s=60, c="#2ca02c", edgecolors="k", linewidths=0.5, zorder=3)
    ego_head = ax.arrow(0, 0, 0, 0, head_width=0.02, head_length=0.03, fc="#2ca02c", ec="#2ca02c", alpha=0.9)

    # Action arrow at last timestep (T-2): from last pos to last pos + (dx, dy)
    x_last, y_last = x_seq[-1], y_seq[-1]
    act_color = "#9467bd"
    ax.arrow(x_last, y_last, dx, dy, head_width=0.02, head_length=0.03, fc=act_color, ec=act_color, alpha=0.9, zorder=3)
    ax.annotate(f"Δ=({dx:+.2f},{dy:+.2f})", (x_last , y_last ), xytext=(6, 6), textcoords="offset points", color=act_color)

    # Info box with final action and per-step (dx,dy) from temporal features
    lines = [f"Final Δ = ({dx:+.2f}, {dy:+.2f})", "Per-step Δ (temporal):"]
    for t in range(len(dx_seq)):
        lines.append(f"t{t:02d}: ({dx_seq[t]:+0.2f}, {dy_seq[t]:+0.2f})")
    info_text = "\n".join(lines)
    ax.text(
        0.98,
        0.98,
        info_text,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=8,
        family="monospace",
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="#444444"),
        zorder=5,
    )

    # Right subplot: per-step Δx, Δy over time
    t_axis = np.arange(len(dx_seq))
    ax_t.plot(t_axis, dx_seq, "-o", color="#1f77b4", label="dx", ms=3, lw=1.5)
    ax_t.plot(t_axis, dy_seq, "-o", color="#ff7f0e", label="dy", ms=3, lw=1.5)
    ax_t.axhline(0.0, color="#666666", lw=0.8, ls=":")
    ax_t.set_xlabel("timestep")
    ax_t.set_ylabel("delta value")
    ax_t.set_title("Per-step Δ values")
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(loc="upper right")
    t_cursor = ax_t.axvline(0, color="#2ca02c", lw=1.5, ls="--", alpha=0.8)

    ax.legend(loc="upper right")

    # Animation functions
    def init():
        path_line.set_data([], [])
        ego_pt.set_offsets(np.empty((0, 2)))
        t_cursor.set_xdata([0])
        return path_line, ego_pt, ego_head, t_cursor

    def update(frame):
        # path up to frame
        xs = x_seq[: frame + 1]
        ys = y_seq[: frame + 1]
        path_line.set_data(xs, ys)
        ego_pt.set_offsets(np.column_stack([xs[-1:], ys[-1:]]))

        # update heading arrow
        nonlocal ego_head
        # remove previous arrow by replacing it
        ego_head.remove()
        vx, vy = angle_to_vec(rot_seq_deg[frame], scale=0.06)
        ego_head = ax.arrow(xs[-1], ys[-1], vx, vy, head_width=0.025, head_length=0.03, fc="#2ca02c", ec="#2ca02c", alpha=0.9)
        # update time cursor
        t_cursor.set_xdata([frame])
        return path_line, ego_pt, ego_head, t_cursor

    frames = len(x_seq)
    interval_ms = int(1000 // max(1, fps))
    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frames,
        interval=interval_ms,
        blit=False,
        repeat=False,
    )

    # Add pause functionality
    paused = [False]  # mutable for nonlocal
    remaining_timer = None

    def on_key(event):
        if event.key == ' ':
            paused[0] = not paused[0]
            if paused[0]:
                anim.event_source.stop()
                if remaining_timer is not None:
                    remaining_timer.stop()
                print("Paused (press space to resume)")
            else:
                anim.event_source.start()
                if auto_close and remaining_timer is not None:
                    remaining_timer.start()
                print("Resumed")

    fig.canvas.mpl_connect('key_press_event', on_key)

    if auto_close:
        total_ms = frames * interval_ms + 500
        remaining_timer = fig.canvas.new_timer(interval=total_ms)
        remaining_timer.add_callback(plt.close, fig)
        remaining_timer.start()

    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize processed attention_v1 sample as a timelapse")
    parser.add_argument("--h5_path", type=str, default="/Users/vaibhav/Desktop/processed_game_logs_attention_1_sub_magnitude_0_2_1000.h5")
    parser.add_argument("--index", type=int, default=-1, help="sample index (use -1 for random)")
    parser.add_argument("--num_samples", type=int, default=3, help="when index=-1, randomly play this many samples sequentially")
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()

    if args.index >= 0:
        visualize_sample(args.h5_path, args.index, fps=args.fps, auto_close=False)
        return

    # Play multiple random samples sequentially with auto-close
    ds = ProcessedH5Dataset(args.h5_path, group_name="processed", return_numpy=True)
    total = len(ds)
    if total == 0:
        print("Dataset empty")
        return
    count = max(1, min(args.num_samples, total))
    indices = np.random.choice(total, count, replace=False)
    print(f"Playing samples: {indices.tolist()}")
    for idx in indices:
        visualize_sample(args.h5_path, int(idx), fps=args.fps, auto_close=True)


if __name__ == "__main__":
    main()

"""
python3 -u visualize_processed.py \
  --h5_path "/Users/vaibhav/Desktop/processed_game_logs_attention_1_sub_1000.h5" \
  --index -1 --num_samples 10 --fps 5
  """
