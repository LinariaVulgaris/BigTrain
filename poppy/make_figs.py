import os, sys, argparse
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, os.getcwd())
from utils.load_data import get_files, load_files

# 论文同款误差配色: 0度=黄, 25度=红, 50度=黑
ERR_CMAP = LinearSegmentedColormap.from_list("err", ["#ffff66", "#ff0000", "#000000"])
ERR_CMAP.set_bad("white")  # mask外的像素显示为白色


def to_np(t):
    """把 (1,C,H,W) 的张量转成 numpy: 3通道 -> (H,W,3), 单通道 -> (H,W)"""
    a = t.detach().float().cpu().numpy() if torch.is_tensor(t) else np.asarray(t, dtype=np.float32)
    a = np.squeeze(a)
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    return a


def unit(n):
    return n / (np.linalg.norm(n, axis=2, keepdims=True) + 1e-8)


def ang_err(p, g):
    """逐像素角度误差(度)"""
    return np.degrees(np.arccos(np.clip((unit(p) * unit(g)).sum(2), -1, 1)))


def vis_normal(n, mask):
    img = np.clip((unit(n) + 1) / 2, 0, 1)
    img[~mask] = 1.0
    return img


def vis_err(e, mask):
    return np.where(mask, e, np.nan)


def vis_pol(phi, rho, mask):
    """偏振图: 颜色=AoLP(偏振角), 亮度=DoLP(偏振度)"""
    aolp = np.mod(phi, np.pi)
    dolp = np.clip(rho, 0, 1)
    dolp = np.clip(dolp / (np.percentile(dolp[mask], 99) + 1e-6), 0, 1)
    rgb = matplotlib.colormaps["hsv"](aolp / np.pi)[..., :3] * dolp[..., None]
    rgb[~mask] = 0
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)            # 数据目录(含pol000等)
    ap.add_argument("--base", required=True)            # 不加Poppy的输出目录
    ap.add_argument("--poppy", required=True)           # 加Poppy的输出目录
    ap.add_argument("--out", default="figs")
    ap.add_argument("--gain", type=float, default=4.0)  # 输入图仅用于显示的亮度放大倍数
    ap.add_argument("--crop", type=int, nargs=4, default=None,
                    help="x0 y0 x1 y1, 额外输出局部放大图(原图像素坐标)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    summary = []

    for name in sorted(get_files(args.data)):
        fb = os.path.join(args.base, "outputs", name + ".npz")
        fp = os.path.join(args.poppy, "outputs", name + ".npz")
        if not (os.path.exists(fb) and os.path.exists(fp)):
            print(f"跳过 {name}: 缺少输出文件")
            continue

        image, I, Q, U, phi, rho, mask, gt = load_files(args.data, name, "cpu")
        image = np.clip(to_np(image), 0, 1)
        if image.ndim == 2:
            image = np.stack([image] * 3, -1)
        image_show = np.clip(image * args.gain, 0, 1)      # 仅用于显示
        phi, rho = to_np(phi), to_np(rho)
        mask = to_np(mask)
        mask = (mask[..., 0] if mask.ndim == 3 else mask) > 0.5
        nb = np.load(fb)["normals"]
        npp = np.load(fp)["normals"]
        gt = None if gt is None else to_np(gt)

        panels = [("Input (S0 x%g, display only)" % args.gain, image_show, None),
                  ("Polarization", vis_pol(phi, rho, mask), None),
                  ("Backbone", vis_normal(nb, mask), None)]
        if gt is not None:
            eb, ep = ang_err(nb, gt), ang_err(npp, gt)
            mb, mp = eb[mask].mean(), ep[mask].mean()
            summary.append((name, mb, mp))
            panels.append((f"Error {mb:.2f} deg", vis_err(eb, mask), "err"))
        panels.append(("+ Poppy", vis_normal(npp, mask), None))
        if gt is not None:
            panels += [(f"Error {mp:.2f} deg", vis_err(ep, mask), "err"),
                       ("Ground truth", vis_normal(gt, mask), None)]

        def draw(sl, fname):
            fig, axs = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.4))
            for ax, (title, im, kind) in zip(axs, panels):
                if kind == "err":
                    ax.imshow(im[sl], cmap=ERR_CMAP, vmin=0, vmax=50)
                else:
                    ax.imshow(im[sl])
                ax.set_title(title, fontsize=10)
                ax.axis("off")
            if gt is not None:
                sm = ScalarMappable(norm=Normalize(0, 50), cmap=ERR_CMAP)
                fig.colorbar(sm, ax=list(axs), fraction=0.012, pad=0.01,
                             label="Angular error (deg)")
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)

        draw((slice(None), slice(None)), os.path.join(args.out, name + ".png"))
        if args.crop:
            x0, y0, x1, y1 = args.crop
            draw((slice(y0, y1), slice(x0, x1)), os.path.join(args.out, name + "_crop.png"))

    summary.sort(key=lambda r: r[1] - r[2], reverse=True)
    print("按提升幅度排序 (backbone误差 -> Poppy误差):")
    for n, b, p in summary:
        print(f"  {n}: {b:.2f} -> {p:.2f}  (降低 {b - p:.2f} 度)")


if __name__ == "__main__":
    main()