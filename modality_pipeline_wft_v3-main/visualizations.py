"""
visualize_scans.py
==================
Edit `comparison_directories` below — that is the ONLY config needed.
Everything else (file discovery, subject count, panel layout) is derived at runtime.

Run:
    python visualize_scans.py
    python visualize_scans.py --subject 2 --plane coronal
    python visualize_scans.py --subject 0 --window 400 --level 200

Keybindings (in viewer):
    ←  /  →     : previous / next slice
    Scroll wheel: navigate slices
    L           : toggle loupe magnification
    D           : toggle circle draw mode
    C           : clear all circles
"""

import argparse
import os
import sys

import nibabel as nib
import nibabel.processing
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, RadioButtons, CheckButtons
from matplotlib.patches import Circle as MplCircle


# ══════════════════════════════════════════════════════════════════
#  CONFIG  ← edit this, nothing else
# ══════════════════════════════════════════════════════════════════
comparison_directories = [
    {
        "model_tag": "3DESAU",
        "logs_dir":  "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/DS_1_ESAU_3D_L1_32_conv",
    },
        {
        "model_tag": "ResVit",
        "logs_dir":  "/storage/an_inam/MR2MR/modality_pipeline_wft_v3/logs/DS_1_ResVIT",
    },
    # Add more models here, e.g.:
    # {
    #     "model_tag": "UNet3D",
    #     "logs_dir":  "logs/UNET_L1_DS2_10fold",
    # },
]
# ══════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────
#  FILE DISCOVERY  (derived from dict at runtime)
# ──────────────────────────────────────────────
def discover(subject_idx):
    first   = comparison_directories[0]
    gen_dir = os.path.join(first["logs_dir"], "generations")

    origs = sorted(f for f in os.listdir(gen_dir) if "_orig" in f)
    fakes = sorted(f for f in os.listdir(gen_dir) if "_fake" in f)

    print(f"Original scans ({len(origs)}):")
    for f in origs: print(f"  {f}")
    print(f"\nFake scans ({len(fakes)}) — first model:")
    for f in fakes: print(f"  {f}")

    if not origs: sys.exit(f"[ERROR] No _orig files in {gen_dir}")
    if not fakes: sys.exit(f"[ERROR] No _fake files in {gen_dir}")

    n_orig, n_fake = len(origs), len(fakes)

    if n_fake > n_orig:
        if n_fake % n_orig != 0:
            sys.exit(f"[ERROR] len(fake)={n_fake} is not a multiple of len(orig)={n_orig}.")
        ratio = n_fake // n_orig
    else:
        ratio = 1

    if subject_idx >= n_orig:
        sys.exit(f"[ERROR] --subject {subject_idx} out of range ({n_orig} subjects).")

    orig_path = os.path.join(gen_dir, origs[subject_idx])

    panels = []
    for cd in comparison_directories:
        gd   = os.path.join(cd["logs_dir"], "generations")
        fks  = sorted(f for f in os.listdir(gd) if "_fake" in f)
        hits = fks[subject_idx * ratio : (subject_idx + 1) * ratio]
        for h in hits:
            panels.append({"tag": cd["model_tag"], "path": os.path.join(gd, h)})

    return orig_path, panels


# ──────────────────────────────────────────────
#  NIBABEL HELPERS
# ──────────────────────────────────────────────
def load_canonical(path):
    return nib.as_closest_canonical(nib.load(path))

def enforce_affine(ref, img):
    if np.allclose(ref.affine, img.affine, atol=1e-3) and ref.shape[:3] == img.shape[:3]:
        return img
    print("    [resampling to match original affine]")
    return nibabel.processing.resample_from_to(img, ref, order=1, cval=0)

def to_array(img):
    return np.asarray(img.dataobj).astype(np.float32)


# ──────────────────────────────────────────────
#  SLICE HELPERS
# ──────────────────────────────────────────────
PLANE_AXIS = {"axial": 2, "coronal": 1, "sagittal": 0}

def n_slices(shape, plane):
    return shape[PLANE_AXIS[plane]]

def get_slice(data, plane, idx):
    return np.rot90(np.take(data, idx, axis=PLANE_AXIS[plane]))

def auto_wl(data):
    fg = data[data > data.min()]
    p2, p98 = np.percentile(fg, [2, 98])
    return float(p98 - p2), float((p2 + p98) / 2)


# ──────────────────────────────────────────────
#  VIEWER
# ──────────────────────────────────────────────
def launch_viewer(orig_data, panel_datas, panel_tags, ww, wl, init_plane, init_slice):

    matplotlib.rcParams.update({
        "figure.facecolor": "#080c10", "axes.facecolor":  "#000000",
        "axes.edgecolor":   "#1e2a35", "text.color":      "#c9d8e8",
        "font.family":      "monospace", "font.size":      8,
    })

    all_datas  = [orig_data] + panel_datas
    all_labels = ["Original"] + panel_tags
    n_panels   = len(all_datas)

    st = {
        "plane":      init_plane,
        "slice":      init_slice,
        "ww": ww, "wl": wl,
        "loupe":      False,
        "draw_mode":  False,
        "circles":    [],
        "drawing":    False,
        "draw_start": None,
    }

    # layout
    fig = plt.figure(figsize=(max(14, 3.6 * n_panels), 9), facecolor="#080c10")
    gs  = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[7, 2.2], hspace=0.06)
    img_gs  = gridspec.GridSpecFromSubplotSpec(1, n_panels, subplot_spec=gs[0], wspace=0.04)
    ctrl_gs = gridspec.GridSpecFromSubplotSpec(3, 6, subplot_spec=gs[1], wspace=0.45, hspace=0.75)

    axes = [fig.add_subplot(img_gs[i]) for i in range(n_panels)]
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor("#1e2a35")

    ax_sl = fig.add_subplot(ctrl_gs[0, :4])
    ax_ww = fig.add_subplot(ctrl_gs[1, :4])
    ax_wl = fig.add_subplot(ctrl_gs[2, :4])
    ax_pl = fig.add_subplot(ctrl_gs[:, 4])
    ax_tg = fig.add_subplot(ctrl_gs[:, 5])
    for a in (ax_sl, ax_ww, ax_wl, ax_pl, ax_tg):
        a.set_facecolor("#0d1117")
        for sp in a.spines.values(): sp.set_edgecolor("#1e2a35")

    kw    = dict(color="#00d4ff", track_color="#1e2a35")
    n_sl0 = n_slices(orig_data.shape, init_plane)
    sl_slice = Slider(ax_sl, "Slice",  0, n_sl0 - 1, valinit=init_slice, valstep=1, **kw)
    sl_ww    = Slider(ax_ww, "Window", 1, max(4000, ww * 2), valinit=ww, **kw)
    sl_wl    = Slider(ax_wl, "Level",  float(orig_data.min()), float(orig_data.max()), valinit=wl, **kw)
    for s in (sl_slice, sl_ww, sl_wl):
        s.label.set_color("#c9d8e8"); s.valtext.set_color("#00d4ff")

    radio  = RadioButtons(ax_pl, ("Axial", "Coronal", "Sagittal"), activecolor="#00d4ff")
    checks = CheckButtons(ax_tg, ["Loupe", "Draw"], actives=[False, False])
    ax_pl.set_title("Plane", color="#4a6070", fontsize=7, pad=2)
    ax_tg.set_title("Tools", color="#4a6070", fontsize=7, pad=2)

    lo0 = wl - ww / 2; hi0 = wl + ww / 2
    im_objs     = []
    loupe_axes  = []
    loupe_ims   = []
    loupe_rings = []
    circ_patches= [[] for _ in range(n_panels)]

    for i, (ax, data, label) in enumerate(zip(axes, all_datas, all_labels)):
        sl2d = get_slice(data, st["plane"], st["slice"])
        im   = ax.imshow(sl2d, cmap="gray", vmin=lo0, vmax=hi0,
                         interpolation="bilinear", aspect="equal")
        im_objs.append(im)
        color = "#00ff88" if i == 0 else "#00d4ff"
        ax.set_title(label, color=color, fontsize=9, pad=4, fontweight="bold")

        lax = ax.inset_axes([0.0, 0.72, 0.28, 0.28])
        lax.set_xticks([]); lax.set_yticks([])
        for sp in lax.spines.values(): sp.set_edgecolor("#00d4ff"); sp.set_linewidth(1.5)
        lax.set_visible(False)
        lim = lax.imshow(sl2d, cmap="gray", vmin=lo0, vmax=hi0,
                         interpolation="bilinear", aspect="equal")
        loupe_axes.append(lax); loupe_ims.append(lim)

        ring = MplCircle((0, 0), radius=20, fill=False, edgecolor="#00d4ff",
                          linewidth=1.2, linestyle="--", visible=False)
        ax.add_patch(ring); loupe_rings.append(ring)

    # helpers
    def clim():
        return st["wl"] - st["ww"] / 2, st["wl"] + st["ww"] / 2

    def redraw():
        lo, hi = clim()
        for im, data in zip(im_objs, all_datas):
            im.set_data(get_slice(data, st["plane"], st["slice"]))
            im.set_clim(lo, hi)
        redraw_circles(); fig.canvas.draw_idle()

    def redraw_circles():
        for i, ax in enumerate(axes):
            for p in circ_patches[i]: p.remove()
            circ_patches[i].clear()
            for (cx, cy, r) in st["circles"]:
                c = MplCircle((cx, cy), radius=r, fill=False, edgecolor="#ffdd00",
                               linewidth=1.5, linestyle="--", alpha=0.85)
                ax.add_patch(c); circ_patches[i].append(c)

    def update_loupe(xd, yd):
        lo, hi = clim(); r = 20
        for lax, lim, data in zip(loupe_axes, loupe_ims, all_datas):
            lim.set_data(get_slice(data, st["plane"], st["slice"]))
            lim.set_clim(lo, hi)
            lax.set_xlim(xd - r, xd + r); lax.set_ylim(yd + r, yd - r)
            lax.set_visible(True)
        for ring in loupe_rings:
            ring.center = (xd, yd); ring.set_visible(True)
        fig.canvas.draw_idle()

    def hide_loupe():
        for lax in loupe_axes: lax.set_visible(False)
        for ring in loupe_rings: ring.set_visible(False)
        fig.canvas.draw_idle()

    # widget callbacks
    sl_slice.on_changed(lambda v: (st.update({"slice": int(v)}), redraw()))
    sl_ww.on_changed(lambda v: (st.update({"ww": v}), [im.set_clim(*clim()) for im in im_objs], fig.canvas.draw_idle()))
    sl_wl.on_changed(lambda v: (st.update({"wl": v}), [im.set_clim(*clim()) for im in im_objs], fig.canvas.draw_idle()))

    def on_plane(label):
        st["plane"] = label.lower(); st["circles"] = []
        n = n_slices(orig_data.shape, st["plane"]); st["slice"] = n // 2
        sl_slice.valmax = n - 1; sl_slice.ax.set_xlim(0, n - 1)
        sl_slice.set_val(st["slice"]); redraw()

    def on_toggle(label):
        if label == "Loupe":
            st["loupe"] = not st["loupe"]
            if not st["loupe"]: hide_loupe()
        elif label == "Draw":
            st["draw_mode"] = not st["draw_mode"]

    radio.on_clicked(on_plane)
    checks.on_clicked(on_toggle)

    # mouse / key
    def on_move(event):
        if event.inaxes in axes and event.xdata and st["loupe"]:
            update_loupe(event.xdata, event.ydata)

    def on_press(event):
        if event.inaxes in axes and st["draw_mode"] and event.button == 1:
            st["drawing"] = True; st["draw_start"] = (event.xdata, event.ydata)

    def on_release(event):
        if not st["drawing"]: return
        st["drawing"] = False
        if event.inaxes not in axes or not st["draw_mode"]: return
        x0, y0 = st["draw_start"]
        r = np.hypot(event.xdata - x0, event.ydata - y0)
        if r > 2:
            st["circles"].append((x0, y0, r)); redraw_circles(); fig.canvas.draw_idle()

    def on_scroll(event):
        if event.inaxes not in axes: return
        n = n_slices(orig_data.shape, st["plane"])
        st["slice"] = max(0, min(n - 1, st["slice"] + (-1 if event.button == "up" else 1)))
        sl_slice.set_val(st["slice"])

    def on_key(event):
        n = n_slices(orig_data.shape, st["plane"])
        if   event.key in ("right", "down"): st["slice"] = min(n - 1, st["slice"] + 1)
        elif event.key in ("left",  "up"):   st["slice"] = max(0,     st["slice"] - 1)
        elif event.key == "l":
            st["loupe"] = not st["loupe"]
            if not st["loupe"]: hide_loupe()
        elif event.key == "d": st["draw_mode"] = not st["draw_mode"]
        elif event.key == "c": st["circles"] = []; redraw_circles(); fig.canvas.draw_idle(); return
        sl_slice.set_val(st["slice"]); redraw()

    fig.canvas.mpl_connect("motion_notify_event",  on_move)
    fig.canvas.mpl_connect("button_press_event",   on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("scroll_event",         on_scroll)
    fig.canvas.mpl_connect("key_press_event",      on_key)

    fig.suptitle("MedVis · Synthetic Scan Comparison",
                 color="#eaf4ff", fontsize=13, fontweight="bold", y=0.99)
    fig.text(0.5, 0.002,
             "Scroll / ←→ : slices   |   L : loupe   |   D : draw circles   |   C : clear",
             ha="center", color="#4a6070", fontsize=7)
    plt.show()


# ──────────────────────────────────────────────
#  CLI  (thin — config lives in the dict above)
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="NIfTI comparison viewer. Edit comparison_directories at the top of this file."
    )
    p.add_argument("--subject", type=int, default=0, help="0-based subject index (default: 0)")
    p.add_argument("--plane",   default="axial",     choices=["axial", "coronal", "sagittal"])
    p.add_argument("--window",  type=float, default=None, help="Override window width")
    p.add_argument("--level",   type=float, default=None, help="Override window level")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n── Discovering files ──────────────────────────")
    orig_path, panels = discover(args.subject)

    print(f"\n── Loading ────────────────────────────────────")
    print(f"  Original: {orig_path}")
    orig_img  = load_canonical(orig_path)
    orig_data = to_array(orig_img)
    print(f"  Shape: {orig_img.shape[:3]}")

    panel_datas, panel_tags = [], []
    for p in panels:
        print(f"\n  [{p['tag']}]: {p['path']}")
        img = enforce_affine(orig_img, load_canonical(p["path"]))
        panel_datas.append(to_array(img))
        panel_tags.append(p["tag"])

    ww, wl = auto_wl(orig_data)
    if args.window is not None: ww = args.window
    if args.level  is not None: wl = args.level

    n_sl    = n_slices(orig_data.shape, args.plane)
    init_sl = n_sl // 2

    print("\n── Launching viewer ───────────────────────────\n")
    launch_viewer(orig_data, panel_datas, panel_tags, ww, wl, args.plane, init_sl)


if __name__ == "__main__":
    main()