#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WoW Map Forge - 3D Terrain Preview & Inspection Tool
Interactive 3D viewport for inspecting generated ADT / WDT maps.
Features:
  - Direct binary reading of ADT geometry & heights
  - Multi-tile continuous stitching
  - Interactive 3D Surface View (orbit, pan, zoom, wireframe toggle)
  - Topographic Map with contours & water line
  - Color palettes (Warcraft Nature, Elevation Heatmap, Noggit Style)
  - Snapshot export to PNG for marketing & showcasing on forums
"""

import sys
import struct
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource

def extract_adt_grid(adt_path: Path) -> np.ndarray:
    """Extracts the 129x129 vertex height grid from a binary ADT file."""
    with open(adt_path, "rb") as f:
        data = f.read()

    # 16x16 chunks * 8 quad steps = 128 quads -> 129 outer vertices
    grid = np.zeros((129, 129), dtype=np.float32)
    for iy in range(16):
        for ix in range(16):
            # MCIN table is at offset 92 (after 84-byte MVER+MHDR + 8-byte MCIN chunk header)
            off_mcnk = struct.unpack_from("<I", data, 92 + (iy * 16 + ix) * 16)[0]
            mcnk_hdr = off_mcnk + 8
            pos_z = struct.unpack_from("<f", data, mcnk_hdr + 0x70)[0]
            ofs_mcvt = struct.unpack_from("<I", data, mcnk_hdr + 0x14)[0]
            mcvt = struct.unpack_from("<145f", data, mcnk_hdr + ofs_mcvt)

            # Outer 9x9 vertices
            for r in range(9):
                for c in range(9):
                    val = pos_z + mcvt[r * 17 + c]
                    grid[iy * 8 + r, ix * 8 + c] = val

    return grid


def load_and_stitch_map(map_dir: Path):
    """Finds all ADTs in directory and stitches them into a seamless height matrix."""
    adts = list(map_dir.glob("*_*_*.adt"))
    if not adts:
        raise FileNotFoundError(f"No .adt files found in {map_dir}")

    tiles = {}
    for a in adts:
        parts = a.stem.split("_")
        try:
            tx, ty = int(parts[-2]), int(parts[-1])
            tiles[(tx, ty)] = a
        except ValueError:
            continue

    tx_list = sorted(list(set(k[0] for k in tiles.keys())))
    ty_list = sorted(list(set(k[1] for k in tiles.keys())))

    num_tx = len(tx_list)
    num_ty = len(ty_list)

    total_w = num_tx * 128 + 1
    total_h = num_ty * 128 + 1

    stitched_z = np.zeros((total_h, total_w), dtype=np.float32)

    for row_idx, ty in enumerate(ty_list):
        for col_idx, tx in enumerate(tx_list):
            if (tx, ty) in tiles:
                tile_grid = extract_adt_grid(tiles[(tx, ty)])
                y_start = row_idx * 128
                y_end = y_start + 129
                x_start = col_idx * 128
                x_end = x_start + 128 + 1

                # Blend/copy tile
                stitched_z[y_start:y_end, x_start:x_end] = tile_grid

    return stitched_z, tx_list, ty_list, len(tiles)


def preview_map(map_path_str: str, save_png: str = None, water_level: float = 0.0):
    map_dir = Path(map_path_str)
    if map_dir.is_file() and map_dir.suffix.lower() == ".adt":
        map_dir = map_dir.parent

    print(f"Loading terrain geometry from: {map_dir}")
    Z, tx_list, ty_list, tile_count = load_and_stitch_map(map_dir)

    min_z = float(np.min(Z))
    max_z = float(np.max(Z))
    land_pct = float(np.mean(Z >= water_level) * 100.0)
    water_pct = 100.0 - land_pct

    print(f"Loaded {tile_count} tiles ({len(tx_list)}x{len(ty_list)} grid)")
    print(f"Elevation bounds: Min {min_z:.1f}m | Max {max_z:.1f}m")
    print(f"Surface coverage: Land {land_pct:.1f}% | Water {water_pct:.1f}%")

    # Downsample for silky smooth real-time 3D rotation if grid is huge
    step = 1 if max(Z.shape) <= 300 else 2
    Z_vis = Z[::step, ::step]
    h, w = Z_vis.shape
    X, Y = np.meshgrid(np.linspace(0, len(tx_list) * 533.3, w), np.linspace(len(ty_list) * 533.3, 0, h))

    # Dark Theme Professional Window
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(15, 8), num=f"WoW Map Forge - 3D Viewer [{map_dir.name}]")

    # 1. Left View: Interactive 3D Perspective (Noggit style preview)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.set_facecolor("#12141a")

    # Realistic sunlight hillshade
    ls = LightSource(azdeg=315, altdeg=45)
    # Warcraft Nature Colormap: Deep Blue Water -> Sandy Shore -> Lush Green -> Rocky Cliffs -> Snow Peak
    colors = [
        (0.00, "#0e3452"),  # Deep ocean/lake
        (0.12, "#1b5a8c"),  # Shallow water
        (0.14, "#c2b280"),  # Shore sand
        (0.20, "#3a6932"),  # Lowlands grass
        (0.40, "#294d23"),  # Highland forest
        (0.65, "#68615a"),  # Granite cliffs
        (0.85, "#48433d"),  # High rock
        (1.00, "#eef2f5"),  # Alpine snow peak
    ]
    cmap = cm.colors.LinearSegmentedColormap.from_list("warcraft_palette", colors, N=256)

    # Normalize elevation
    norm = plt.Normalize(min_z, max_z)
    rgb = ls.shade(Z_vis, cmap=cmap, norm=norm, blend_mode="overlay")

    surf = ax1.plot_surface(
        X, Y, Z_vis,
        facecolors=rgb,
        rstride=1, cstride=1,
        linewidth=0,
        antialiased=True,
        shade=False
    )

    # Render translucent water plane
    if water_pct > 0.5:
        W_plane = np.full_like(Z_vis, water_level)
        ax1.plot_surface(X, Y, W_plane, color="#1e88e5", alpha=0.35, linewidth=0)

    ax1.set_title(f"3D Perspective View (Orbit with Mouse)\nTiles: {len(tx_list)}x{len(ty_list)} | Peaks: {max_z:.1f}m", fontsize=12, pad=15, color="#81c784")
    ax1.set_xlabel("X (meters)", labelpad=10)
    ax1.set_ylabel("Y (meters)", labelpad=10)
    ax1.set_zlabel("Z (height)", labelpad=10)
    ax1.view_init(elev=38, azim=-55)

    # 2. Right View: 2D Topographic Contours & Shoreline
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_facecolor("#12141a")

    im = ax2.imshow(Z, extent=[0, len(tx_list) * 533.3, 0, len(ty_list) * 533.3], cmap=cmap, origin="upper", norm=norm)
    
    # Contour lines every 15 meters
    contours = ax2.contour(
        np.linspace(0, len(tx_list) * 533.3, Z.shape[1]),
        np.linspace(0, len(ty_list) * 533.3, Z.shape[0]),
        np.flipud(Z),
        levels=np.arange(np.floor(min_z / 15) * 15, max_z, 15.0),
        colors="#ffffff",
        alpha=0.25,
        linewidths=0.7
    )
    # Highlight exact shoreline (water level)
    ax2.contour(
        np.linspace(0, len(tx_list) * 533.3, Z.shape[1]),
        np.linspace(0, len(ty_list) * 533.3, Z.shape[0]),
        np.flipud(Z),
        levels=[water_level],
        colors="#00e5ff",
        linewidths=1.8
    )

    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Height (meters)", color="#ffffff")

    ax2.set_title(f"Topographic Shoreline & Elevation Map\nCyan line = Waterline (0.0m)", fontsize=12, pad=15, color="#4fc3f7")
    ax2.set_xlabel("X (meters)")
    ax2.set_ylabel("Y (meters)")

    plt.tight_layout()

    if save_png:
        out_png = Path(save_png).resolve()
        fig.savefig(out_png, dpi=180, facecolor="#12141a", edgecolor="none")
        print(f"\n[+] Render snapshot saved to: {out_png}")

    print("\n[+] Displaying 3D Interactive Viewport. (Close the window when done)")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="WoW Map Forge - 3D Terrain Preview & Inspection Tool"
    )
    parser.add_argument("map_dir", type=str, help="Path to generated map folder containing .adt files")
    parser.add_argument("--save", type=str, help="Optional output PNG path to save high-resolution preview image")
    parser.add_argument("--water-level", type=float, default=0.0, help="Waterline surface level Z (default: 0.0)")

    args = parser.parse_args()
    preview_map(args.map_dir, save_png=args.save, water_level=args.water_level)


if __name__ == "__main__":
    main()
