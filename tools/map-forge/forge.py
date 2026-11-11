#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WoW Map Forge - CLI Entrypoint
A lightning-fast, full-pipeline procedural map generator for World of Warcraft 3.3.5a (12340).
Generates client ADT/WDT with Big Alpha (0x04) and server .map collision files in seconds.
"""

import os
import sys
import time
import json
import struct
import tempfile
import argparse
import subprocess
import numpy as np
from pathlib import Path

# Add current dir & sibling map-converter dir to import path
FORGE_DIR = Path(__file__).resolve().parent
CONVERTER_DIR = FORGE_DIR.parent / "map-converter"
sys.path.insert(0, str(FORGE_DIR))
if CONVERTER_DIR.exists():
    sys.path.insert(0, str(CONVERTER_DIR))

from adt_builder import (
    ADTBuilder,
    build_wdt,
    build_smooth_water_layer,
    tile_to_world,
    ZEROPOINT,
    TILE_SIZE,
    CHUNK_SIZE,
)
from terrain_generators import (
    IslandGenerator,
    LakeBasinGenerator,
    HeightmapImageSampler,
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def forge_map(args):
    t_start = time.time()
    
    # Load config if specified, otherwise use arguments
    cfg = {}
    if args.config and os.path.exists(args.config):
        cfg = load_config(args.config)

    map_name = args.name or cfg.get("map_name", "custom_map")
    map_id = args.map_id if args.map_id is not None else cfg.get("map_id", 955)
    size_str = args.size or "3x3"
    preset = args.preset or cfg.get("preset", "island")
    water_enabled = not args.no_water
    water_level = args.water_level if args.water_level is not None else cfg.get("water", {}).get("level", 0.0)
    output_dir = Path(args.output or cfg.get("output_paths", {}).get("client_map_dir", f"./output/{map_name}")).resolve()
    server_dir = Path(args.server_maps or cfg.get("output_paths", {}).get("server_maps_dir", "./output/maps")).resolve()
    fmt = getattr(args, "format", "wotlk")
    if fmt in ("retail", "both") and not getattr(args, "listfile", None):
        raise SystemExit("--listfile is required when --format is 'retail' or 'both'")
    if fmt in ("retail", "both") and not CONVERTER_DIR.exists():
        raise SystemExit(f"map-converter package not found at {CONVERTER_DIR}")

    tmp_wotlk_ctx = None
    if fmt == "retail":
        tmp_wotlk_ctx = tempfile.TemporaryDirectory(prefix=f"forge_wotlk_{map_name}_")
        wotlk_dir = Path(tmp_wotlk_ctx.name)
        retail_dir = Path(args.output_retail).resolve() if getattr(args, "output_retail", None) else output_dir
    elif fmt == "both":
        wotlk_dir = output_dir / "wotlk"
        retail_dir = Path(args.output_retail).resolve() if getattr(args, "output_retail", None) else (output_dir / "retail")
    else:
        wotlk_dir = output_dir
        retail_dir = None

    # Parse grid size (e.g. 3x3 -> start_tile 31..33)
    parts = size_str.lower().split("x")
    grid_w = int(parts[0])
    grid_h = int(parts[1]) if len(parts) > 1 else grid_w
    
    start_tx = args.start_x if args.start_x is not None else (32 - grid_w // 2)
    start_ty = args.start_y if args.start_y is not None else (32 - grid_h // 2)
    
    tx_list = list(range(start_tx, start_tx + grid_w))
    ty_list = list(range(start_ty, start_ty + grid_h))
    total_tiles = len(tx_list) * len(ty_list)

    print("======================================================================")
    print(f"  WoW Map Forge - Procedural World Generator (format={fmt.upper()})")
    print(f"  Map: {map_name} (ID: {map_id}) | Grid: {grid_w}x{grid_h} tiles ({total_tiles} ADTs)")
    print(f"  Preset: {preset.upper()} | Water: {'Enabled (' + str(water_level) + 'm)' if water_enabled else 'Disabled'}")
    print(f"  Client output: {retail_dir if fmt == 'retail' else output_dir}")
    print(f"  Server output: {server_dir}")
    print("======================================================================")

    wotlk_dir.mkdir(parents=True, exist_ok=True)
    if retail_dir is not None:
        retail_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_server_maps:
        server_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup Terrain Generator
    if args.heightmap and os.path.exists(args.heightmap):
        print(f"\n[1/4] Loading custom heightmap image: {args.heightmap}")
        # World coordinate bounds across generated tiles
        max_wx = ZEROPOINT - min(ty_list) * TILE_SIZE
        min_wx = ZEROPOINT - (max(ty_list) + 1) * TILE_SIZE
        sampler = HeightmapImageSampler(args.heightmap, bounds_min_xy=min_wx, bounds_max_xy=max_wx, min_z=args.min_height, max_z=args.peak_height)
    elif preset == "lake":
        print(f"\n[1/4] Initializing procedural Lake Basin generator (peak={args.peak_height}m)...")
        sampler = LakeBasinGenerator(peak_height=args.peak_height, water_level=water_level)
    else:
        print(f"\n[1/4] Initializing procedural Island generator (radius={args.island_radius}m, peak={args.peak_height}m)...")
        sampler = IslandGenerator(island_radius=args.island_radius, peak_height=args.peak_height, water_level=water_level)

    # 2. Textures Palette (config override or standard high-res textures)
    cfg_pal = cfg.get("palette", {})
    palette = [
        cfg_pal.get("base_grass", "tileset/expansion05/nagrand/6ng_grass01_512.blp"),
        cfg_pal.get("shore_sand", "tileset/silverpine/silverpineshore.blp"),
        cfg_pal.get("dirt_soil", "tileset/alteracmtns/alteracdirtbase02.blp"),
        cfg_pal.get("mountain_rock", "tileset/expansion07/general/8war_rock01_512.blp"),
    ]

    # Precompute local chunk offsets
    step_vert = CHUNK_SIZE / 8.0
    step_pixel = CHUNK_SIZE / 64.0

    v_local_x = np.zeros(145, dtype=float)
    v_local_y = np.zeros(145, dtype=float)
    for r in range(9):
        for c in range(9):
            idx = r * 17 + c
            v_local_x[idx] = r * step_vert
            v_local_y[idx] = c * step_vert
    for r in range(8):
        for c in range(8):
            idx = r * 17 + 9 + c
            v_local_x[idx] = (r + 0.5) * step_vert
            v_local_y[idx] = (c + 0.5) * step_vert

    px_indices, py_indices = np.meshgrid(np.arange(64), np.arange(64))
    p_local_x = (py_indices + 0.5) * step_pixel
    p_local_y = (px_indices + 0.5) * step_pixel

    # Load verified MH2O template bytes
    mh2o_template_path = FORGE_DIR / "mh2o_template.bin"
    mh2o_template = None
    if water_enabled and mh2o_template_path.exists():
        with open(mh2o_template_path, "rb") as f:
            mh2o_template = f.read()

    # 3. Generate WDT
    print("\n[2/4] Assembling .WDT header with Big Alpha (0x04)...")
    wdt_path = wotlk_dir / f"{map_name}.wdt"
    build_wdt(str(wdt_path), tx_list, ty_list)
    print(f"  -> Created: {wdt_path.name}")

    # 4. Generate ADT Tiles
    print(f"\n[3/4] Generating {total_tiles} ADT tiles with Big Alpha and analytical lighting...")
    tile_count = 0
    adt_files = []

    for tx in tx_list:
        for ty in ty_list:
            tile_count += 1
            t0 = time.time()
            adt = ADTBuilder(tx, ty, palette)

            tile_wx_top = ZEROPOINT - ty * TILE_SIZE
            tile_wy_left = ZEROPOINT - tx * TILE_SIZE

            # Seamless MH2O Water Layer
            if water_enabled and mh2o_template:
                adt.mh2o = build_smooth_water_layer(
                    mh2o_template, tile_wx_top, tile_wy_left, sampler.get_elevation, water_level=water_level
                )

            # Process 256 MCNK Chunks
            for iy in range(16):
                for ix in range(16):
                    c = adt.chunks[iy][ix]
                    c.pos_x = tile_wx_top - iy * CHUNK_SIZE
                    c.pos_y = tile_wy_left - ix * CHUNK_SIZE
                    c.areaid = 193  # Standard zone

                    center_wx = c.pos_x - CHUNK_SIZE * 0.5
                    center_wy = c.pos_y - CHUNK_SIZE * 0.5
                    c.pos_z = float(sampler.get_elevation(center_wx, center_wy))

                    # 145 Vertex Heights
                    vwx_arr = c.pos_x - v_local_x
                    vwy_arr = c.pos_y - v_local_y
                    vh_arr = sampler.get_elevation(vwx_arr, vwy_arr)
                    c.mcvt = [float(h - c.pos_z) for h in vh_arr]

                    # Analytical Normal Vectors (MCNR) via central differences
                    eps = 0.4
                    h_xp = sampler.get_elevation(vwx_arr + eps, vwy_arr)
                    h_xm = sampler.get_elevation(vwx_arr - eps, vwy_arr)
                    h_yp = sampler.get_elevation(vwx_arr, vwy_arr + eps)
                    h_ym = sampler.get_elevation(vwx_arr, vwy_arr - eps)
                    nx = -(h_xp - h_xm) / (2.0 * eps)
                    ny = -(h_yp - h_ym) / (2.0 * eps)
                    nz = np.ones_like(nx)
                    lengths = np.hypot(np.hypot(nx, ny), nz)

                    nx_b = np.round(127.0 * nx / lengths).astype(np.int8)
                    nz_b = np.round(127.0 * nz / lengths).astype(np.int8)
                    ny_b = np.round(127.0 * ny / lengths).astype(np.int8)

                    mcnr_bytes = bytearray(448)
                    for vi in range(145):
                        struct.pack_into("<3b", mcnr_bytes, vi * 3, int(nx_b[vi]), int(nz_b[vi]), int(ny_b[vi]))
                    c.mcnr = mcnr_bytes

                    # 4 Texture Layers setup
                    c.layers = [
                        [0, 0, 0, 0],              # Grass base
                        [1, 0x100, 0, 0],          # Sand
                        [2, 0x100, 4096, 0],       # Dirt
                        [3, 0x100, 8192, 0],       # Rock
                    ]

                    # 64x64 Alpha Maps calculation
                    p_wx_grid = c.pos_x - p_local_x
                    p_wy_grid = c.pos_y - p_local_y
                    h_grid = sampler.get_elevation(p_wx_grid, p_wy_grid)

                    h_xp_g = sampler.get_elevation(p_wx_grid + 0.5, p_wy_grid)
                    h_xm_g = sampler.get_elevation(p_wx_grid - 0.5, p_wy_grid)
                    h_yp_g = sampler.get_elevation(p_wx_grid, p_wy_grid + 0.5)
                    h_ym_g = sampler.get_elevation(p_wx_grid, p_wy_grid - 0.5)
                    slope_grid = np.hypot(h_xp_g - h_xm_g, h_yp_g - h_ym_g)
                    nz_grid = 1.0 / np.sqrt(1.0 + slope_grid**2)

                    # 1. Beach Sand Layer (along water waterline)
                    t_sand = np.clip((h_grid - (water_level - 2.0)) / 4.5, 0.0, 1.0)
                    alpha_sand = 255.0 * (1.0 - t_sand * t_sand * (3.0 - 2.0 * t_sand))

                    # 2. Dirt / Plateau Layer
                    t_dirt = np.clip((h_grid - (water_level + 2.0)) / 10.0, 0.0, 1.0)
                    alpha_dirt = 120.0 * t_dirt * (1.0 - np.clip((h_grid - 30.0) / 20.0, 0.0, 1.0))

                    # 3. Granite Rock Layer (steep slopes and high mountain altitudes)
                    rock_slope = np.clip((0.85 - nz_grid) / 0.15, 0.0, 1.0)
                    rock_alt = np.clip((h_grid - 45.0) / 35.0, 0.0, 1.0)
                    alpha_rock = 255.0 * np.maximum(rock_slope, rock_alt)

                    c.alpha_maps = {
                        1: bytearray(np.clip(alpha_sand, 0, 255).astype(np.uint8).flatten()),
                        2: bytearray(np.clip(alpha_dirt, 0, 255).astype(np.uint8).flatten()),
                        3: bytearray(np.clip(alpha_rock, 0, 255).astype(np.uint8).flatten()),
                    }

            out_adt = wotlk_dir / f"{map_name}_{tx}_{ty}.adt"
            adt.save(str(out_adt))
            adt_files.append(out_adt)
            dt = time.time() - t0
            print(f"  [{tile_count}/{total_tiles}] Built {out_adt.name} in {dt:.2f}s")

    # 5. Fast Server .map Conversion
    if not args.no_server_maps:
        print(f"\n[4/4] Converting {len(adt_files)} ADTs to server .map collision files...")
        converter_script = FORGE_DIR / "server_map_converter.py"
        for adt_p in adt_files:
            bn = adt_p.stem
            parts = bn.split("_")
            tx, ty = int(parts[-2]), int(parts[-1])
            out_map = server_dir / f"{map_id:03d}{ty:02d}{tx:02d}.map"
            subprocess.run([sys.executable, str(converter_script), str(adt_p), str(out_map)], stdout=subprocess.DEVNULL)
            print(f"  -> Converted: {out_map.name}")

    # 6. Optional Modern Retail / Shadowlands Split-ADT Conversion
    if fmt in ("retail", "both") and retail_dir is not None:
        from listfile_resolver import ListfileResolver
        from wotlk_to_retail import convert_tiles

        alias_map = None
        if getattr(args, "alias", None):
            alias_map = json.loads(Path(args.alias).read_text(encoding="utf-8"))
        lf_paths = [Path(p) for p in args.listfile]
        print(f"\n[Retail] Converting {len(adt_files)} forged tiles -> {args.target} ({retail_dir})...")
        resolver = ListfileResolver(lf_paths, alias_map=alias_map, enable_fuzzy=True)
        convert_tiles(
            input_paths=adt_files,
            output_dir=retail_dir,
            map_name=map_name,
            resolver=resolver,
            big_alpha_override=True,
            target_version=args.target,
            missing_policy=args.missing_policy,
            compress_alpha=args.compress_alpha,
            height_textures=not args.no_height_textures,
            emit_mtxp=args.emit_mtxp,
            with_obj1=not args.no_obj1,
            with_lod=not args.no_lod,
            synth_mfbo=True,
            fix_uids=True,
            wdt_mode=args.wdt_mode,
            base_fdid=args.base_fdid,
            existing_wdt_path=wdt_path,
            existing_wdl_path=None,
            assets_dir=None,
            noggitproj_dir=(retail_dir if args.noggitproj else None),
            client_path=args.client_path or "",
            map_id=map_id,
        )

    if tmp_wotlk_ctx is not None:
        tmp_wotlk_ctx.cleanup()

    elapsed = round(time.time() - t_start, 2)
    print("\n======================================================================")
    print(f"  SUCCESS! World map '{map_name}' forged in {elapsed}s.")
    if fmt == "both":
        print(f"  WotLK client files:  {wotlk_dir}")
        print(f"  Retail client files: {retail_dir}")
    elif fmt == "retail":
        print(f"  Retail client files: {retail_dir}")
    else:
        print(f"  Client files: {wotlk_dir}")
    print(f"  Server files: {server_dir}")
    print(f"  Ready to open in Noggit Red without crashes!")
    print("======================================================================")


def main():
    parser = argparse.ArgumentParser(
        description="WoW Map Forge - Procedural world generator for WotLK 3.3.5a and Modern Retail / Shadowlands"
    )
    parser.add_argument("--name", type=str, default="custom_map", help="Internal map name (e.g. emerald_island)")
    parser.add_argument("--map-id", type=int, default=955, help="Map ID from Map.dbc (default: 955)")
    parser.add_argument("--size", type=str, default="3x3", help="Grid size in tiles, e.g. 1x1, 2x2, 3x3, 4x4 (default: 3x3)")
    parser.add_argument("--preset", type=str, choices=["island", "lake"], default="island", help="Terrain preset (default: island)")
    parser.add_argument("--heightmap", type=str, help="Optional grayscale image file to generate terrain from")
    parser.add_argument("--peak-height", type=float, default=160.0, help="Maximum mountain peak elevation in meters (default: 160.0)")
    parser.add_argument("--island-radius", type=float, default=380.0, help="Island shoreline radius in meters (default: 380.0)")
    parser.add_argument("--min-height", type=float, default=-20.0, help="Minimum height for heightmaps (default: -20.0)")
    parser.add_argument("--water-level", type=float, default=0.0, help="Water surface height Z (default: 0.0)")
    parser.add_argument("--no-water", action="store_true", help="Disable MH2O liquid generation completely")
    parser.add_argument("--no-server-maps", action="store_true", help="Skip converting to server .map files")
    parser.add_argument("--output", type=str, help="Client ADT output directory")
    parser.add_argument("--output-retail", type=str, help="Optional separate Retail output directory when --format both")
    parser.add_argument("--server-maps", type=str, help="Server .map output directory")
    parser.add_argument("--start-x", type=int, help="Starting tile X (default: centered around 32)")
    parser.add_argument("--start-y", type=int, help="Starting tile Y (default: centered around 32)")
    parser.add_argument("--config", type=str, help="Optional JSON config file")
    parser.add_argument("--format", choices=["wotlk", "retail", "both"], default="wotlk", help="Output map format (default: wotlk)")
    parser.add_argument("--listfile", nargs="+", help="Retail CSV listfile(s) (required for --format retail/both)")
    parser.add_argument("--alias", type=str, help="Optional JSON mapping custom paths -> Retail paths or FDIDs")
    parser.add_argument("--target", choices=["retail", "shadowlands"], default="retail", help="Retail split ADT target (default: retail)")
    parser.add_argument("--wdt-mode", choices=["path", "maid"], default="path", help="WDT addressing mode (default: path)")
    parser.add_argument("--base-fdid", type=int, default=5000000, help="Starting FileDataID when --wdt-mode maid")
    parser.add_argument("--no-lod", action="store_true", help="Skip generating _lod.adt terrain LOD files in Retail mode")
    parser.add_argument("--no-obj1", action="store_true", help="Skip generating _obj1.adt in Retail mode")
    parser.add_argument("--no-height-textures", action="store_true", help="Do not resolve *_h.blp MHID/MTXP")
    parser.add_argument("--compress-alpha", action="store_true", help="Emit RLE-compressed MCAL (0x300) in _tex0")
    parser.add_argument("--emit-mtxp", action="store_true", help="Always write MTXP chunk in _tex0")
    parser.add_argument("--missing-policy", choices=["skip", "zero", "fail"], default="skip")
    parser.add_argument("--noggitproj", action="store_true", help="Write ShadowWorld .noggitproj into Retail output folder")
    parser.add_argument("--client-path", type=str, default="", help="Client path stored in .noggitproj")

    args = parser.parse_args()
    forge_map(args)


if __name__ == "__main__":
    main()
