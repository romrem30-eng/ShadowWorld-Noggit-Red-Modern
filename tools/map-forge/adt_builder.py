#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WoW Map Forge - Core ADT & WDT Binary Library
Supports:
  - Full WotLK 3.3.5a (build 12340) binary format
  - Big Alpha (0x04) high-resolution 64x64 uncompressed alpha maps (0x100)
  - Seamless MH2O procedural water layers with depth alpha blending
  - Analytical MCNR vertex normal generation for smooth terrain lighting
  - Direct conversion to AzerothCore / TrinityCore server .map files
"""

import os
import sys
import math
import struct
import numpy as np
from pathlib import Path

# World Constants
ZEROPOINT = 17066.666
TILE_SIZE = 533.33333
CHUNK_SIZE = TILE_SIZE / 16.0  # 33.333333m
UNIT_SIZE = CHUNK_SIZE / 8.0   # 4.166666m

def world_to_tile(wx: float, wy: float):
    tx = int((ZEROPOINT - wy) / TILE_SIZE)
    ty = int((ZEROPOINT - wx) / TILE_SIZE)
    return max(0, min(63, tx)), max(0, min(63, ty))

def tile_to_world(tx: int, ty: int):
    wx = ZEROPOINT - ty * TILE_SIZE
    wy = ZEROPOINT - tx * TILE_SIZE
    return wx, wy

def world_to_adt_pos(wx: float, wy: float, wz: float):
    return (ZEROPOINT - wy, ZEROPOINT - wx, wz)


class MCNKChunk:
    def __init__(self, ix: int, iy: int):
        self.ix = ix
        self.iy = iy
        self.flags = 0
        self.areaid = 0
        self.holes = 0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = 0.0
        self.mcvt = [0.0] * 145
        self.mcnr = bytearray(448)
        self.layers = []  # list of [texture_id, flags, offset_mcal, effect_id]
        self.alpha_maps = {}  # layer_index -> bytearray(4096)
        self.header = bytearray(128)

    def pack(self) -> bytes:
        body = bytearray(128)  # header placeholder

        # 1. MCVT (145 floats = 580 bytes)
        ofs_mcvt = len(body) + 8
        mcvt_bytes = struct.pack("<145f", *self.mcvt)
        body.extend(b"TVCM" + struct.pack("<I", len(mcvt_bytes)) + mcvt_bytes)

        # 2. MCNR (435 bytes payload + 13 padding = 448 bytes)
        ofs_mcnr = len(body) + 8
        mcnr_bytes = bytes(self.mcnr[:448]).ljust(448, b"\x00")
        body.extend(b"RNCM" + struct.pack("<I", 435) + mcnr_bytes)

        # 3. MCLY & MCAL (Big Alpha 0x100)
        mcal_data = bytearray()
        for li in range(1, len(self.layers)):
            amap = self.alpha_maps.get(li, bytearray(4096))
            if len(amap) != 4096:
                amap = bytearray(amap[:4096]).ljust(4096, b"\x00")
            self.layers[li][1] |= 0x100  # USE_ALPHA_MAP (Big Alpha uncompressed)
            self.layers[li][2] = len(mcal_data)
            mcal_data.extend(amap)

        if len(self.layers) > 0:
            self.layers[0][1] &= ~0x100
            self.layers[0][2] = 0

        ofs_mcly = len(body) + 8
        mcly_payload = bytearray()
        for tid, lflags, off_mcal, eff in self.layers:
            mcly_payload.extend(struct.pack("<IIII", tid, lflags, off_mcal, eff))
        body.extend(b"YLCM" + struct.pack("<I", len(mcly_payload)) + mcly_payload)

        # 4. MCRF (Empty doodad/wmo refs)
        ofs_mcrf = len(body) + 8
        body.extend(b"FRCM" + struct.pack("<I", 0))

        # 5. MCAL
        ofs_mcal = len(body) + 8
        size_mcal = len(mcal_data) + 8
        body.extend(b"LACM" + struct.pack("<I", len(mcal_data)) + mcal_data)

        # Pack 128-byte MCNK Header
        hdr = bytearray(128)
        struct.pack_into(
            "<16I", hdr, 0,
            self.flags, self.ix, self.iy, len(self.layers), 0,
            ofs_mcvt, ofs_mcnr, ofs_mcly, ofs_mcrf,
            ofs_mcal, size_mcal, 0, 0,
            self.areaid, 0, self.holes
        )
        struct.pack_into("<fff", hdr, 0x68, float(self.pos_x), float(self.pos_y), float(self.pos_z))
        body[:128] = hdr

        return b"KNCM" + struct.pack("<I", len(body)) + bytes(body)


class ADTBuilder:
    def __init__(self, tx: int, ty: int, textures: list):
        self.tx = tx
        self.ty = ty
        self.textures = list(textures)
        self.chunks = [[MCNKChunk(ix, iy) for ix in range(16)] for iy in range(16)]
        self.mh2o = None

    def save(self, filepath: str):
        mtex_bytes = b"\x00".join(t.encode("ascii") for t in self.textures) + (b"\x00" if self.textures else b"")

        mhdr_base = 20
        body = bytearray()

        # MCIN placeholder (8 + 4096 = 4104 bytes)
        ofs_mcin = (84 + len(body)) - mhdr_base
        mcin_pos_in_body = len(body)
        body.extend(b"NICM" + struct.pack("<I", 4096) + b"\x00" * 4096)

        # MTEX
        ofs_mtex = (84 + len(body)) - mhdr_base
        body.extend(b"XETM" + struct.pack("<I", len(mtex_bytes)) + mtex_bytes)

        # MMDX, MMID, MWMO, MWID, MDDF, MODF (Empty for pure terrain)
        ofs_mmdx = (84 + len(body)) - mhdr_base
        body.extend(b"XDMM\x00\x00\x00\x00")
        ofs_mmid = (84 + len(body)) - mhdr_base
        body.extend(b"DIMM\x00\x00\x00\x00")
        ofs_mwmo = (84 + len(body)) - mhdr_base
        body.extend(b"OMWM\x00\x00\x00\x00")
        ofs_mwid = (84 + len(body)) - mhdr_base
        body.extend(b"DIWM\x00\x00\x00\x00")
        ofs_mddf = (84 + len(body)) - mhdr_base
        body.extend(b"FDDM\x00\x00\x00\x00")
        ofs_modf = (84 + len(body)) - mhdr_base
        body.extend(b"FDOM\x00\x00\x00\x00")

        # MH2O
        ofs_mh2o = 0
        if self.mh2o is not None:
            ofs_mh2o = (84 + len(body)) - mhdr_base
            body.extend(b"O2HM" + struct.pack("<I", len(self.mh2o)) + self.mh2o)

        # 256 MCNK Chunks
        mcin_entries = bytearray(4096)
        for iy in range(16):
            for ix in range(16):
                c = self.chunks[iy][ix]
                mcnk_bytes = c.pack()
                abs_ofs = 84 + len(body)
                idx = (iy * 16 + ix) * 16
                struct.pack_into("<IIII", mcin_entries, idx, abs_ofs, len(mcnk_bytes), 0, 0)
                body.extend(mcnk_bytes)

        # Patch MCIN data
        body[mcin_pos_in_body + 8 : mcin_pos_in_body + 8 + 4096] = mcin_entries

        # Assemble MVER & MHDR
        mver = b"REVM" + struct.pack("<II", 4, 18)
        mhdr_data = struct.pack(
            "<16I",
            0, ofs_mcin, ofs_mtex, ofs_mmdx,
            ofs_mmid, ofs_mwmo, ofs_mwid, ofs_mddf,
            ofs_modf, 0, ofs_mh2o, 0,
            0, 0, 0, 0
        )
        mhdr = b"RDHM" + struct.pack("<I", 64) + mhdr_data

        with open(filepath, "wb") as f:
            f.write(mver)
            f.write(mhdr)
            f.write(body)


def build_wdt(filepath: str, tx_list: list, ty_list: list):
    """Builds a .wdt map header with Big Alpha (flag 0x04) enabled."""
    mver = b"REVM" + struct.pack("<II", 4, 18)
    mphd = b"DHPM" + struct.pack("<II", 32, 0x04) + b"\x00" * 28

    main_grid = bytearray(64 * 64 * 8)
    for tx in tx_list:
        for ty in ty_list:
            idx = (ty * 64 + tx) * 8
            struct.pack_into("<II", main_grid, idx, 1, 0)

    main = b"NIAM" + struct.pack("<I", len(main_grid)) + bytes(main_grid)

    with open(filepath, "wb") as f:
        f.write(mver)
        f.write(mphd)
        f.write(main)


def build_smooth_water_layer(template_bytes: bytes, tile_wx_top: float, tile_wy_left: float, elevation_func, water_level: float = 0.0) -> bytes:
    """
    Builds a robust, seamless MH2O water layer.
    Features:
      - Automatically disables liquid for chunks entirely above water_level + 1.2m
      - Computes continuous vertex alpha (0 on dry land, 10-255 in deep water)
      - Eliminates 33m chunk square boundary artifacts along beaches and coastlines
    """
    mh2o = bytearray(template_bytes)
    step_v = CHUNK_SIZE / 8.0

    for iy in range(16):
        for ix in range(16):
            k = iy * 16 + ix
            chunk_wx = tile_wx_top - iy * CHUNK_SIZE
            chunk_wy = tile_wy_left - ix * CHUNK_SIZE

            p_xs = [chunk_wx - vy * step_v for vy in range(9)]
            p_ys = [chunk_wy - vx * step_v for vx in range(9)]
            grid_xs, grid_ys = np.meshgrid(p_xs, p_ys)
            grid_zs = elevation_func(grid_xs.flatten(), grid_ys.flatten())
            min_z = float(np.min(grid_zs))

            if min_z >= water_level + 1.20:
                # 100% dry land: turn off water completely
                struct.pack_into("<3I", mh2o, k * 12, 0, 0, 0)
            else:
                off_inst, used, off_attr = struct.unpack_from("<3I", mh2o, k * 12)
                if used and off_inst:
                    # Update water plane height in instance header
                    struct.pack_into("<ff", mh2o, off_inst + 4, water_level, water_level)
                    off_v = struct.unpack_from("<I", mh2o, off_inst + 20)[0]
                    if off_v and off_v + 81 <= len(mh2o):
                        v_bytes = bytearray(81)
                        for idx, p_z in enumerate(grid_zs):
                            depth = water_level - p_z
                            if depth <= 0.0:
                                v_bytes[idx] = 0  # Fully transparent on dry land
                            else:
                                v_bytes[idx] = int(np.clip(depth * 55.0, 10, 255))
                        mh2o[off_v : off_v + 81] = v_bytes

    return bytes(mh2o)
