#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retail OBJ1, LOD (_lod.adt), WDL, and WDT builders matching ShadowWorld validators."""

from __future__ import annotations

import math
import struct
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from retail_chunks import (
    CHUNK_SIZE,
    ZEROPOINT,
    compress_holes_high_to_low,
    iter_chunks,
    make_chunk,
)

GRID_OUTER = 129 * 129
GRID_COUNT = 129 * 129 + 128 * 128


def build_obj1_bytes(
    doodad_rows: List[Tuple[bytes, Tuple[Tuple[float, float, float], Tuple[float, float, float], float], int]],
    wmo_rows: List[Tuple[bytes, Tuple[Tuple[float, float, float], Tuple[float, float, float], float], int]],
    mwdr_bytes: bytes = b"",
    mwds_bytes: bytes = b"",
    include_mldl: bool = True,
) -> bytes:
    """
    Assemble a Retail/Shadowlands _obj1.adt file validated by RetailObjectTables::validate
    and compatible with ShadowlandsADTCodec / RetailObjectTransfer.

    doodad_rows: [(mddf_36B, ((min3),(max3), radius), mldb_byte), ...]
    wmo_rows:    [(mlmd_40B, ((min3),(max3), radius), mlmb_byte), ...]
    """
    d_sorted = sorted(doodad_rows, key=lambda r: -r[1][2])
    w_sorted = sorted(wmo_rows, key=lambda r: -r[1][2])

    d_cnt = len(d_sorted)
    w_cnt = len(w_sorted)

    mver = make_chunk("MVER", struct.pack("<I", 18))

    # MLFD: [offset_d[3], count_d[3], offset_w[3], count_w[3]]
    # Put all placements in level 0 (indices 0..cnt), and mirror count in level 2 offset
    # if needed? No, standard level 0: off=[0,0,0], cnt=[D,0,0], off_w=[0,0,0], cnt_w=[W,0,0].
    # Note: wait - ShadowlandsADTCodec legacy fallback expects counts[8]==counts[9], while Retail
    # uses RetailObjectTransfer::write directly. Check: for Retail + RetailObjectTables,
    # [0,0,0, d_cnt, 0, 0,  0,0,0, w_cnt, 0, 0] puts every entry cleanly in level 0.
    mlfd_payload = struct.pack(
        "<12I",
        0, 0, 0, d_cnt, 0, 0,
        0, 0, 0, w_cnt, 0, 0,
    )

    mldd_buf = bytearray()
    mldx_buf = bytearray()
    mldb_buf = bytearray()
    mldl_buf = bytearray()
    for entry36, (bmin, bmax, radius), mldb_val in d_sorted:
        mldd_buf.extend(entry36)
        mldx_buf.extend(struct.pack("<7f", bmin[0], bmin[1], bmin[2], bmax[0], bmax[1], bmax[2], radius))
        mldb_buf.append(mldb_val & 0xFF)
        if include_mldl:
            mldl_buf.extend(struct.pack("<f", 0.0))

    mlmd_buf = bytearray()
    mlmx_buf = bytearray()
    mlmb_buf = bytearray()
    for entry40, (bmin, bmax, radius), mlmb_val in w_sorted:
        mlmd_buf.extend(entry40)
        mlmx_buf.extend(struct.pack("<7f", bmin[0], bmin[1], bmin[2], bmax[0], bmax[1], bmax[2], radius))
        mlmb_buf.append(mlmb_val & 0xFF)

    out = bytearray(mver)
    out.extend(make_chunk("MLFD", mlfd_payload))
    out.extend(make_chunk("MLDD", bytes(mldd_buf)))
    out.extend(make_chunk("MLDX", bytes(mldx_buf)))
    out.extend(make_chunk("MLDB", bytes(mldb_buf)))
    if include_mldl:
        out.extend(make_chunk("MLDL", bytes(mldl_buf)))
    if mwdr_bytes:
        out.extend(make_chunk("MWDR", mwdr_bytes))
    if mwds_bytes:
        out.extend(make_chunk("MWDS", mwds_bytes))
    out.extend(make_chunk("MLMD", bytes(mlmd_buf)))
    out.extend(make_chunk("MLMX", bytes(mlmx_buf)))
    out.extend(make_chunk("MLMB", bytes(mlmb_buf)))
    return bytes(out)


def _hole_set_from_masks(holes_64: Sequence[int]) -> List[bool]:
    holes = [False] * (128 * 128)
    for cy in range(16):
        for cx in range(16):
            mask = holes_64[cy * 16 + cx]
            if not mask:
                continue
            for ly in range(8):
                for lx in range(8):
                    if mask & (1 << (ly * 8 + lx)):
                        holes[(cy * 8 + ly) * 128 + (cx * 8 + lx)] = True
    return holes


def _triangulate_lod_node(
    nx: int,
    ny: int,
    side: int,
    step: int,
    holes: List[bool],
) -> List[int]:
    """
    Generate clockwise manifold terrain triangles inside a quadtree node [nx..nx+side)x[ny..ny+side).
    Coarse blocks overlapping any terrain hole automatically refine to cell step=1 while
    splitting perimeter edges cleanly against neighboring solid coarse blocks?
    To keep exact 2-manifold topology without T-junctions when holes exist, any node containing
    holes falls back cleanly to step=1 solid cells within its block (still well under the 65536
    per-level triangle budget for typical maps).
    """
    indices: List[int] = []
    # Check if any cell inside node is holed
    has_hole = any(
        holes[y * 128 + x]
        for y in range(ny, ny + side)
        for x in range(nx, nx + side)
    )
    eff_step = 1 if has_hole else step

    # Build node triangles in 2-tri-per-quad clockwise orientation (Y-down screen/grid):
    # v_a=(x,y), v_b=(x+s,y), v_d=(x,y+s), v_e=(x+s,y+s) -> (a, d, b), (b, d, e)
    for by in range(ny, ny + side, eff_step):
        for bx in range(nx, nx + side, eff_step):
            if eff_step == 1 and holes[by * 128 + bx]:
                continue
            a = by * 129 + bx
            b = by * 129 + (bx + eff_step)
            d = (by + eff_step) * 129 + bx
            e = (by + eff_step) * 129 + (bx + eff_step)
            if eff_step == 1 and (nx % step or ny % step or True):
                # 4-triangle center fan at step=1 uses exact MCVT center heights too!
                c = GRID_OUTER + by * 128 + bx
                indices.extend((c, b, a, c, e, b, c, d, e, c, a, d))
            else:
                indices.extend((a, d, b, b, d, e))
    return indices


def _emit_border_skirts(
    step: int,
    holes: List[bool],
    skirt_vertex_fn,
) -> List[int]:
    """
    Emit clockwise skirt quads along the 4 tile boundaries matching TerrainLodUpdater rules:
    top (y=0: x->x+1 => {b, a} -> {b, sa, a}?), let's keep edge order matching TerrainLodUpdater:
    x==0: (a, d), x==127: (e, b), y==0: (b, a), y==127: (d, e) with skirt quad (u, su, v, v, su, sv).
    """
    skirts: List[int] = []
    # Note: TerrainLodUpdater expects coarse borders not touching T-junctions if hole cut.
    # Determine effective edge step: use step if edge row/col has no holes, else 1.
    def add_seg(u: int, v: int) -> None:
        su = skirt_vertex_fn(u)
        sv = skirt_vertex_fn(v)
        skirts.extend((u, su, v, v, su, sv))

    # Top border (y=0): edge (b -> a) as (x+1, 0) -> (x, 0)
    top_step = 1 if any(holes[x] for x in range(128)) else step
    for x in range(0, 128, top_step):
        if top_step == 1 and holes[x]:
            continue
        add_seg(0 * 129 + (x + top_step), 0 * 129 + x)

    # Bottom border (y=127): edge (d -> e) as (x, 128) -> (x+1, 128)
    bot_step = 1 if any(holes[127 * 128 + x] for x in range(128)) else step
    for x in range(0, 128, bot_step):
        if bot_step == 1 and holes[127 * 128 + x]:
            continue
        add_seg(128 * 129 + x, 128 * 129 + (x + bot_step))

    # Left border (x=0): edge (a -> d) as (0, y) -> (0, y+1)
    left_step = 1 if any(holes[y * 128] for y in range(128)) else step
    for y in range(0, 128, left_step):
        if left_step == 1 and holes[y * 128]:
            continue
        add_seg(y * 129 + 0, (y + left_step) * 129 + 0)

    # Right border (x=127): edge (e -> b) as (128, y+1) -> (128, y)
    right_step = 1 if any(holes[y * 128 + 127] for y in range(128)) else step
    for y in range(0, 128, right_step):
        if right_step == 1 and holes[y * 128 + 127]:
            continue
        add_seg((y + right_step) * 129 + 128, y * 129 + 128)

    return skirts


def build_lod_bytes(
    mcvt_world_heights: List[List[float]],
    holes_64: Sequence[int],
    liquid_min_h: float = 1.0,
    liquid_max_h: float = -1.0,
) -> bytes:
    """
    Build a Retail _lod.adt terrain LOD file from 256 chunks of 145 absolute world heights
    and 256 uint64 high-res hole masks (passes TerrainLodUpdater::inspectReference & update).
    """
    # 1. Build base MLVH (129*129 outer + 128*128 center) in YOuterZeroCorner layout (layout=0)
    mlvh: List[float] = []
    for y in range(129):
        cy = min(y // 8, 15)
        ly = y - cy * 8
        for x in range(129):
            cx = min(x // 8, 15)
            lx = x - cx * 8
            mlvh.append(mcvt_world_heights[cy * 16 + cx][17 * ly + lx])
    for y in range(128):
        cy = min(y // 8, 15)
        ly = y - cy * 8
        for x in range(128):
            cx = min(x // 8, 15)
            lx = x - cx * 8
            mlvh.append(mcvt_world_heights[cy * 16 + cx][17 * ly + 9 + lx])

    assert len(mlvh) == GRID_COUNT

    mlsi: List[int] = []
    skirt_lookup: Dict[int, int] = {}

    def get_skirt_vertex(base_idx: int) -> int:
        if base_idx in skirt_lookup:
            return skirt_lookup[base_idx]
        new_idx = len(mlvh)
        if new_idx > 0xFFFF:
            raise ValueError("MLVH skirt count exceeded uint16 limit")
        mlsi.append(base_idx)
        mlvh.append(mlvh[base_idx] - 32.0)
        skirt_lookup[base_idx] = new_idx
        return new_idx

    holes = _hole_set_from_masks(holes_64)

    # 2. Build 85 breadth-first quadtree nodes across 4 LOD levels (steps: 8, 4, 2, 1)
    # Level 0: 1 node (side=128)
    # Level 1: 4 nodes (side=64)
    # Level 2: 16 nodes (side=32)
    # Level 3: 64 nodes (side=16)
    level_steps = [8, 4, 2, 1]
    level_lod_values = [16.0, 8.0, 4.0, 2.0]

    nodes: List[Tuple[int, int, int, int]] = [(0, 0, 128, 0)]
    node_children: List[Tuple[int, int, int, int]] = []
    idx_q = 0
    while idx_q < len(nodes):
        nx, ny, side, lvl = nodes[idx_q]
        if lvl == 3:
            node_children.append((0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF))
        else:
            cids = []
            half = side // 2
            for child in range(4):
                cid = len(nodes)
                cx = nx + (child % 2) * half
                cy = ny + (child // 2) * half
                nodes.append((cx, cy, half, lvl + 1))
                cids.append(cid)
            node_children.append(tuple(cids))  # type: ignore[arg-type]
        idx_q += 1
    assert len(nodes) == 85

    # Generate triangles per node & skirts per level (finest level 3 has no skirts)
    mlvi: List[int] = []
    mlnd_bytes = bytearray()
    mlll_bytes = bytearray()

    node_cursor = 0
    for lvl in range(4):
        lvl_start = len(mlvi)
        step = level_steps[lvl]
        while node_cursor < 85 and nodes[node_cursor][3] == lvl:
            nx, ny, side, _ = nodes[node_cursor]
            # If node had holes, neighbor nodes might differ unless coarse levels skip/align holes;
            # Using step=1 when ANY hole exists in the tile avoids inter-node T-junctions.
            eff_step = 1 if any(holes) else step
            tri_idx = _triangulate_lod_node(nx, ny, side, eff_step, holes)
            mlnd_bytes.extend(
                struct.pack(
                    "<IIII4H",
                    len(mlvi),
                    len(tri_idx),
                    0,
                    0,
                    *node_children[node_cursor],
                )
            )
            mlvi.extend(tri_idx)
            node_cursor += 1
        lvl_len = len(mlvi) - lvl_start

        if lvl < 3:
            eff_skirt_step = 1 if any(holes) else step
            skirt_idx = _emit_border_skirts(eff_skirt_step, holes, get_skirt_vertex)
            skirt_start = len(mlvi) if skirt_idx else 0
            mlvi.extend(skirt_idx)
        else:
            skirt_idx = []
            skirt_start = 0

        mlll_bytes.extend(
            struct.pack(
                "<fIIII",
                level_lod_values[lvl],
                lvl_start,
                lvl_len,
                len(skirt_idx),
                skirt_start,
            )
        )

    min_h = min(mlvh)
    max_h = max(mlvh)
    mlhd_bytes = struct.pack("<IffIIff", 0, min_h, max_h, 0, 0, liquid_min_h, liquid_max_h)
    mlvh_bytes = struct.pack(f"<{len(mlvh)}f", *mlvh)
    mlvi_bytes = struct.pack(f"<{len(mlvi)}H", *mlvi)
    mlsi_bytes = struct.pack(f"<{len(mlsi)}H", *mlsi)

    out = bytearray()
    out.extend(make_chunk("MLHD", mlhd_bytes))
    out.extend(make_chunk("MLVH", mlvh_bytes))
    out.extend(make_chunk("MLVI", mlvi_bytes))
    out.extend(make_chunk("MLSI", mlsi_bytes))
    out.extend(make_chunk("MLLL", bytes(mlll_bytes)))
    out.extend(make_chunk("MLND", bytes(mlnd_bytes)))
    return bytes(out)


def build_retail_wdt(
    active_tiles: Iterable[Tuple[int, int]],
    has_mccv: bool = True,
    has_height_textures: bool = True,
    has_lod: bool = False,
    maid_mode: bool = False,
    base_fdid: int = 5000000,
    map_name: str = "custom_map",
) -> Tuple[bytes, List[Tuple[int, str]]]:
    """
    Build a Retail/Shadowlands WDT file.
    Returns (wdt_bytes, custom_listfile_entries[(fdid, path)]).
    """
    flags = 0x04 | 0x08  # adt_has_big_alpha | adt_has_doodadrefs_sorted_by_size_cat
    if has_mccv:
        flags |= 0x02    # adt_has_mccv
    if has_height_textures:
        flags |= 0x80    # adt_has_height_texturing
    if has_lod:
        flags |= 0x100   # wdt_has_terrain_lod
    if maid_mode:
        flags |= 0x200   # wdt_has_maid

    mphd = struct.pack("<8I", flags, 0, 0, 0, 0, 0, 0, 0)
    main = bytearray(64 * 64 * 8)
    maid = bytearray(64 * 64 * 32)
    extra_listfile: List[Tuple[int, str]] = []
    next_fid = base_fdid

    active_set = sorted(set((int(tx), int(ty)) for tx, ty in active_tiles))
    for tx, ty in active_set:
        if not (0 <= tx < 64 and 0 <= ty < 64):
            continue
        struct.pack_into("<II", main, (ty * 64 + tx) * 8, 1, 0)
        if maid_mode:
            root_id = next_fid
            obj0_id = next_fid + 1
            obj1_id = next_fid + 2
            tex0_id = next_fid + 3
            lod_id = (next_fid + 4) if has_lod else 0
            next_fid += 5 if has_lod else 4
            struct.pack_into(
                "<8I",
                maid,
                (ty * 64 + tx) * 32,
                root_id, obj0_id, obj1_id, tex0_id, lod_id, 0, 0, 0,
            )
            prefix = f"world/maps/{map_name}/{map_name}_{tx}_{ty}"
            extra_listfile.append((root_id, f"{prefix}.adt"))
            extra_listfile.append((obj0_id, f"{prefix}_obj0.adt"))
            extra_listfile.append((obj1_id, f"{prefix}_obj1.adt"))
            extra_listfile.append((tex0_id, f"{prefix}_tex0.adt"))
            if has_lod:
                extra_listfile.append((lod_id, f"{prefix}_lod.adt"))

    out = bytearray()
    out.extend(make_chunk("MVER", struct.pack("<I", 18)))
    out.extend(make_chunk("MPHD", mphd))
    out.extend(make_chunk("MAIN", bytes(main)))
    if maid_mode:
        out.extend(make_chunk("MAID", bytes(maid)))
    return bytes(out), extra_listfile


def sample_wdl_heights_for_tile(
    mcvt_world_heights: List[List[float]], holes_64: Sequence[int]
) -> Tuple[List[int], List[int], List[int]]:
    """
    Compute WDL 17x17 outer + 16x16 inner int16 heights + 16x uint16 MAHO masks
    matching RetailWorldTables::updateWdlTile & wdlHoles.
    """
    outer: List[int] = []
    for y in range(17):
        cy = min(y, 15)
        vy = 8 if y == 16 else 0
        for x in range(17):
            cx = min(x, 15)
            vx = 8 if x == 16 else 0
            h = mcvt_world_heights[cy * 16 + cx][17 * vy + vx]
            val = int(round(max(-32768.0, min(32767.0, h))))
            outer.append(val)

    inner: List[int] = []
    for cy in range(16):
        for cx in range(16):
            # Center vertex (4,4) inside chunk MCVT = index 17*4 + 4 = 72
            h = mcvt_world_heights[cy * 16 + cx][72]
            val = int(round(max(-32768.0, min(32767.0, h))))
            inner.append(val)

    maho: List[int] = []
    for cy in range(16):
        row_mask = 0
        for cx in range(16):
            low16 = compress_holes_high_to_low(holes_64[cy * 16 + cx])
            if low16 == 0xFFFF:
                row_mask |= 1 << cx
        maho.append(row_mask)

    return outer, inner, maho


def update_or_create_wdl(
    existing_wdl: Optional[bytes],
    tiles_data: Dict[Tuple[int, int], Tuple[List[int], List[int], List[int]]],
) -> bytes:
    """
    Update existing MARE/MAHO entries for converted tiles or build a clean new WDL v18.
    Preserves existing unknown chunks / untouched tiles when existing_wdl is supplied.
    """
    if existing_wdl is None:
        maof = bytearray(64 * 64 * 4)
        body = bytearray()
        prefix = bytearray()
        prefix.extend(make_chunk("MVER", struct.pack("<I", 18)))
        prefix.extend(make_chunk("MWMO", b""))
        prefix.extend(make_chunk("MWID", b""))
        prefix.extend(make_chunk("MODF", b""))
        maof_header_end = len(prefix) + 8 + len(maof)

        for ty in range(64):
            for tx in range(64):
                if (tx, ty) not in tiles_data:
                    continue
                outer, inner, maho = tiles_data[(tx, ty)]
                mare_ofs = maof_header_end + len(body)
                struct.pack_into("<I", maof, (ty * 64 + tx) * 4, mare_ofs)
                mare_payload = struct.pack("<289h", *outer) + struct.pack("<256h", *inner)
                maho_payload = struct.pack("<16H", *maho)
                body.extend(make_chunk("MARE", mare_payload))
                body.extend(make_chunk("MAHO", maho_payload))

        return bytes(prefix) + make_chunk("MAOF", bytes(maof)) + bytes(body)

    # Patch existing WDL cleanly (preserving unknown chunks/other tiles if all target tiles exist in MAOF)
    buf = bytearray(existing_wdl)
    parsed = list(iter_chunks(existing_wdl, skip_null=False))
    maof_entry = next((c for c in parsed if c[0] == "MAOF"), None)
    if maof_entry is None or maof_entry[2] != 16384:
        return update_or_create_wdl(None, tiles_data)

    maof_off = maof_entry[1] + 8
    chunks_by_pos = {pos: (name, sz) for name, pos, sz, _ in parsed}
    missing_in_wdl = False
    for (tx, ty), (outer, inner, maho) in tiles_data.items():
        ofs = struct.unpack_from("<I", buf, maof_off + (ty * 64 + tx) * 4)[0]
        if ofs == 0:
            missing_in_wdl = True
            continue
        cinfo = chunks_by_pos.get(ofs)
        if not cinfo or cinfo[0] != "MARE" or cinfo[1] != 1090:
            missing_in_wdl = True
            continue
        mare_payload = struct.pack("<289h", *outer) + struct.pack("<256h", *inner)
        buf[ofs + 8 : ofs + 8 + 1090] = mare_payload
        maho_pos = ofs + 8 + 1090
        if maho_pos in chunks_by_pos and chunks_by_pos[maho_pos][0] == "MAHO":
            buf[maho_pos + 8 : maho_pos + 8 + 32] = struct.pack("<16H", *maho)

    if not missing_in_wdl:
        return bytes(buf)

    # Rebuild full WDL merging old tiles and newly converted tiles
    merged: Dict[Tuple[int, int], Tuple[List[int], List[int], List[int]]] = {}
    for ty in range(64):
        for tx in range(64):
            ofs = struct.unpack_from("<I", buf, maof_off + (ty * 64 + tx) * 4)[0]
            if ofs and ofs in chunks_by_pos and chunks_by_pos[ofs][0] == "MARE":
                outer = list(struct.unpack_from("<289h", buf, ofs + 8))
                inner = list(struct.unpack_from("<256h", buf, ofs + 8 + 578))
                maho_pos = ofs + 8 + 1090
                if maho_pos in chunks_by_pos and chunks_by_pos[maho_pos][0] == "MAHO":
                    maho = list(struct.unpack_from("<16H", buf, maho_pos + 8))
                else:
                    maho = [0] * 16
                merged[(tx, ty)] = (outer, inner, maho)
    merged.update(tiles_data)
    return update_or_create_wdl(None, merged)
