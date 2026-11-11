#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WotLK 3.3.5a -> Modern Retail (11.x/12.x) & Shadowlands (9.2.7) Split-ADT Converter CLI.

Converts monolithic WotLK .adt tiles (+ optional .wdt / .wdl) into validated split Retail maps:
  - <map>_X_Y.adt        (ROOT: MVER, MHDR, MH2O, 256x MCNK w/ 0x18000 hi-res holes & MCNR-448, MFBO)
  - <map>_X_Y_tex0.adt   (TEX0: MVER, MAMP, MDID, MHID, optional MTXP, 256x 0-hdr MCNK w/ MCLY+MCAL)
  - <map>_X_Y_obj0.adt   (OBJ0: MVER, MDDF w/ 0x40 FDID bit, MODF w/ 0x0C, 256x 0-hdr MCNK MCRD+MCRW)
  - <map>_X_Y_obj1.adt   (OBJ1: MLFD, MLDD, MLDX, MLDB, MLDL, MLMD, MLMX, MLMB radius-sorted)
  - <map>_X_Y_lod.adt    (optional LOD: MLHD, MLVH, MLVI, MLSI, MLLL, MLND 4-level quadtree + holes)
  - <map>.wdt            (path-addressed or MAID-addressed MPHD+MAIN)
  - <map>.wdl            (updated/synthesized MARE + MAHO)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from listfile_resolver import (
    ListfileResolver,
    ResolutionStats,
    load_alias_file,
    normalize_wow_path,
)
from lod_and_world import (
    build_lod_bytes,
    build_obj1_bytes,
    build_retail_wdt,
    sample_wdl_heights_for_tile,
    update_or_create_wdl,
)
from retail_chunks import (
    MAP_MIDPOINT,
    MCLY_ALPHA_COMPRESSED,
    MCLY_USE_ALPHA_MAP,
    MCNK_FLAG_DO_NOT_FIX_ALPHA_MAP,
    MCNK_FLAG_HAS_MCCV,
    MCNK_FLAG_HIGH_RES_HOLES,
    MCNK_FLAG_IMPASS,
    MCNK_FLAG_LQ_MAGMA,
    MCNK_FLAG_LQ_OCEAN,
    MCNK_FLAG_LQ_RIVER,
    MCNK_FLAG_LQ_SLIME,
    MDDF_ENTRY_IS_FILEDATA_ID,
    MODF_ENTRY_IS_FILEDATA_ID,
    MODF_USE_MODEL_SCALE,
    compress_rle_mcal,
    convert_mclq_to_mh2o,
    decode_mcal_layer,
    doodad_lod_extent,
    expand_holes_low_to_high,
    iter_chunks,
    iter_wotlk_mcnk_subchunks,
    make_chunk,
    modf_world_extents_to_lod,
    upgrade_mcnr_payload,
)

TILE_RE = re.compile(r"^(?P<map>.+)_(?P<x>\d+)_(?P<y>\d+)$", re.IGNORECASE)


@dataclass
class ParsedTile:
    path: Path
    map_name: str
    tx: int
    ty: int
    mhdr_flags: int = 0
    mtex: List[str] = field(default_factory=list)
    mmdx: List[str] = field(default_factory=list)
    mwmo: List[str] = field(default_factory=list)
    mddf: List[Tuple[int, int, Tuple[float, float, float], Tuple[float, float, float], int, int]] = field(
        default_factory=list
    )
    modf: List[
        Tuple[
            int,
            int,
            Tuple[float, float, float],
            Tuple[float, float, float],
            Tuple[float, float, float],
            Tuple[float, float, float],
            int,
            int,
            int,
            int,
        ]
    ] = field(default_factory=list)
    mh2o: Optional[bytes] = None
    mfbo: Optional[bytes] = None
    chunks: List[dict] = field(default_factory=list)  # length 256 in (iy*16+ix) order


def _read_cstr_table(strings_blob: bytes, offsets_blob: bytes) -> List[str]:
    if len(offsets_blob) % 4 != 0:
        raise ValueError(f"String offset table not 4-byte aligned ({len(offsets_blob)})")
    count = len(offsets_blob) // 4
    result: List[str] = []
    for i in range(count):
        off = struct.unpack_from("<I", offsets_blob, i * 4)[0]
        if off >= len(strings_blob):
            raise ValueError(f"String offset {off} out of range {len(strings_blob)}")
        end = strings_blob.find(b"\x00", off)
        if end < 0:
            end = len(strings_blob)
        result.append(strings_blob[off:end].decode("latin1", "replace"))
    return result


def parse_wotlk_adt(adt_path: Path) -> ParsedTile:
    """Parse a monolithic WotLK ADT into structured chunks."""
    m = TILE_RE.match(adt_path.stem)
    if not m:
        raise ValueError(f"Cannot infer tile X_Y coordinates from filename: {adt_path.name}")
    data = adt_path.read_bytes()

    mhdr_flags = 0
    mtex: List[str] = []
    mmdx_raw = b""
    mmid_raw = b""
    mwmo_raw = b""
    mwid_raw = b""
    mddf_raw = b""
    modf_raw = b""
    mh2o: Optional[bytes] = None
    mfbo: Optional[bytes] = None
    raw_mcnks: Dict[Tuple[int, int], dict] = {}
    seq_mcnks: List[dict] = []

    for name, _, size, payload in iter_chunks(data, skip_null=True):
        if name == "MVER":
            ver = struct.unpack_from("<I", payload, 0)[0]
            if ver != 18:
                raise ValueError(f"{adt_path.name}: unsupported MVER {ver}")
        elif name == "MHDR":
            mhdr_flags = struct.unpack_from("<I", payload, 0)[0]
        elif name == "MTEX":
            mtex = [s.decode("latin1", "replace") for s in payload.split(b"\x00") if s]
        elif name == "MMDX":
            mmdx_raw = payload
        elif name == "MMID":
            mmid_raw = payload
        elif name == "MWMO":
            mwmo_raw = payload
        elif name == "MWID":
            mwid_raw = payload
        elif name == "MDDF":
            mddf_raw = payload
        elif name == "MODF":
            modf_raw = payload
        elif name == "MH2O":
            mh2o = payload
        elif name == "MFBO":
            mfbo = payload
        elif name == "MCNK":
            if len(payload) < 128:
                raise ValueError(f"{adt_path.name}: truncated MCNK payload ({len(payload)}B)")
            hdr = struct.unpack_from("<16I", payload, 0)
            cflags = hdr[0]
            ix, iy = hdr[1], hdr[2]
            n_doodad = hdr[4]
            hi_holes_u64 = struct.unpack_from("<Q", payload, 20)[0]
            areaid = hdr[13]
            n_mapobj = hdr[14]
            low_holes = hdr[15] & 0xFFFF
            n_snd_emitters = struct.unpack_from("<I", payload, 92)[0]
            xpos, ypos, zpos = struct.unpack_from("<3f", payload, 104)

            subs: Dict[str, bytes] = {}
            for sname, _, _, spay in iter_wotlk_mcnk_subchunks(payload):
                subs[sname] = spay

            cdict = {
                "flags": cflags,
                "ix": ix,
                "iy": iy,
                "nDoodadRefs": n_doodad,
                "nMapObjRefs": n_mapobj,
                "hiResHoles": hi_holes_u64,
                "lowHoles": low_holes,
                "areaid": areaid,
                "nSndEmitters": n_snd_emitters,
                "pos": (xpos, ypos, zpos),
                "subs": subs,
            }
            seq_mcnks.append(cdict)
            if 0 <= ix < 16 and 0 <= iy < 16:
                raw_mcnks[(ix, iy)] = cdict

    if len(seq_mcnks) != 256:
        raise ValueError(f"{adt_path.name}: expected 256 MCNKs, found {len(seq_mcnks)}")

    ordered_chunks: List[dict] = []
    if len(raw_mcnks) == 256:
        for iy in range(16):
            for ix in range(16):
                ordered_chunks.append(raw_mcnks[(ix, iy)])
    else:
        for idx, c in enumerate(seq_mcnks):
            c["ix"] = idx % 16
            c["iy"] = idx // 16
            ordered_chunks.append(c)

    mmdx = _read_cstr_table(mmdx_raw, mmid_raw) if mmid_raw else []
    mwmo = _read_cstr_table(mwmo_raw, mwid_raw) if mwid_raw else []

    mddf = []
    for i in range(len(mddf_raw) // 36):
        mid, uid, px, py, pz, rx, ry, rz, scale, fl = struct.unpack_from("<II6fHH", mddf_raw, i * 36)
        mddf.append((mid, uid, (px, py, pz), (rx, ry, rz), scale, fl))

    modf = []
    for i in range(len(modf_raw) // 64):
        (
            wid, uid, px, py, pz, rx, ry, rz,
            ex1, ey1, ez1, ex2, ey2, ez2,
            fl, dset, nset, scale,
        ) = struct.unpack_from("<II12f4H", modf_raw, i * 64)
        modf.append(
            (wid, uid, (px, py, pz), (rx, ry, rz), (ex1, ey1, ez1), (ex2, ey2, ez2), fl, dset, nset, scale)
        )

    return ParsedTile(
        path=adt_path,
        map_name=m.group("map"),
        tx=int(m.group("x")),
        ty=int(m.group("y")),
        mhdr_flags=mhdr_flags,
        mtex=mtex,
        mmdx=mmdx,
        mwmo=mwmo,
        mddf=mddf,
        modf=modf,
        mh2o=mh2o,
        mfbo=mfbo,
        chunks=ordered_chunks,
    )


def _read_local_m2_bounds(
    assets_dir: Optional[Path], resolved_rel_path: Optional[str], source_rel_path: str
) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """Optional local M2 bounding-box reader (supports legacy MD20 and Retail MD21 headers)."""
    if assets_dir is None:
        return None
    for rel in (resolved_rel_path, source_rel_path):
        if not rel:
            continue
        candidate = assets_dir / rel.replace("/", os.sep)
        if candidate.is_file():
            raw = candidate.read_bytes()
            base = 0
            if raw[:4] == b"MD21":
                base = 8
            if len(raw) >= base + 0xB8 and raw[base : base + 4] == b"MD20":
                bmin = struct.unpack_from("<3f", raw, base + 0xA0)
                bmax = struct.unpack_from("<3f", raw, base + 0xAC)
                if all(math.isfinite(v) for v in (*bmin, *bmax)):
                    return bmin, bmax
    return None


def read_wdt_big_alpha_flag(wdt_path: Optional[Path]) -> Optional[bool]:
    if wdt_path is None or not wdt_path.exists():
        return None
    for name, _, _, payload in iter_chunks(wdt_path.read_bytes()):
        if name == "MPHD" and len(payload) >= 4:
            flags = struct.unpack_from("<I", payload, 0)[0]
            return (flags & 0x04) != 0
    return None


def infer_tile_big_alpha(tile: ParsedTile) -> bool:
    """Fallback Big Alpha detection if WDT is unavailable: inspects compressed vs raw MCAL sizes."""
    for c in tile.chunks:
        mcly = c["subs"].get("MCLY", b"")
        mcal = c["subs"].get("MCAL", b"")
        if len(mcly) < 32 or not mcal:
            continue
        layers = [struct.unpack_from("<IIII", mcly, i * 16) for i in range(len(mcly) // 16)]
        uncompressed_layers = [
            i for i in range(1, len(layers)) if not (layers[i][1] & MCLY_ALPHA_COMPRESSED)
        ]
        if uncompressed_layers:
            i0 = uncompressed_layers[0]
            end_off = layers[i0 + 1][2] if i0 + 1 < len(layers) else len(mcal)
            if end_off - layers[i0][2] >= 4096:
                return True
            return False
    return True


def _record_resolution(stats: ResolutionStats, res, key: str) -> None:
    if res.method == "exact":
        stats.exact += 1
    elif res.method == "alias":
        stats.alias += 1
    elif res.method == "basename":
        stats.basename += 1
        stats.fuzzy_mappings[res.source_path] = res.resolved_path or str(res.fdid)
    elif res.method == "fuzzy":
        stats.fuzzy += 1
        stats.fuzzy_mappings[res.source_path] = res.resolved_path or str(res.fdid)
    else:
        stats.missing += 1
        stats.missing_assets.setdefault(key, []).append(res.source_path)


def convert_tiles(
    input_paths: List[Path],
    output_dir: Path,
    map_name: str,
    resolver: ListfileResolver,
    big_alpha_override: Optional[bool] = None,
    target_version: str = "retail",
    missing_policy: str = "skip",
    compress_alpha: bool = False,
    height_textures: bool = True,
    emit_mtxp: bool = False,
    with_obj1: bool = True,
    with_lod: bool = False,
    synth_mfbo: bool = True,
    fix_uids: bool = True,
    wdt_mode: str = "path",
    base_fdid: int = 5000000,
    existing_wdt_path: Optional[Path] = None,
    existing_wdl_path: Optional[Path] = None,
    assets_dir: Optional[Path] = None,
    noggitproj_dir: Optional[Path] = None,
    client_path: str = "",
    map_id: int = 0,
) -> dict:
    """Convert parsed WotLK ADTs into Retail/Shadowlands split ADTs + WDT + WDL."""
    t_start = time.time()
    stats = ResolutionStats()
    tiles: List[ParsedTile] = [parse_wotlk_adt(p) for p in sorted(input_paths)]

    wdt_big_alpha = read_wdt_big_alpha_flag(existing_wdt_path)
    out_maps_dir = (noggitproj_dir / "world" / "maps" / map_name) if noggitproj_dir else output_dir
    out_maps_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: Global asset resolution and optional UID conflict resolution
    max_uid = 0
    seen_identities: Dict[Tuple[str, int, Tuple[float, ...], Tuple[float, ...], int, int], int] = {}
    uid_owner: Dict[int, Tuple[str, int, Tuple[float, ...], Tuple[float, ...], int, int]] = {}
    # Reserve first pass to find max_uid cleanly before minting any collision UIDs
    for tile in tiles:
        for _, uid, *_ in tile.mddf:
            if uid > max_uid:
                max_uid = uid
        for _, uid, *_ in tile.modf:
            if uid > max_uid:
                max_uid = uid

    resolved_tiles = []
    wdl_tile_samples: Dict[Tuple[int, int], Tuple[List[int], List[int], List[int]]] = {}
    any_mccv = False

    def alloc_uid(ident) -> int:
        nonlocal max_uid
        if not fix_uids:
            return ident[-1]
        if ident in seen_identities:
            return seen_identities[ident]
        orig_uid = ident[-1]
        if orig_uid != 0 and orig_uid not in uid_owner:
            uid_owner[orig_uid] = ident
            seen_identities[ident] = orig_uid
            return orig_uid
        max_uid += 1
        uid_owner[max_uid] = ident
        seen_identities[ident] = max_uid
        return max_uid

    for tile in tiles:
        big_alpha = (
            big_alpha_override
            if big_alpha_override is not None
            else (wdt_big_alpha if wdt_big_alpha is not None else infer_tile_big_alpha(tile))
        )

        # Resolve textures
        mdids: List[int] = []
        mhids: List[int] = []
        for tex_path in tile.mtex:
            res = resolver.resolve(tex_path, is_model=False, kind="texture")
            _record_resolution(stats, res, "texture")
            if res.fdid == 0 and missing_policy == "fail":
                raise RuntimeError(f"{tile.path.name}: unresolved texture {tex_path!r}")
            mdids.append(res.fdid)
            hid = resolver.resolve_height_texture(res.resolved_path or tex_path) if height_textures else 0
            if hid:
                stats.height_textures_found += 1
            mhids.append(hid)

        # Resolve M2 placements & remap indices
        m2_remap: Dict[int, Optional[int]] = {}
        out_mddf: List[Tuple[bytes, Tuple, int]] = []
        for old_idx, (mid, uid, pos, rot, scale, fl) in enumerate(tile.mddf):
            if mid >= len(tile.mmdx):
                m2_remap[old_idx] = None
                continue
            src_path = tile.mmdx[mid]
            res = resolver.resolve(src_path, is_model=True, kind="m2")
            _record_resolution(stats, res, "m2")
            if res.fdid == 0:
                if missing_policy == "fail":
                    raise RuntimeError(f"{tile.path.name}: unresolved M2 {src_path!r}")
                elif missing_policy == "skip":
                    m2_remap[old_idx] = None
                    continue

            norm_scale = scale if scale > 0 else 1024
            rpos = tuple(round(v, 3) for v in pos)
            rrot = tuple(round(v, 3) for v in rot)
            final_uid = alloc_uid(("M", res.fdid, rpos, rrot, norm_scale, uid))
            # Preserve existing valid bits while enforcing Retail/SL FileDataID flag 0x40
            final_fl = fl | MDDF_ENTRY_IS_FILEDATA_ID
            entry36 = struct.pack(
                "<II6fHH",
                res.fdid, final_uid, pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], norm_scale, final_fl
            )
            local_bounds = _read_local_m2_bounds(assets_dir, res.resolved_path, src_path)
            extent = doodad_lod_extent(pos, norm_scale, local_bounds)
            m2_remap[old_idx] = len(out_mddf)
            out_mddf.append((entry36, extent, 0x40))

        # Resolve WMO placements & remap indices
        wmo_remap: Dict[int, Optional[int]] = {}
        out_modf: List[Tuple[bytes, bytes, Tuple, int]] = []
        for old_idx, (wid, uid, pos, rot, bmin, bmax, fl, dset, nset, scale) in enumerate(tile.modf):
            if wid >= len(tile.mwmo):
                wmo_remap[old_idx] = None
                continue
            src_path = tile.mwmo[wid]
            res = resolver.resolve(src_path, is_model=False, kind="wmo")
            _record_resolution(stats, res, "wmo")
            if res.fdid == 0:
                if missing_policy == "fail":
                    raise RuntimeError(f"{tile.path.name}: unresolved WMO {src_path!r}")
                elif missing_policy == "skip":
                    wmo_remap[old_idx] = None
                    continue

            norm_scale = scale if scale > 0 else 1024
            rpos = tuple(round(v, 3) for v in pos)
            rrot = tuple(round(v, 3) for v in rot)
            final_uid = alloc_uid(("W", res.fdid, rpos, rrot, norm_scale, uid))
            final_fl = fl | MODF_ENTRY_IS_FILEDATA_ID | MODF_USE_MODEL_SCALE
            entry64 = struct.pack(
                "<II12f4H",
                res.fdid,
                final_uid,
                pos[0], pos[1], pos[2],
                rot[0], rot[1], rot[2],
                bmin[0], bmin[1], bmin[2],
                bmax[0], bmax[1], bmax[2],
                final_fl, dset, nset, norm_scale,
            )
            # OBJ1 MLMD is 40B: first 32B (id, uid, pos[3], rot[3]) + 8B (flags, dset, nset, scale)
            mlmd40 = entry64[:32] + entry64[56:64]
            extent = modf_world_extents_to_lod(bmin, bmax)
            wmo_remap[old_idx] = len(out_modf)
            out_modf.append((entry64, mlmd40, extent, 0x80))

        # Liquid: preserve MH2O or upgrade legacy MCLQ
        mh2o_payload = tile.mh2o
        if mh2o_payload is None:
            mclq_pairs = [(c["flags"], c["subs"].get("MCLQ", b"")) for c in tile.chunks]
            mh2o_payload = convert_mclq_to_mh2o(mclq_pairs)

        # Build ROOT, TEX0, OBJ0 MCNKs simultaneously
        root_mcnks = bytearray()
        tex0_mcnks = bytearray()
        obj0_mcnks = bytearray()
        mcvt_world_heights: List[List[float]] = []
        holes_64: List[int] = []
        max_tile_z = -1e9
        min_tile_z = 1e9

        for idx, c in enumerate(tile.chunks):
            cflags = c["flags"]
            ix, iy = c["ix"], c["iy"]
            px, py, pz = c["pos"]
            subs = c["subs"]

            mcvt = subs.get("MCVT")
            if not mcvt or len(mcvt) != 580:
                raise ValueError(f"{tile.path.name} MCNK[{idx}]: invalid MCVT size")
            rel_h = struct.unpack("<145f", mcvt)
            world_h = [pz + h for h in rel_h]
            mcvt_world_heights.append(world_h)
            max_tile_z = max(max_tile_z, max(world_h))
            min_tile_z = min(min_tile_z, min(world_h))

            mcnr = upgrade_mcnr_payload(subs.get("MCNR", b""))
            mccv = subs.get("MCCV") if (cflags & MCNK_FLAG_HAS_MCCV and len(subs.get("MCCV", b"")) == 580) else None
            if mccv is not None:
                any_mccv = True
            mcse = subs.get("MCSE", b"")

            # High-res 8x8 hole mask
            hi_holes = (
                c["hiResHoles"]
                if (cflags & MCNK_FLAG_HIGH_RES_HOLES)
                else expand_holes_low_to_high(c["lowHoles"])
            )
            holes_64.append(hi_holes)

            # Build ROOT MCNK body strictly in RetailRootLayout order: MCVT, MCNR(448), [MCCV], MCSE
            root_subs = bytearray()
            root_subs.extend(make_chunk("MCVT", mcvt))
            root_subs.extend(make_chunk("MCNR", mcnr))
            ofs_mccv = 0
            if mccv is not None:
                ofs_mccv = 128 + len(root_subs) + 8
                root_subs.extend(make_chunk("MCCV", mccv))
            ofs_snd = 0
            n_snd = 0
            if mcse:
                ofs_snd = 128 + len(root_subs) + 8
                n_snd = len(mcse) // 28
            root_subs.extend(make_chunk("MCSE", mcse))

            out_cflags = (
                MCNK_FLAG_HIGH_RES_HOLES
                | MCNK_FLAG_DO_NOT_FIX_ALPHA_MAP
                | (MCNK_FLAG_HAS_MCCV if mccv is not None else 0)
                | (cflags & (MCNK_FLAG_IMPASS | MCNK_FLAG_LQ_RIVER | MCNK_FLAG_LQ_OCEAN | MCNK_FLAG_LQ_MAGMA | MCNK_FLAG_LQ_SLIME))
            )

            root_hdr = bytearray(128)
            struct.pack_into(
                "<5IQ8I",
                root_hdr,
                0,
                out_cflags, ix, iy, 0, 0,
                hi_holes,
                0, 0, 0, 0, 0, 0, c["areaid"], 0,
            )
            struct.pack_into("<4I", root_hdr, 88, ofs_snd, n_snd, 0, 0)
            struct.pack_into("<3f3I", root_hdr, 104, px, py, pz, ofs_mccv, 0, 0)
            root_mcnks.extend(make_chunk("MCNK", bytes(root_hdr) + bytes(root_subs)))

            # Build TEX0 MCNK (0-byte header + MCLY + MCAL)
            mcly_raw = subs.get("MCLY", b"")
            mcal_raw = subs.get("MCAL", b"")
            apply_edge_fix = not bool(cflags & MCNK_FLAG_DO_NOT_FIX_ALPHA_MAP)
            tex_subs = bytearray()
            if mcly_raw:
                layer_count = len(mcly_raw) // 16
                parsed_layers = [struct.unpack_from("<4I", mcly_raw, li * 16) for li in range(layer_count)]
                out_mcly = bytearray()
                out_mcal = bytearray()
                for li, (tid, lflags, oa, eid) in enumerate(parsed_layers):
                    if tid >= len(mdids):
                        tid = 0
                    if li == 0:
                        new_fl = lflags & ~(MCLY_USE_ALPHA_MAP | MCLY_ALPHA_COMPRESSED)
                        out_mcly.extend(struct.pack("<4I", tid, new_fl, 0, eid))
                    else:
                        next_oa = parsed_layers[li + 1][2] if li + 1 < layer_count else len(mcal_raw)
                        alpha4096 = decode_mcal_layer(
                            mcal_raw, oa, next_oa, lflags, big_alpha, apply_edge_fix
                        )
                        layer_off = len(out_mcal)
                        if compress_alpha:
                            encoded = compress_rle_mcal(alpha4096)
                            new_fl = lflags | MCLY_USE_ALPHA_MAP | MCLY_ALPHA_COMPRESSED
                            out_mcal.extend(encoded)
                        else:
                            new_fl = (lflags & ~MCLY_ALPHA_COMPRESSED) | MCLY_USE_ALPHA_MAP
                            out_mcal.extend(alpha4096)
                        out_mcly.extend(struct.pack("<4I", tid, new_fl, layer_off, eid))
                tex_subs.extend(make_chunk("MCLY", bytes(out_mcly)))
                tex_subs.extend(make_chunk("MCAL", bytes(out_mcal)))
            tex0_mcnks.extend(make_chunk("MCNK", bytes(tex_subs)))

            # Build OBJ0 MCNK (0-byte header + MCRD + MCRW)
            mcrf_raw = subs.get("MCRF", b"")
            n_d = c["nDoodadRefs"]
            n_w = c["nMapObjRefs"]
            mcrd_ids: List[int] = []
            mcrw_ids: List[int] = []
            if len(mcrf_raw) >= (n_d + n_w) * 4:
                for i in range(n_d):
                    oid = struct.unpack_from("<I", mcrf_raw, i * 4)[0]
                    nid = m2_remap.get(oid)
                    if nid is not None:
                        mcrd_ids.append(nid)
                for i in range(n_w):
                    oid = struct.unpack_from("<I", mcrf_raw, (n_d + i) * 4)[0]
                    nid = wmo_remap.get(oid)
                    if nid is not None:
                        mcrw_ids.append(nid)

            obj_subs = bytearray()
            if mcrd_ids:
                obj_subs.extend(make_chunk("MCRD", struct.pack(f"<{len(mcrd_ids)}I", *mcrd_ids)))
            if mcrw_ids:
                obj_subs.extend(make_chunk("MCRW", struct.pack(f"<{len(mcrw_ids)}I", *mcrw_ids)))
            obj0_mcnks.extend(make_chunk("MCNK", bytes(obj_subs)))

        # Assemble ROOT file
        mfbo_payload = tile.mfbo
        if mfbo_payload is None and synth_mfbo:
            top_v = int(min(32767, max(1700, math.ceil(max_tile_z + 500.0))))
            bot_v = int(max(-32768, min(-1000, math.floor(min_tile_z - 500.0))))
            mfbo_payload = struct.pack("<18h", *([top_v] * 9 + [bot_v] * 9))

        mhdr_flags = 1 if mfbo_payload is not None else 0
        ofs_mh2o = 64 if mh2o_payload is not None else 0
        root_body = bytearray()
        if mh2o_payload is not None:
            root_body.extend(make_chunk("MH2O", mh2o_payload))
        root_body.extend(root_mcnks)
        ofs_mfbo = (84 + len(root_body)) - 20 if mfbo_payload is not None else 0
        if mfbo_payload is not None:
            root_body.extend(make_chunk("MFBO", mfbo_payload))

        mhdr_payload = struct.pack(
            "<16I",
            mhdr_flags,
            0, 0, 0, 0, 0, 0, 0, 0,
            ofs_mfbo,
            ofs_mh2o,
            0, 0, 0, 0, 0,
        )
        root_bytes = make_chunk("MVER", struct.pack("<I", 18)) + make_chunk("MHDR", mhdr_payload) + bytes(root_body)

        # Assemble TEX0 file
        tex_hdr = bytearray()
        tex_hdr.extend(make_chunk("MVER", struct.pack("<I", 18)))
        tex_hdr.extend(make_chunk("MAMP", struct.pack("<I", 0)))
        tex_hdr.extend(make_chunk("MDID", struct.pack(f"<{len(mdids)}I", *mdids) if mdids else b""))
        tex_hdr.extend(make_chunk("MHID", struct.pack(f"<{len(mhids)}I", *mhids) if mhids else b""))
        if emit_mtxp or any(hid != 0 for hid in mhids):
            mtxp_buf = bytearray()
            for _ in mdids:
                mtxp_buf.extend(struct.pack("<IffI", 0, 0.0, 1.0, 0))
            tex_hdr.extend(make_chunk("MTXP", bytes(mtxp_buf)))
        tex0_bytes = bytes(tex_hdr) + bytes(tex0_mcnks)

        # Assemble OBJ0 file
        mddf_bytes = b"".join(row[0] for row in out_mddf)
        modf_bytes = b"".join(row[0] for row in out_modf)
        obj0_bytes = (
            make_chunk("MVER", struct.pack("<I", 18))
            + make_chunk("MDDF", mddf_bytes)
            + make_chunk("MODF", modf_bytes)
            + bytes(obj0_mcnks)
        )

        # Write outputs
        stem = f"{map_name}_{tile.tx}_{tile.ty}"
        (out_maps_dir / f"{stem}.adt").write_bytes(root_bytes)
        (out_maps_dir / f"{stem}_tex0.adt").write_bytes(tex0_bytes)
        (out_maps_dir / f"{stem}_obj0.adt").write_bytes(obj0_bytes)

        if with_obj1:
            d_rows = [(r[0], r[1], r[2]) for r in out_mddf]
            w_rows = [(r[1], r[2], r[3]) for r in out_modf]
            obj1_bytes = build_obj1_bytes(d_rows, w_rows)
            (out_maps_dir / f"{stem}_obj1.adt").write_bytes(obj1_bytes)

        if with_lod:
            lod_bytes = build_lod_bytes(mcvt_world_heights, holes_64)
            (out_maps_dir / f"{stem}_lod.adt").write_bytes(lod_bytes)

        wdl_tile_samples[(tile.tx, tile.ty)] = sample_wdl_heights_for_tile(mcvt_world_heights, holes_64)
        resolved_tiles.append((tile.tx, tile.ty))
        print(
            f"  [OK] {tile.path.name} -> {stem} "
            f"(tex={len(mdids)}, M2={len(out_mddf)}, WMO={len(out_modf)}, h_tex={sum(1 for x in mhids if x)})"
        )

    # Write WDT
    wdt_bytes, extra_fdids = build_retail_wdt(
        resolved_tiles,
        has_mccv=any_mccv,
        has_height_textures=height_textures,
        has_lod=with_lod,
        maid_mode=(wdt_mode == "maid"),
        base_fdid=base_fdid,
        map_name=map_name,
    )
    (out_maps_dir / f"{map_name}.wdt").write_bytes(wdt_bytes)

    if extra_fdids:
        csv_path = out_maps_dir / f"{map_name}_custom_listfile.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            for fid, p in extra_fdids:
                f.write(f"{fid};{p}\n")

    # Write or update WDL
    existing_wdl = existing_wdl_path.read_bytes() if (existing_wdl_path and existing_wdl_path.exists()) else None
    wdl_bytes = update_or_create_wdl(existing_wdl, wdl_tile_samples)
    (out_maps_dir / f"{map_name}.wdl").write_bytes(wdl_bytes)

    # Optional .noggitproj
    if noggitproj_dir:
        proj_ver_str = "Shadowlands" if target_version.lower() in ("shadowlands", "sl") else "Retail"
        proj = {
            "Project": {
                "ProjectName": map_name,
                "Bookmarks": [],
                "PinnedMaps": [{"MapId": map_id, "MapName": map_name}],
                "Client": {
                    "ClientPath": client_path,
                    "ClientVersion": proj_ver_str,
                },
            }
        }
        proj_file = noggitproj_dir / f"{map_name}.noggitproj"
        proj_file.write_text(json.dumps(proj, indent=2), encoding="utf-8")

    elapsed = round(time.time() - t_start, 2)
    report = {
        "map_name": map_name,
        "target_version": target_version,
        "wdt_mode": wdt_mode,
        "tiles_converted": [f"{tx}_{ty}" for tx, ty in resolved_tiles],
        "output_dir": str(out_maps_dir),
        "elapsed_seconds": elapsed,
        "resolution": {
            "exact": stats.exact,
            "alias": stats.alias,
            "basename": stats.basename,
            "fuzzy": stats.fuzzy,
            "missing": stats.missing,
            "height_textures_matched": stats.height_textures_found,
            "fuzzy_mappings": stats.fuzzy_mappings,
            "missing_assets": stats.missing_assets,
        },
    }
    report_path = out_maps_dir / f"{map_name}_conversion_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def collect_input_adts(inputs: List[str]) -> List[Path]:
    found: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            candidates = sorted(p.glob("*.adt"))
        elif p.is_file():
            candidates = [p]
        else:
            candidates = sorted(Path(g) for g in glob.glob(raw))
        for c in candidates:
            low = c.name.lower()
            if "_tex" in low or "_obj" in low or "_lod" in low:
                continue
            found.append(c)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert WotLK 3.3.5a ADTs/WDT/WDL into Modern Retail / Shadowlands split ADTs."
    )
    parser.add_argument("inputs", nargs="+", help="Input .adt file(s), glob(s), or directory.")
    parser.add_argument("--map-name", required=True, help="Output map name (e.g. lordaeronbefore).")
    parser.add_argument("--output", required=True, help="Output directory (or Noggit project folder with --noggitproj).")
    parser.add_argument("--listfile", nargs="+", required=True, help="One or more Retail CSV listfile paths.")
    parser.add_argument("--alias", type=Path, help="Optional JSON mapping custom paths -> Retail paths or FDIDs.")
    parser.add_argument("--wdt", type=Path, help="Optional source WotLK .wdt (reads Big Alpha flag).")
    parser.add_argument("--wdl", type=Path, help="Optional source .wdl to patch cleanly in-place.")
    parser.add_argument("--target", choices=["retail", "shadowlands"], default="retail")
    parser.add_argument("--missing-policy", choices=["skip", "zero", "fail"], default="skip")
    parser.add_argument("--no-fuzzy", action="store_true", help="Disable relocated/typo asset fallback matching.")
    parser.add_argument("--fuzzy-cutoff", type=float, default=0.84, help="Similarity cutoff for same-dir typo match.")
    parser.add_argument("--big-alpha", dest="big_alpha", action="store_true", default=None)
    parser.add_argument("--no-big-alpha", dest="big_alpha", action="store_false")
    parser.add_argument("--compress-alpha", action="store_true", help="Emit RLE-compressed MCAL (0x300) in _tex0.")
    parser.add_argument("--no-height-textures", action="store_true", help="Do not resolve *_h.blp MHID/MTXP.")
    parser.add_argument("--emit-mtxp", action="store_true", help="Always write MTXP chunk in _tex0.")
    parser.add_argument("--no-obj1", action="store_true", help="Skip generating _obj1.adt.")
    parser.add_argument("--with-lod", action="store_true", help="Generate _lod.adt terrain LOD files & set MPHD 0x100.")
    parser.add_argument("--no-mfbo", action="store_true", help="Do not synthesize MFBO if absent in WotLK tile.")
    parser.add_argument("--no-fix-uids", action="store_true", help="Keep raw UIDs without cross-tile collision repair.")
    parser.add_argument("--wdt-mode", choices=["path", "maid"], default="path")
    parser.add_argument("--base-fdid", type=int, default=5000000, help="Starting FileDataID for --wdt-mode maid.")
    parser.add_argument("--assets-dir", type=Path, help="Optional extracted assets folder to read local .m2 bounds.")
    parser.add_argument("--noggitproj", action="store_true", help="Write ShadowWorld .noggitproj into --output.")
    parser.add_argument("--client-path", default="", help="Client path stored in .noggitproj.")
    parser.add_argument("--map-id", type=int, default=0, help="Map ID stored in .noggitproj PinnedMaps.")

    args = parser.parse_args()
    input_adts = collect_input_adts(args.inputs)
    if not input_adts:
        print("Error: no source terrain .adt files found.", file=sys.stderr)
        return 1

    print(f"Loading listfile(s): {[str(x) for x in args.listfile]} ...")
    resolver = ListfileResolver(
        [Path(x) for x in args.listfile],
        alias_map=load_alias_file(args.alias),
        enable_fuzzy=not args.no_fuzzy,
        fuzzy_cutoff=args.fuzzy_cutoff,
    )
    print(f"Indexed {len(resolver.exact)} asset paths. Converting {len(input_adts)} tile(s) -> {args.target}...")

    out_root = Path(args.output).resolve()
    report = convert_tiles(
        input_paths=input_adts,
        output_dir=out_root,
        map_name=args.map_name,
        resolver=resolver,
        big_alpha_override=args.big_alpha,
        target_version=args.target,
        missing_policy=args.missing_policy,
        compress_alpha=args.compress_alpha,
        height_textures=not args.no_height_textures,
        emit_mtxp=args.emit_mtxp,
        with_obj1=not args.no_obj1,
        with_lod=args.with_lod,
        synth_mfbo=not args.no_mfbo,
        fix_uids=not args.no_fix_uids,
        wdt_mode=args.wdt_mode,
        base_fdid=args.base_fdid,
        existing_wdt_path=args.wdt,
        existing_wdl_path=args.wdl,
        assets_dir=args.assets_dir,
        noggitproj_dir=out_root if args.noggitproj else None,
        client_path=args.client_path,
        map_id=args.map_id,
    )
    r = report["resolution"]
    print(
        f"\nDone in {report['elapsed_seconds']}s | tiles={len(report['tiles_converted'])} "
        f"| exact={r['exact']} alias={r['alias']} relocated={r['basename']} fuzzy={r['fuzzy']} "
        f"missing={r['missing']} _h.blp={r['height_textures_matched']}"
    )
    if r["fuzzy_mappings"]:
        print("Relocated/Fuzzy mappings:")
        for k, v in sorted(r["fuzzy_mappings"].items()):
            print(f"  {k} -> {v}")
    if r["missing_assets"]:
        print("Unresolved assets:")
        for kind, lst in r["missing_assets"].items():
            for item in sorted(set(lst)):
                print(f"  [{kind}] {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
