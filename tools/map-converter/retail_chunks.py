#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Binary chunk helpers and WotLK -> Modern Retail ADT primitives."""

from __future__ import annotations

import math
import struct
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

TILE_SIZE = 17066.666666666668 / 32.0  # 533.3333333333334
CHUNK_SIZE = TILE_SIZE / 16.0          # 33.333333333333336
ZEROPOINT = 17066.666666666668
MAP_MIDPOINT = 17066.666

# MCNK header bitflags (MapHeaders.h)
MCNK_FLAG_MCSH = 0x1
MCNK_FLAG_IMPASS = 0x2
MCNK_FLAG_LQ_RIVER = 0x4
MCNK_FLAG_LQ_OCEAN = 0x8
MCNK_FLAG_LQ_MAGMA = 0x10
MCNK_FLAG_LQ_SLIME = 0x20
MCNK_FLAG_HAS_MCCV = 0x40
MCNK_FLAG_DO_NOT_FIX_ALPHA_MAP = 0x8000
MCNK_FLAG_HIGH_RES_HOLES = 0x10000

# MCLY layer bitflags
MCLY_USE_ALPHA_MAP = 0x100
MCLY_ALPHA_COMPRESSED = 0x200

# Modern MDDF / MODF flags
MDDF_ENTRY_IS_FILEDATA_ID = 0x40
MODF_ENTRY_IS_FILEDATA_ID = 0x08
MODF_USE_MODEL_SCALE = 0x04

WOTLK_KNOWN_MCNK_SUBS = {
    "MCVT", "MCCV", "MCNR", "MCLY", "MCRF",
    "MCSH", "MCAL", "MCLQ", "MCSE", "MCLV",
}


def tag_bytes(name: str) -> bytes:
    """Return little-endian reversed 4-byte chunk signature."""
    if len(name) != 4:
        raise ValueError(f"Chunk name must be 4 characters: {name!r}")
    return name[::-1].encode("ascii")


def make_chunk(name: str, payload: bytes = b"") -> bytes:
    """Pack a single IFF-style chunk with 4-byte tag and uint32 size."""
    return tag_bytes(name) + struct.pack("<I", len(payload)) + payload


def iter_chunks(
    data: bytes, start: int = 0, end: Optional[int] = None, skip_null: bool = True
) -> Iterator[Tuple[str, int, int, bytes]]:
    """Iterate (name, header_offset, payload_size, payload_bytes) over ADT/WDT chunks."""
    if end is None:
        end = len(data)
    pos = start
    while pos + 8 <= end:
        raw_tag = data[pos : pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        if raw_tag == b"\x00\x00\x00\x00" and size == 0:
            if skip_null:
                pos += 8
                continue
        if pos + 8 + size > end:
            raise ValueError(
                f"Truncated chunk {raw_tag!r} at 0x{pos:X} (size={size}, end={end})"
            )
        name = raw_tag[::-1].decode("latin1")
        yield name, pos, size, data[pos + 8 : pos + 8 + size]
        pos += 8 + size


def iter_wotlk_mcnk_subchunks(payload: bytes) -> Iterator[Tuple[str, int, int, bytes]]:
    """
    Iterate WotLK MCNK subchunks starting after the 128-byte MapChunkHeader.
    Corrects the classic MCNR size=435 + 13 uncounted padding bytes quirk.
    """
    pos = 128
    end = len(payload)
    while pos + 8 <= end:
        raw_tag = payload[pos : pos + 4]
        if raw_tag == b"\x00\x00\x00\x00":
            break
        name = raw_tag[::-1].decode("latin1")
        size = struct.unpack_from("<I", payload, pos + 4)[0]
        if pos + 8 + size > end:
            raise ValueError(
                f"MCNK subchunk {name!r} at {pos} overflows payload ({size} > {end - pos - 8})"
            )
        yield name, pos, size, payload[pos + 8 : pos + 8 + size]
        step = 8 + size
        if name == "MCNR" and size == 435 and pos + 8 + 448 <= end:
            next_off = pos + 8 + size
            next_tag = payload[next_off : next_off + 4][::-1].decode("latin1", "replace")
            after_pad_off = pos + 8 + 448
            after_pad_tag = (
                payload[after_pad_off : after_pad_off + 4][::-1].decode("latin1", "replace")
                if after_pad_off + 4 <= end
                else ""
            )
            if next_tag not in WOTLK_KNOWN_MCNK_SUBS and (
                after_pad_off == end or after_pad_tag in WOTLK_KNOWN_MCNK_SUBS
            ):
                step = 8 + 448
        pos += step


def expand_holes_low_to_high(low_mask: int) -> int:
    """Convert legacy 4x4 uint16 hole bitmask into Retail 8x8 uint64 high-res mask."""
    result = 0
    low = low_mask & 0xFFFF
    for y in range(4):
        for x in range(4):
            if low & (1 << (y * 4 + x)):
                first = y * 16 + x * 2
                result |= (3 << first) | (3 << (first + 8))
    return result


def compress_holes_high_to_low(high_mask: int) -> int:
    """
    Convert Retail 8x8 uint64 hole mask to coarse 4x4 uint16 bitmask (matching WDL logic:
    a 2x2 group is holed when all 4 fine cells inside it are holed).
    """
    result = 0
    for y in range(4):
        for x in range(4):
            first = y * 16 + x * 2
            quad = (3 << first) | (3 << (first + 8))
            if (high_mask & quad) == quad:
                result |= 1 << (y * 4 + x)
    return result


def decompress_rle_mcal(buf: bytes, start: int) -> Tuple[bytes, int]:
    """Decompress one 64x64 RLE-compressed MCAL alpha layer. Returns (4096B, end_offset)."""
    out = bytearray()
    pos = start
    end = len(buf)
    while len(out) < 4096:
        if pos >= end:
            raise ValueError("Truncated compressed MCAL stream")
        ctrl = buf[pos]
        pos += 1
        fill = (ctrl & 0x80) != 0
        count = ctrl & 0x7F
        if count == 0:
            continue
        remaining = 4096 - len(out)
        if count > remaining:
            count = remaining
        if fill:
            if pos >= end:
                raise ValueError("Truncated compressed MCAL fill value")
            val = buf[pos]
            pos += 1
            out.extend([val] * count)
        else:
            if pos + count > end:
                raise ValueError("Truncated compressed MCAL copy run")
            out.extend(buf[pos : pos + count])
            pos += count
    return bytes(out[:4096]), pos


def compress_rle_mcal(alpha4096: bytes) -> bytes:
    """Encode a 4096-byte 64x64 alpha layer using Blizzard ADT RLE compression."""
    if len(alpha4096) != 4096:
        raise ValueError(f"Expected 4096 alpha bytes, got {len(alpha4096)}")
    out = bytearray()
    n = 4096
    i = 0
    while i < n:
        # Look for >=3 identical consecutive bytes for fill run (up to 127)
        run = 1
        while i + run < n and run < 127 and alpha4096[i + run] == alpha4096[i]:
            run += 1
        if run >= 3:
            out.append(0x80 | run)
            out.append(alpha4096[i])
            i += run
        else:
            lit_start = i
            i += 1
            while i < n and (i - lit_start) < 127:
                # Stop before a fill run of length >= 3
                if i + 2 < n and alpha4096[i] == alpha4096[i + 1] == alpha4096[i + 2]:
                    break
                i += 1
            count = i - lit_start
            out.append(count)
            out.extend(alpha4096[lit_start:i])
    return bytes(out)


def decode_mcal_layer(
    mcal: bytes,
    offset: int,
    next_offset: Optional[int],
    layer_flags: int,
    big_alpha: bool,
    apply_edge_fix: bool,
) -> bytes:
    """
    Decode any WotLK MCAL layer format (4-bit 2048B, uncompressed 4096B, or RLE)
    into normalized 64x64 8-bit uncompressed bytes (4096 bytes).
    """
    if layer_flags & MCLY_ALPHA_COMPRESSED:
        alpha, _ = decompress_rle_mcal(mcal, offset)
        return alpha

    avail = (next_offset if next_offset is not None else len(mcal)) - offset
    if big_alpha or avail >= 4096:
        raw = bytearray(mcal[offset : offset + 4096])
        if len(raw) < 4096:
            raw.extend(b"\x00" * (4096 - len(raw)))
        if apply_edge_fix:
            for row in range(63):
                raw[row * 64 + 63] = raw[row * 64 + 62]
            raw[63 * 64 : 64 * 64] = raw[62 * 64 : 63 * 64]
        return bytes(raw)

    # Classic 4-bit per pixel (2048 bytes: 64x32 packed pairs)
    packed = mcal[offset : offset + 2048]
    out = bytearray(4096)
    for y in range(64):
        for x_pair in range(32):
            src_idx = y * 32 + x_pair
            b = packed[src_idx] if src_idx < len(packed) else 0
            lo = (b & 0x0F) * 17
            hi = ((b >> 4) & 0x0F) * 17
            out[y * 64 + x_pair * 2] = lo
            out[y * 64 + x_pair * 2 + 1] = hi
    if apply_edge_fix:
        for row in range(63):
            out[row * 64 + 63] = out[row * 64 + 62]
        out[63 * 64 : 64 * 64] = out[62 * 64 : 63 * 64]
    return bytes(out)


def upgrade_mcnr_payload(mcnr_payload: bytes) -> bytes:
    """Ensure MCNR payload is padded to 448 bytes (Retail size=448)."""
    if len(mcnr_payload) == 448:
        return mcnr_payload
    if len(mcnr_payload) >= 435:
        return mcnr_payload[:435].ljust(448, b"\x00")
    return mcnr_payload.ljust(448, b"\x00")


def server_position(pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Convert placement coordinates (pos) to server/LOD world coordinates."""
    return (MAP_MIDPOINT - pos[2], MAP_MIDPOINT - pos[0], pos[1])


def doodad_lod_extent(
    pos: Tuple[float, float, float],
    scale: int,
    local_bounds: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float]:
    """
    Compute MLDX 28-byte LodExtent (min3, max3, radius) for an M2 placement.
    If local_bounds ((min_x,y,z),(max_x,y,z)) aren't available from MD20,
    matches ShadowlandsADTCodec::new_doodad_extent heuristic (radius = 32 * scale/1024).
    """
    s = (scale if scale > 0 else 1024) / 1024.0
    if local_bounds is None:
        cx, cy, cz = server_position(pos)
        r = 32.0 * s
        return ((cx - r, cy - r, cz - r), (cx + r, cy + r, cz + r), r)

    # Approximate conservative sphere around transformed local AABB when rot not given,
    # or caller can pass precomputed world box. Here fallback sphere radius = half diagonal * scale.
    bmin, bmax = local_bounds
    dx = (bmax[0] - bmin[0]) * s
    dy = (bmax[1] - bmin[1]) * s
    dz = (bmax[2] - bmin[2]) * s
    r = max(1.0, 0.5 * math.sqrt(dx * dx + dy * dy + dz * dz))
    cx, cy, cz = server_position(pos)
    return ((cx - r, cy - r, cz - r), (cx + r, cy + r, cz + r), r)


def modf_world_extents_to_lod(
    ext_low: Tuple[float, float, float], ext_high: Tuple[float, float, float]
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float]:
    """Convert MODF CAaBox extents into Retail MLMX LodExtent (exact ShadowlandsADTCodec math)."""
    lo = (MAP_MIDPOINT - ext_high[2], MAP_MIDPOINT - ext_high[0], ext_low[1])
    hi = (MAP_MIDPOINT - ext_low[2], MAP_MIDPOINT - ext_low[0], ext_high[1])
    dx = hi[0] - lo[0]
    dy = hi[1] - lo[1]
    dz = hi[2] - lo[2]
    radius = math.sqrt(dx * dx + dy * dy + dz * dz)
    return lo, hi, radius


def convert_mclq_to_mh2o(mcnk_entries: List[Tuple[int, bytes]]) -> Optional[bytes]:
    """
    Synthesize an MH2O chunk from legacy per-chunk MCLQ liquid blocks (pre-WotLK / Classic ADTs)
    when top-level MH2O is absent.
    Each item in mcnk_entries is (mcnk_flags, mclq_payload).
    Returns MH2O payload bytes or None if no chunk has MCLQ.
    """
    has_any = any(len(p) >= 720 for _, p in mcnk_entries)
    if not has_any:
        return None

    headers = bytearray(256 * 12)
    extra = bytearray()
    base_data_offset = 256 * 12

    for idx, (cflags, mclq) in enumerate(mcnk_entries):
        if len(mclq) < 8 + 81 * 8 + 64:
            continue
        # Determine LiquidType ID matching client flags
        if cflags & MCNK_FLAG_LQ_MAGMA:
            ltype, lvf = 3, 2  # Magma
        elif cflags & MCNK_FLAG_LQ_SLIME:
            ltype, lvf = 4, 0  # Slime
        elif cflags & MCNK_FLAG_LQ_OCEAN:
            ltype, lvf = 2, 2  # Ocean
        else:
            ltype, lvf = 5, 0  # River

        min_h, max_h = struct.unpack_from("<ff", mclq, 0)
        verts_raw = mclq[8 : 8 + 81 * 8]
        tiles_raw = mclq[8 + 81 * 8 : 8 + 81 * 8 + 64]

        exists_mask = 0
        fish_mask = 0
        for ty in range(8):
            for tx in range(8):
                t = tiles_raw[ty * 8 + tx]
                if (t & 0x0F) != 0x0F:  # not "don't render"
                    bit = 1 << (ty * 8 + tx)
                    exists_mask |= bit
                    if not (t & 0x40):
                        fish_mask |= bit
        if exists_mask == 0:
            continue

        ofs_info = base_data_offset + len(extra)
        info_pos = len(extra)
        extra.extend(b"\x00" * 24)

        ofs_attrs = base_data_offset + len(extra)
        extra.extend(struct.pack("<QQ", fish_mask, 0))

        ofs_exists = base_data_offset + len(extra)
        extra.extend(struct.pack("<Q", exists_mask))

        ofs_vdata = base_data_offset + len(extra)
        heights = []
        depths = bytearray(81)
        for vi in range(81):
            b0 = verts_raw[vi * 8]
            vh = struct.unpack_from("<f", verts_raw, vi * 8 + 4)[0]
            if not math.isfinite(vh):
                vh = max_h
            heights.append(vh)
            depths[vi] = b0

        if lvf == 2:
            extra.extend(bytes(depths))
        else:
            extra.extend(struct.pack("<81f", *heights))
            extra.extend(bytes(depths))

        struct.pack_into(
            "<HHffBBBBII",
            extra,
            info_pos,
            ltype,
            lvf,
            min_h,
            max_h,
            0,
            0,
            8,
            8,
            ofs_exists,
            ofs_vdata,
        )
        struct.pack_into("<3I", headers, idx * 12, ofs_info, 1, ofs_attrs)

    return bytes(headers) + bytes(extra)
