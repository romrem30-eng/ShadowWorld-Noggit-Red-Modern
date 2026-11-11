#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit and synthetic end-to-end tests for tools/map-converter."""

from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from listfile_resolver import ListfileResolver, normalize_wow_path
from lod_and_world import build_lod_bytes, build_obj1_bytes, build_retail_wdt, update_or_create_wdl
from retail_chunks import (
    compress_holes_high_to_low,
    compress_rle_mcal,
    convert_mclq_to_mh2o,
    decode_mcal_layer,
    expand_holes_low_to_high,
    iter_chunks,
    iter_wotlk_mcnk_subchunks,
    make_chunk,
)
from wotlk_to_retail import convert_tiles, parse_wotlk_adt


def _build_synthetic_wotlk_adt(
    tx: int = 33,
    ty: int = 28,
    include_mccv: bool = True,
    with_mcnr_padding: bool = True,
    with_null_after_mh2o: bool = True,
) -> bytes:
    """Construct a valid WotLK 3.3.5a ADT exercising edge cases seen in custom maps."""
    textures = [
        b"Tileset\\Lordaeron\\LordaeronGrass01.blp",
        b"Tileset\\Lordaeron\\LordaeronDirt01.blp",
    ]
    mtex = b"\x00".join(textures) + b"\x00"

    m2s = [b"World\\Doodad\\Tree01.mdx", b"World\\Custom\\MissingProp.m2"]
    mmdx = bytearray()
    mmid_offs = []
    for m in m2s:
        mmid_offs.append(len(mmdx))
        mmdx.extend(m)
        mmdx.append(0)
    mmid = struct.pack(f"<{len(mmid_offs)}I", *mmid_offs)

    wmos = [b"World\\Wmo\\Building01.wmo"]
    mwmo = bytearray(wmos[0] + b"\x00")
    mwid = struct.pack("<I", 0)

    # M2 #0 scale=0 (common WotLK quirk), M2 #1 missing asset, colliding UID=100
    mddf = struct.pack("<II6fHH", 0, 100, 1000.0, 2000.0, 50.0, 0.0, 45.0, 0.0, 0, 0)
    mddf += struct.pack("<II6fHH", 1, 100, 1010.0, 2000.0, 50.0, 0.0, 0.0, 0.0, 1024, 0)
    # WMO #0 colliding UID=100, scale=0
    modf = struct.pack(
        "<II12f4H",
        0,
        100,
        1020.0, 2020.0, 60.0,
        0.0, 90.0, 0.0,
        1000.0, 2000.0, 50.0,
        1040.0, 2040.0, 90.0,
        0x0C,
        0,
        0,
        0,
    )

    mcin_entries = bytearray()
    mcnks_bytes = bytearray()

    for cy in range(16):
        for cx in range(16):
            subs = bytearray()
            # MCVT (145 floats)
            mcvt = struct.pack("<145f", *([float(cx + cy) * 0.25] * 145))
            subs.extend(make_chunk("MCVT", mcvt))

            if include_mccv:
                mccv = bytearray()
                for _ in range(145):
                    mccv.extend(bytes((127, 127, 127, 127)))
                subs.extend(make_chunk("MCCV", bytes(mccv)))

            # MCNR: optionally emit WotLK-style reported 435 bytes + 13 trailing padding bytes
            mcnr_payload = bytes([0, 0, 127] * 145)
            if with_mcnr_padding:
                subs.extend(b"RNCM" + struct.pack("<I", 435) + mcnr_payload + b"\x00" * 13)
            else:
                subs.extend(make_chunk("MCNR", mcnr_payload + b"\x00" * 13))

            # MCLY + MCAL + MCSH
            mcly = struct.pack("<IIII", 0, 0, 0, 0) + struct.pack("<IIII", 1, 0x100, 0, 42)
            subs.extend(make_chunk("MCLY", mcly))
            mcal = bytes([(i % 256) for i in range(4096)])
            subs.extend(make_chunk("MCAL", mcal))
            subs.extend(make_chunk("MCSE", b""))

            # MCNK 128-byte header
            px = 17066.666 - (ty * 16 + cy) * 33.333333
            py = 17066.666 - (tx * 16 + cx) * 33.333333
            pz = 100.0
            has_refs = (cx == 0 and cy == 0)
            if has_refs:
                # Reference M2 #0, skipped M2 #1, WMO #0
                mcrf = struct.pack("<3I", 0, 1, 0)
                # Rebuild subs cleanly with MCRF after MCNR
                subs = bytearray()
                subs.extend(make_chunk("MCVT", mcvt))
                if include_mccv:
                    subs.extend(make_chunk("MCCV", bytes([127] * (145 * 4))))
                if with_mcnr_padding:
                    subs.extend(b"RNCM" + struct.pack("<I", 435) + mcnr_payload + b"\x00" * 13)
                else:
                    subs.extend(make_chunk("MCNR", mcnr_payload + b"\x00" * 13))
                subs.extend(make_chunk("MCLY", mcly))
                subs.extend(make_chunk("MCRF", mcrf))
                subs.extend(make_chunk("MCAL", mcal))
                subs.extend(make_chunk("MCSE", b""))

            flags = 0x41 if include_mccv else 0x01
            holes_low = 0x0001 if (cx == 1 and cy == 1) else 0
            hdr = bytearray(128)
            struct.pack_into("<10I", hdr, 0, flags, cx, cy, 2, 2 if has_refs else 0, 0, 0, 0, 0, 0)
            struct.pack_into("<II", hdr, 0x34, 193, 1 if has_refs else 0)  # areaid, nMapObjRefs
            struct.pack_into("<HH", hdr, 0x3C, holes_low, 0)
            struct.pack_into("<3f", hdr, 0x68, px, py, pz)

            mcnk_chunk = make_chunk("MCNK", bytes(hdr) + bytes(subs))
            mcin_entries.extend(struct.pack("<IIII", 0, len(mcnk_chunk), 0, 0))
            mcnks_bytes.extend(mcnk_chunk)

    mh2o_header = bytearray(256 * 12)
    # Chunk 0 has 1 flat water layer at height 10.0
    struct.pack_into("<III", mh2o_header, 0, 256 * 12, 1, 0)
    inst = struct.pack("<HHff4BII", 5, 0, 10.0, 10.0, 0, 0, 8, 8, 0, 0)
    mh2o_payload = bytes(mh2o_header) + inst

    body = bytearray()
    body.extend(make_chunk("MCIN", bytes(mcin_entries)))
    body.extend(make_chunk("MTEX", mtex))
    body.extend(make_chunk("MMDX", bytes(mmdx)))
    body.extend(make_chunk("MMID", mmid))
    body.extend(make_chunk("MWMO", bytes(mwmo)))
    body.extend(make_chunk("MWID", mwid))
    body.extend(make_chunk("MDDF", mddf))
    body.extend(make_chunk("MODF", modf))
    body.extend(make_chunk("MH2O", mh2o_payload))
    if with_null_after_mh2o:
        body.extend(b"\x00" * 8)
    body.extend(mcnks_bytes)

    return (
        make_chunk("MVER", struct.pack("<I", 18))
        + make_chunk("MHDR", bytes(64))
        + bytes(body)
    )


class ConverterTests(unittest.TestCase):
    def test_holes_round_trip(self) -> None:
        for bit in range(16):
            low = 1 << bit
            high = expand_holes_low_to_high(low)
            self.assertNotEqual(high, 0)
            self.assertEqual(compress_holes_high_to_low(high), low)
        self.assertEqual(expand_holes_low_to_high(0xFFFF), 0xFFFFFFFFFFFFFFFF)

    def test_mcal_rle_round_trip(self) -> None:
        samples = [
            bytes([0] * 4096),
            bytes([255] * 4096),
            bytes([(i * 17) & 0xFF for i in range(4096)]),
            bytes(([42] * 300) + [(i % 97) for i in range(500)] + ([99] * 3296)),
        ]
        for raw in samples:
            rle = compress_rle_mcal(raw)
            decoded = decode_mcal_layer(rle, 0, len(rle), 0x300, big_alpha=True, apply_edge_fix=False)
            self.assertEqual(decoded, raw)

        # Legacy 2048-byte packed 4-bit nibble alpha
        packed_nibble = bytes([0xFA] * 2048)
        up = decode_mcal_layer(packed_nibble, 0, 2048, 0x100, big_alpha=False, apply_edge_fix=False)
        self.assertEqual(len(up), 4096)
        self.assertEqual(up[0], 0xAA)
        self.assertEqual(up[1], 0xFF)

    def test_mclq_upgrade_to_mh2o(self) -> None:
        # Build minimal single-chunk legacy MCLQ (height 15.5) with MCNK flag 0x04 (river)
        verts = bytearray()
        for _ in range(81):
            verts.extend(struct.pack("<4Bf", 0, 0, 0, 255, 15.5))
        tiles = bytes([0x00] * 64)
        mclq = struct.pack("<ff", 15.5, 15.5) + bytes(verts) + tiles + struct.pack("<I", 0) + bytes(84)
        pairs = [(0x04, mclq)] + [(0x00, b"")] * 255
        mh2o = convert_mclq_to_mh2o(pairs)
        self.assertGreater(len(mh2o), 256 * 12)
        ofs, cnt, _ = struct.unpack_from("<3I", mh2o, 0)
        self.assertEqual(cnt, 1)
        self.assertGreater(ofs, 0)

    def test_listfile_resolver_modes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            csv = Path(td) / "listfile.csv"
            csv.write_text(
                "\n".join(
                    [
                        "1001;tileset/lordaeron/lordaerongrass01.blp",
                        "1002;tileset/lordaeron/lordaerongrass01_h.blp",
                        "1003;tileset/lordaeron/lordaerondirt01.blp",
                        "2001;world/expansion05/doodads/ashran/6as_a_towershack_set_1.m2",
                        "2002;world/doodad/tree01.m2",
                        "2003;world/expansion07/doodads/riverzone/8riv_farmcorn_01.m2",
                        "3001;world/wmo/building01.wmo",
                        "4001;world/custom/aliased.wmo",
                    ]
                ),
                encoding="utf-8",
            )
            res = ListfileResolver([csv], alias_map={"world/old/my.wmo": 4001})
            self.assertEqual(res.resolve("World\\Doodad\\Tree01.MDX", is_model=True).fdid, 2002)
            self.assertEqual(res.resolve("World\\Doodad\\Tree01.MDX", is_model=True).method, "exact")
            self.assertEqual(res.resolve_height_texture("tileset/lordaeron/lordaerongrass01.blp"), 1002)
            self.assertEqual(res.resolve("world/old/my.wmo").fdid, 4001)
            self.assertEqual(res.resolve("world/relocated/6as_a_towershack_set_1.m2", True).fdid, 2001)
            fuzzy = res.resolve("world/expansion07/doodads/riverzone/8riv_farmbcorn_01.m2", True)
            self.assertEqual(fuzzy.fdid, 2003)
            self.assertEqual(fuzzy.method, "fuzzy")
            self.assertEqual(normalize_wow_path("A\\\\B\\C.MDL", is_model=True), "a/b/c.m2")

    def test_end_to_end_wotlk_to_retail_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root_tmp = Path(td)
            csv = root_tmp / "listfile.csv"
            csv.write_text(
                "\n".join(
                    [
                        "1001;tileset/lordaeron/lordaerongrass01.blp",
                        "1002;tileset/lordaeron/lordaerongrass01_h.blp",
                        "1003;tileset/lordaeron/lordaerondirt01.blp",
                        "2002;world/doodad/tree01.m2",
                        "3001;world/wmo/building01.wmo",
                    ]
                ),
                encoding="utf-8",
            )
            adt_path = root_tmp / "testmap_33_28.adt"
            adt_path.write_bytes(_build_synthetic_wotlk_adt())

            parsed = parse_wotlk_adt(adt_path)
            self.assertEqual(len(parsed.chunks), 256)
            self.assertIsNotNone(parsed.mh2o)

            resolver = ListfileResolver([csv])
            out_dir = root_tmp / "out"
            report = convert_tiles(
                input_paths=[adt_path],
                output_dir=out_dir,
                map_name="testmap",
                resolver=resolver,
                big_alpha_override=True,
                target_version="retail",
                with_obj1=True,
                with_lod=True,
                noggitproj_dir=out_dir,
                map_id=999,
            )
            self.assertEqual(report["tiles_converted"], ["33_28"])
            self.assertEqual(report["resolution"]["missing"], 1)  # MissingProp.m2

            maps_dir = out_dir / "world" / "maps" / "testmap"
            self.assertTrue((out_dir / "testmap.noggitproj").exists())
            root_bytes = (maps_dir / "testmap_33_28.adt").read_bytes()
            tex0_bytes = (maps_dir / "testmap_33_28_tex0.adt").read_bytes()
            obj0_bytes = (maps_dir / "testmap_33_28_obj0.adt").read_bytes()
            obj1_bytes = (maps_dir / "testmap_33_28_obj1.adt").read_bytes()
            lod_bytes = (maps_dir / "testmap_33_28_lod.adt").read_bytes()
            wdt_bytes = (maps_dir / "testmap.wdt").read_bytes()
            wdl_bytes = (maps_dir / "testmap.wdl").read_bytes()

            # Check root MCNK offsets & holes
            root_chunks = list(iter_chunks(root_bytes))
            mcnks = [c for c in root_chunks if c[0] == "MCNK"]
            self.assertEqual(len(mcnks), 256)
            for idx, (_, _, _, payload) in enumerate(mcnks):
                hdr = payload[:128]
                flags = struct.unpack_from("<I", hdr, 0)[0]
                self.assertTrue(flags & 0x10000)
                ofs_height = struct.unpack_from("<I", hdr, 0x14 + 4)[0]
                ofs_mccv = struct.unpack_from("<I", hdr, 0x74)[0]
                self.assertEqual(ofs_mccv, 1180)
                if idx == 1 * 16 + 1:
                    holes_u64 = struct.unpack_from("<Q", hdr, 0x14)[0]
                    self.assertNotEqual(holes_u64, 0)

            # Check OBJ0 MDDF 0x40 + MODF 0x40|0x80 + UID collision repaired + MCRD/MCRW pruned
            obj0_map = {n: p for n, _, _, p in iter_chunks(obj0_bytes)}
            self.assertEqual(len(obj0_map["MDDF"]), 36)  # Skipped missing prop!
            mddf_uid = struct.unpack_from("<I", obj0_map["MDDF"], 4)[0]
            mddf_scale, mddf_flags = struct.unpack_from("<HH", obj0_map["MDDF"], 32)
            self.assertEqual(mddf_scale, 1024)
            self.assertEqual(mddf_flags & 0x40, 0x40)

            self.assertEqual(len(obj0_map["MODF"]), 64)
            modf_uid = struct.unpack_from("<I", obj0_map["MODF"], 4)[0]
            modf_flags, _, _, modf_scale = struct.unpack_from("<4H", obj0_map["MODF"], 56)
            self.assertNotEqual(mddf_uid, modf_uid)  # UID 100 collision resolved!
            self.assertEqual(modf_scale, 1024)
            self.assertEqual(modf_flags & 0x0C, 0x0C)

            # Chunk 0 should have 1 valid MCRD (index 0) & 1 valid MCRW (index 0)
            obj_mcnks = [p for n, _, _, p in iter_chunks(obj0_bytes) if n == "MCNK"]
            c0_subs = {n: p for n, _, _, p in iter_chunks(obj_mcnks[0])}
            self.assertEqual(struct.unpack("<I", c0_subs["MCRD"]), (0,))
            self.assertEqual(struct.unpack("<I", c0_subs["MCRW"]), (0,))

            # Check TEX0 FDIDs + MTXP
            tex_map = {n: p for n, _, _, p in iter_chunks(tex0_bytes)}
            self.assertEqual(struct.unpack("<2I", tex_map["MDID"]), (1001, 1003))
            self.assertEqual(struct.unpack("<2I", tex_map["MHID"]), (1002, 0))
            self.assertEqual(len(tex_map["MTXP"]), 32)

            # Check OBJ1 & LOD & WDT & WDL presence
            self.assertTrue(any(n == "MLFD" for n, _, _, _ in iter_chunks(obj1_bytes)))
            self.assertTrue(any(n == "MLND" for n, _, _, _ in iter_chunks(lod_bytes)))
            self.assertTrue(any(n == "MAIN" for n, _, _, _ in iter_chunks(wdt_bytes)))
            self.assertTrue(any(n == "MARE" for n, _, _, _ in iter_chunks(wdl_bytes)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
