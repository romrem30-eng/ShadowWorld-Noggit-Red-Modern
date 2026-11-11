# WotLK 3.3.5a -> Modern Retail / Shadowlands Map Converter

Pure-Python CLI toolchain converting monolithic WotLK 3.3.5a ADTs (`.adt`, `.wdt`, `.wdl`) into Modern Retail (11.x/12.x) and Shadowlands (9.2.7) split ADTs compatible with ShadowWorld Noggit-Red-Modern.

## Generated Outputs

Per map tile (`<map>_X_Y`):
- `<map>_X_Y.adt` — Root terrain file (`MVER=18`, `MHDR` with `ofsMFBO`/`ofsMH2O`, `MH2O` or upgraded legacy `MCLQ` water, `256` `MCNK` chunks with `0x10000` high-res 8x8 holes, 448-byte `MCNR`, `MCVT`, optional `MCCV` with `ofsMCCV=1180`, `MCSE`, and `MFBO`). Pass-through idempotent under `Noggit::Formats::ADT::finalizeRetailRootLayout`.
- `<map>_X_Y_tex0.adt` — Split texture file (`MVER=18`, `MAMP=0`, `MDID`, `MHID` height textures `*_h.blp`, optional `MTXP`, `256` zero-header `MCNK`s with `MCLY` + `MCAL` supporting 4-bit packed nibble, 4096-byte Big Alpha, and Blizzard RLE `0x300`).
- `<map>_X_Y_obj0.adt` — Split placements file (`MVER=18`, `MDDF` with `0x40` FileDataID flag and normalized nonzero `scale`, `MODF` with `0x0C` FileDataID+Scale flags and normalized `scale`, plus remapped `MCRD`/`MCRW`).
- `<map>_X_Y_obj1.adt` — Object LOD file (`MLFD`, `MLDD`, `MLDX`, `MLDB`, `MLDL`, `MLMD`, `MLMX`, `MLMB`) sorted descending by radius and validated against `RetailObjectTables::validate` and `UidRepair::inspect`.
- `<map>_X_Y_lod.adt` — Optional (`--with-lod`) 4-level terrain quadtree + skirts (`MLHD`, `MLVH`, `MLVI`, `MLSI`, `MLLL`, `MLND`) in `YOuterZeroCorner` layout with manifold hole-handling.
- `<map>.wdt` — Path-addressed (`--wdt-mode path`) or MAID-addressed (`--wdt-mode maid`, emits `_custom_listfile.csv`) WDT with `MPHD` (`0x04 | 0x08` + optional `0x02`, `0x80`, `0x100`, `0x200`) and `MAIN`.
- `<map>.wdl` — Updated or synthesized `MARE` (17x17 outer + 16x16 inner) + `MAHO` hole masks preserving untouched tiles and unknown chunks.
- `<map>.noggitproj` — Optional (`--noggitproj`) project descriptor for ShadowWorld Noggit-Red-Modern.

## Usage

```bash
python tools/map-converter/wotlk_to_retail.py D:/path/to/wotlk_adts \
  --map-name lordaeronbefore \
  --output D:/path/to/retail_project \
  --listfile D:/path/to/listfile.csv \
  --wdt D:/path/to/lordaeronbefore.wdt \
  --wdl D:/path/to/lordaeronbefore.wdl \
  --target retail \
  --with-lod \
  --noggitproj
```

Custom/retroported assets not present in stock Retail listfiles can be supplied with multiple `--listfile` CSVs and/or `--alias aliases.json`.

## Procedural Retail Map Forge

The sibling `tools/map-forge/forge.py` generator also supports `--format {wotlk,retail,both}`:

```bash
python tools/map-forge/forge.py \
  --name custom_island \
  --size 3x3 \
  --format retail \
  --listfile D:/path/to/listfile.csv \
  --noggitproj
```

## Tests

```bash
python tools/map-converter/test_converter.py
```
