# WoW Map Forge (WotLK 3.3.5a) 🗺️⚡

> **Lightning-fast, full-pipeline procedural map generator & server collision compiler for World of Warcraft 3.3.5a (build 12340).**

Generate brand new, production-ready world maps with natural terrain, high-resolution textures, seamless water, and server collision files in **under 30 seconds** — without manual Noggit sculpturing and without heavy Trinity/AzerothCore extractors.

---

## ✨ Features

- **⚡ End-to-End Pipeline in One Command:** Generates client `.WDT` + all `.ADT` tiles + server `.map` collision files simultaneously.
- **🎨 Big Alpha (0x04) High-Res Texturing:** 64x64 uncompressed 8-bit alpha blending maps (`0x100`) for crisp, modern ground textures.
- **🌊 Seamless MH2O Water System:**
  - Automatic depth-gradient vertex alpha blending ($0..255$).
  - Eliminates the infamous 33-meter square chunk saw-tooth cut artifacts along coastlines.
  - Automatically deactivates liquid chunks on dry land.
- **☀️ Analytical Normal Vectors (MCNR):** Partial differential lighting normals precomputed via central differences for cinematic sun shadows.
- **🏔️ Multiple Terrain Generation Modes:**
  - `island`: Organic ocean-surrounded landmass with smooth beach shelves and central peaks.
  - `lake`: Vast freshwater basin with building plateaus and alpine mountain backdrops.
  - `heightmap`: Import **any** 8-bit/16-bit grayscale PNG/BMP/TIFF image into true WoW 3D terrain!
- **🚀 Built-in Server Collision Compiler:** Directly converts ADT geometry into binary `.map` files for AzerothCore / TrinityCore in 0.05s per tile — no game client or MPQ extraction required.
- **🛡️ 100% Stable:** Clean binary chunks (`MVER`, `MHDR`, `MCIN`, `MTEX`, `MCNK`, `MCVT`, `MCNR`, `MCLY`, `MCAL`, `MH2O`). Opens smoothly in **Noggit Red** with zero crashes.

---

## 📦 Requirements

- Python 3.8+
- Dependencies:
  ```bash
  pip install numpy pillow
  ```

---

## 🚀 Quick Start

### 1. Generate a Tropical Island (2x2 tiles)
```bash
python forge.py --name tropical_island --size 2x2 --preset island --peak-height 120 --island-radius 280
```

### 2. Generate an Epic Lake Basin (3x3 tiles, 1.6 km x 1.6 km)
```bash
python forge.py --name mountain_lake --size 3x3 --preset lake --peak-height 180
```

### 3. Generate Terrain from a Custom PNG Heightmap
```bash
python forge.py --name from_image --size 3x3 --heightmap my_world.png --peak-height 150 --min-height -20
```

---

## ⚙️ CLI Options

| Argument | Type | Default | Description |
|---|---|---|---|
| `--name` | string | `custom_map` | Internal map folder name (e.g. `emerald_island`) |
| `--map-id` | integer | `955` | Target Map ID from client/server `Map.dbc` |
| `--size` | string | `3x3` | Grid size in tiles (`1x1`, `2x2`, `3x3`, `4x4`) |
| `--preset` | string | `island` | Terrain generation preset: `island` or `lake` |
| `--heightmap` | path | `None` | Path to a grayscale PNG/BMP image to sample elevation |
| `--peak-height`| float | `160.0` | Maximum mountain peak height in meters |
| `--island-radius`| float | `380.0` | Shoreline radius in meters (for island preset) |
| `--water-level`| float | `0.0` | Sea / lake water surface height Z in meters |
| `--no-water` | flag | `false` | Disable liquid generation completely |
| `--no-server-maps` | flag | `false` | Skip compiling server `.map` files |
| `--output` | path | `./output/<name>` | Output directory for client ADT & WDT |
| `--server-maps` | path | `./output/maps` | Output directory for server `.map` files |
| `--config` | path | `None` | Load settings from a JSON config file |

---

## 🎮 How to Put It in the Game

1. **Client:**
   - Copy the generated `.wdt` and all `.adt` files into your MPQ patch: `World\Maps\<map_name>\`
   - Register your Map ID in `DBFilesClient\Map.dbc`.
2. **Server:**
   - Copy the generated `*.map` files into your server directory: `bin/Data/maps/`
   - Register your Map ID in `worldserver` DBC and start the server.
3. **Noggit Red:**
   - Open your project in Noggit Red, select your map from the list, and immediately start placing buildings, doodads, and roads on a pristine natural canvas!

---

## 📜 License
MIT License. Created for the World of Warcraft modding and private server community.
