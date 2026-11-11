#!/usr/bin/env python3
"""
ADT to MAP converter for AzerothCore WotLK 3.3.5a
Reads WoW ADT terrain files directly from disk (no MPQ needed)
and produces .map files compatible with the server.

Usage: adt_to_map.py <input.adt> <output.map> [build_number]
"""

import struct
import sys
from pathlib import Path

# Constants from System.cpp
ADT_CELLS_PER_GRID = 16
ADT_CELL_SIZE = 8
ADT_GRID_SIZE = ADT_CELLS_PER_GRID * ADT_CELL_SIZE

MAP_MAGIC = b'MAPS'
MAP_VERSION_MAGIC = 9
MAP_AREA_MAGIC = b'AREA'
MAP_HEIGHT_MAGIC = b'MHGT'
MAP_LIQUID_MAGIC = b'MLIQ'

MAP_AREA_NO_AREA = 0x0001
MAP_HEIGHT_NO_HEIGHT = 0x0001
MAP_HEIGHT_AS_INT16 = 0x0002
MAP_HEIGHT_AS_INT8 = 0x0004
MAP_HEIGHT_HAS_FLIGHT_BOUNDS = 0x0008

MAP_LIQUID_NO_TYPE = 0x0001
MAP_LIQUID_NO_HEIGHT = 0x0002

LIQUID_TYPE_WATER = 0
LIQUID_TYPE_OCEAN = 1
LIQUID_TYPE_MAGMA = 2
LIQUID_TYPE_SLIME = 3

MAP_LIQUID_TYPE_NO_WATER = 0x00
MAP_LIQUID_TYPE_WATER = 0x01
MAP_LIQUID_TYPE_OCEAN = 0x02
MAP_LIQUID_TYPE_MAGMA = 0x04
MAP_LIQUID_TYPE_SLIME = 0x08
MAP_LIQUID_TYPE_DARK_WATER = 0x10

CONF_ALLOW_HEIGHT_LIMIT = True
CONF_USE_MIN_HEIGHT = -500.0
CONF_ALLOW_FLOAT_TO_INT = True
CONF_FLOAT_TO_INT8_LIMIT = 2.0
CONF_FLOAT_TO_INT16_LIMIT = 2048.0
CONF_FLAT_HEIGHT_DELTA_LIMIT = 0.005
CONF_FLAT_LIQUID_DELTA_LIMIT = 0.001

# Liquid type sound bank mapping (loaded from LiquidType.dbc)
liquid_types = {}  # liquid_id -> sound_bank

def load_liquid_type_dbc(dbc_path):
    """Load LiquidType.dbc to map liquid types to sound banks."""
    global liquid_types
    with open(dbc_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'WDBC':
            raise ValueError(f"Invalid DBC magic: {magic}")
        num_records = struct.unpack('<I', f.read(4))[0]
        num_fields = struct.unpack('<I', f.read(4))[0]
        record_size = struct.unpack('<I', f.read(4))[0]
        string_size = struct.unpack('<I', f.read(4))[0]
        
        # Read records
        data_start = f.tell()
        for i in range(num_records):
            f.seek(data_start + i * record_size)
            liquid_id = struct.unpack('<I', f.read(4))[0]
            f.read(4)  # type
            f.read(4)  # sound_id
            sound_bank = struct.unpack('<I', f.read(4))[0]
            liquid_types[liquid_id] = sound_bank
    
    print(f"Loaded {len(liquid_types)} liquid types from DBC")

def select_uint8_step(max_diff):
    return 255.0 / max_diff if max_diff != 0 else 0

def select_uint16_step(max_diff):
    return 65535.0 / max_diff if max_diff != 0 else 0

def read_chunk(data, offset):
    """Read a chunk header (magic + size)."""
    magic = data[offset:offset+4]
    size = struct.unpack_from('<I', data, offset + 4)[0]
    return magic, size

def find_chunk(data, offset, magic):
    """Find a chunk by magic starting from offset."""
    pos = offset
    while pos < len(data):
        chunk_magic, chunk_size = read_chunk(data, pos)
        if chunk_magic == magic:
            return pos, chunk_size
        pos += 8 + chunk_size
    return None, None

class ADTParser:
    def __init__(self, data):
        self.data = data
        self.version = None
        self.mhdr = None
        self.mcin = None
        self.mh2o = None
        self.mfbo = None
        self.cells = []  # list of MCNK chunks
        
    def parse(self):
        # Parse MVER
        magic, size = read_chunk(self.data, 0)
        if magic != b'REVM':
            raise ValueError(f"Invalid ADT version chunk: {magic}")
        self.version = struct.unpack_from('<I', self.data, 8)[0]
        
        # Parse MHDR
        mhdr_offset = 12  # after MVER
        magic, size = read_chunk(self.data, mhdr_offset)
        if magic != b'RDHM':
            raise ValueError(f"Invalid MHDR chunk: {magic}")
        self.mhdr = self.data[mhdr_offset:mhdr_offset + 8 + size]
        
        # MHDR structure: magic(4) + size(4) + flags(4) + offsMCIN(4) + ... + offsMFBO(4) + offsMH2O(4) + ...
        # Offsets are relative to the flags field (mhdr_offset + 8)
        flags_offset = mhdr_offset + 8
        mfbo_flags = struct.unpack_from('<I', self.data, flags_offset + 0)[0]
        offs_mcin = struct.unpack_from('<I', self.data, flags_offset + 4)[0]
        offs_mfbo = struct.unpack_from('<I', self.data, flags_offset + 36)[0]
        offs_mh2o = struct.unpack_from('<I', self.data, flags_offset + 40)[0]
        
        # Parse MCIN (cell index)
        if offs_mcin:
            mcin_abs = flags_offset + offs_mcin
            self.mcin = self.data[mcin_abs:mcin_abs + 8 + 256*16]  # 16x16 cells, each with 16 bytes
            # Parse individual cells
            for i in range(ADT_CELLS_PER_GRID):
                for j in range(ADT_CELLS_PER_GRID):
                    cell_offset_in_mcin = 8 + (i * ADT_CELLS_PER_GRID + j) * 16
                    offs_mcnk = struct.unpack_from('<I', self.mcin, cell_offset_in_mcin)[0]
                    if offs_mcnk:
                        # MCNK offset in MCIN is an absolute offset from start of ADT file
                        mcnk_abs = offs_mcnk
                        self.cells.append((i, j, mcnk_abs))
        
        # Parse MH2O (liquid data)
        if offs_mh2o:
            mh2o_abs = flags_offset + offs_mh2o
            self.mh2o = mh2o_abs
        
        # Parse MFBO (flight bounds)
        if mfbo_flags & 1 and offs_mfbo:
            mfbo_abs = flags_offset + offs_mfbo
            self.mfbo = mfbo_abs
    
    def get_mcnk(self, i, j):
        """Get MCNK chunk data for cell (i, j)."""
        for ci, cj, mcnk_abs in self.cells:
            if ci == i and cj == j:
                return mcnk_abs
        return None
    
    def get_mcv_t(self, mcnk_offset):
        """Get MCVT (height map) from MCNK."""
        offs_mcv_t = struct.unpack_from('<I', self.data, mcnk_offset + 28)[0]
        if offs_mcv_t == 0:
            return None
        mcv_t_abs = mcnk_offset + offs_mcv_t
        # MCVT: magic(4) + size(4) + 145 floats
        return mcv_t_abs + 8  # skip header
    
    def get_mclq(self, mcnk_offset):
        """Get MCLQ (old liquid) from MCNK."""
        offs_mclq = struct.unpack_from('<I', self.data, mcnk_offset + 104)[0]
        size_mclq = struct.unpack_from('<I', self.data, mcnk_offset + 108)[0]
        if offs_mclq == 0 or size_mclq <= 8:
            return None, 0
        return mcnk_offset + offs_mclq, size_mclq

def convert_adt_to_map(adt_path, map_path, build=12340):
    """Convert ADT file to MAP file."""
    with open(adt_path, 'rb') as f:
        data = f.read()
    
    adt = ADTParser(data)
    adt.parse()
    
    # Initialize output arrays
    area_ids = [[0] * ADT_CELLS_PER_GRID for _ in range(ADT_CELLS_PER_GRID)]
    v8 = [[0.0] * ADT_GRID_SIZE for _ in range(ADT_GRID_SIZE)]
    v9 = [[0.0] * (ADT_GRID_SIZE + 1) for _ in range(ADT_GRID_SIZE + 1)]
    
    liquid_show = [[False] * ADT_GRID_SIZE for _ in range(ADT_GRID_SIZE)]
    liquid_flags = [[0] * ADT_CELLS_PER_GRID for _ in range(ADT_CELLS_PER_GRID)]
    liquid_entry = [[0] * ADT_CELLS_PER_GRID for _ in range(ADT_CELLS_PER_GRID)]
    liquid_height = [[0.0] * (ADT_GRID_SIZE + 1) for _ in range(ADT_GRID_SIZE + 1)]
    holes = [[0] * ADT_CELLS_PER_GRID for _ in range(ADT_CELLS_PER_GRID)]
    
    # Get area IDs from MCNK chunks
    for i in range(ADT_CELLS_PER_GRID):
        for j in range(ADT_CELLS_PER_GRID):
            mcnk_offset = adt.get_mcnk(i, j)
            if mcnk_offset:
                # areaid is at offset 8 + 52 = 60 in MCNK
                area_ids[i][j] = struct.unpack_from('<I', data, mcnk_offset + 60)[0] & 0xFFFF
    
    # Pack area data
    full_area_data = False
    first_area = area_ids[0][0]
    for i in range(ADT_CELLS_PER_GRID):
        for j in range(ADT_CELLS_PER_GRID):
            if area_ids[i][j] != first_area:
                full_area_data = True
                break
        if full_area_data:
            break
    
    # Get height map from MCNK chunks
    for i in range(ADT_CELLS_PER_GRID):
        for j in range(ADT_CELLS_PER_GRID):
            mcnk_offset = adt.get_mcnk(i, j)
            if not mcnk_offset:
                continue
            
            # zpos (vertical height) is at offset 8 + 112 = 120 in MCNK
            ypos = struct.unpack_from('<f', data, mcnk_offset + 120)[0]
            
            # Initialize height with cell base height
            for y in range(ADT_CELL_SIZE + 1):
                cy = i * ADT_CELL_SIZE + y
                for x in range(ADT_CELL_SIZE + 1):
                    cx = j * ADT_CELL_SIZE + x
                    v9[cy][cx] = ypos
            
            for y in range(ADT_CELL_SIZE):
                cy = i * ADT_CELL_SIZE + y
                for x in range(ADT_CELL_SIZE):
                    cx = j * ADT_CELL_SIZE + x
                    v8[cy][cx] = ypos
            
            # Get custom height from MCVT
            mcv_t_offset = adt.get_mcv_t(mcnk_offset)
            if mcv_t_offset:
                # MCVT has (ADT_CELL_SIZE+1)^2 + ADT_CELL_SIZE^2 = 81 + 64 = 145 floats
                # V9: y * (ADT_CELL_SIZE*2+1) + x
                # V8: y * (ADT_CELL_SIZE*2+1) + ADT_CELL_SIZE+1 + x
                for y in range(ADT_CELL_SIZE + 1):
                    cy = i * ADT_CELL_SIZE + y
                    for x in range(ADT_CELL_SIZE + 1):
                        cx = j * ADT_CELL_SIZE + x
                        idx = y * (ADT_CELL_SIZE * 2 + 1) + x
                        v9[cy][cx] += struct.unpack_from('<f', data, mcv_t_offset + idx * 4)[0]
                
                for y in range(ADT_CELL_SIZE):
                    cy = i * ADT_CELL_SIZE + y
                    for x in range(ADT_CELL_SIZE):
                        cx = j * ADT_CELL_SIZE + x
                        idx = y * (ADT_CELL_SIZE * 2 + 1) + ADT_CELL_SIZE + 1 + x
                        v8[cy][cx] += struct.unpack_from('<f', data, mcv_t_offset + idx * 4)[0]
    
    # Get height bounds
    min_height = min(min(row) for row in v8)
    max_height = max(max(row) for row in v8)
    min_height = min(min_height, min(min(row) for row in v9))
    max_height = max(max_height, max(max(row) for row in v9))
    
    # Apply height limits
    if CONF_ALLOW_HEIGHT_LIMIT and min_height < CONF_USE_MIN_HEIGHT:
        for i in range(ADT_GRID_SIZE):
            for j in range(ADT_GRID_SIZE):
                if v8[i][j] < CONF_USE_MIN_HEIGHT:
                    v8[i][j] = CONF_USE_MIN_HEIGHT
        for i in range(ADT_GRID_SIZE + 1):
            for j in range(ADT_GRID_SIZE + 1):
                if v9[i][j] < CONF_USE_MIN_HEIGHT:
                    v9[i][j] = CONF_USE_MIN_HEIGHT
        min_height = max(min_height, CONF_USE_MIN_HEIGHT)
        max_height = max(max_height, CONF_USE_MIN_HEIGHT)
    
    # Get flight bounds
    flight_box_max = None
    flight_box_min = None
    if adt.mfbo:
        # MFBO: magic(4) + size(4) + max(18 bytes) + min(18 bytes)
        mfbo_data = adt.mfbo + 8
        flight_box_max = struct.unpack_from('<9h', data, mfbo_data)
        flight_box_min = struct.unpack_from('<9h', data, mfbo_data + 18)
    
    # Get liquid data from MCLQ (old)
    for i in range(ADT_CELLS_PER_GRID):
        for j in range(ADT_CELLS_PER_GRID):
            mcnk_offset = adt.get_mcnk(i, j)
            if not mcnk_offset:
                continue
            
            mclq_offset, size_mclq = adt.get_mclq(mcnk_offset)
            if not mclq_offset:
                continue
            
            # MCLQ structure: magic(4) + size(4) + height1(4) + height2(4) + liquid[9][9] + flags[8][8] + data[84]
            # liquid[y][x]: light(4) + height(4)
            # flags[y][x]: 1 byte
            count = 0
            for y in range(ADT_CELL_SIZE):
                cy = i * ADT_CELL_SIZE + y
                for x in range(ADT_CELL_SIZE):
                    cx = j * ADT_CELL_SIZE + x
                    flag_offset = mclq_offset + 8 + 2 * 4 + 81 * 8 + y * ADT_CELL_SIZE + x
                    flag = struct.unpack_from('<B', data, flag_offset)[0]
                    if flag != 0x0F:
                        liquid_show[cy][cx] = True
                        if flag & (1 << 7):
                            liquid_flags[i][j] |= MAP_LIQUID_TYPE_DARK_WATER
                        count += 1
            
            # Check cell flags for liquid type
            cell_flags = struct.unpack_from('<I', data, mcnk_offset + 8)[0]
            if cell_flags & (1 << 2):
                liquid_entry[i][j] = 1
                liquid_flags[i][j] |= MAP_LIQUID_TYPE_WATER
            if cell_flags & (1 << 3):
                liquid_entry[i][j] = 2
                liquid_flags[i][j] |= MAP_LIQUID_TYPE_OCEAN
            if cell_flags & (1 << 4):
                liquid_entry[i][j] = 3
                liquid_flags[i][j] |= MAP_LIQUID_TYPE_MAGMA
            
            # Get liquid heights
            for y in range(ADT_CELL_SIZE + 1):
                cy = i * ADT_CELL_SIZE + y
                for x in range(ADT_CELL_SIZE + 1):
                    cx = j * ADT_CELL_SIZE + x
                    height_offset = mclq_offset + 8 + 2 * 4 + (y * (ADT_CELL_SIZE + 1) + x) * 8 + 4
                    liquid_height[cy][cx] = struct.unpack_from('<f', data, height_offset)[0]
    
    # Get liquid data from MH2O (new, WotLK)
    if adt.mh2o and liquid_types:
        mh2o_offset = adt.mh2o
        # MH2O structure: magic(4) + size(4) + liquid[16][16]
        # Each liquid entry: offset_instances(4) + used(4) + offset_attributes(4)
        for i in range(ADT_CELLS_PER_GRID):
            for j in range(ADT_CELLS_PER_GRID):
                entry_offset = mh2o_offset + 8 + (i * ADT_CELLS_PER_GRID + j) * 12
                offset_instances = struct.unpack_from('<I', data, entry_offset)[0]
                used = struct.unpack_from('<I', data, entry_offset + 4)[0]
                offset_attributes = struct.unpack_from('<I', data, entry_offset + 8)[0]
                
                if used == 0 or offset_instances == 0:
                    continue
                
                # Liquid instance: liquid_type(2) + vertex_format(2) + min_height(4) + max_height(4) + offset_x(1) + offset_y(1) + width(1) + height(1) + offset_exists_bitmap(4) + offset_vertex_data(4)
                inst_offset = mh2o_offset + 8 + offset_instances
                liquid_type = struct.unpack_from('<H', data, inst_offset)[0]
                vertex_format = struct.unpack_from('<H', data, inst_offset + 2)[0]
                offset_x = struct.unpack_from('<B', data, inst_offset + 12)[0]
                offset_y = struct.unpack_from('<B', data, inst_offset + 13)[0]
                width = struct.unpack_from('<B', data, inst_offset + 14)[0]
                height = struct.unpack_from('<B', data, inst_offset + 15)[0]
                offset_exists_bitmap = struct.unpack_from('<I', data, inst_offset + 16)[0]
                offset_vertex_data = struct.unpack_from('<I', data, inst_offset + 20)[0]
                
                # Get sound bank for this liquid type
                sound_bank = liquid_types.get(liquid_type, 0)
                
                # Get exists bitmap
                if offset_exists_bitmap:
                    exists_bitmap = struct.unpack_from('<Q', data, mh2o_offset + 8 + offset_exists_bitmap)[0]
                else:
                    exists_bitmap = 0xFFFFFFFFFFFFFFFF
                
                # Process liquid
                count = 0
                exists_mask = exists_bitmap
                for y in range(height):
                    cy = i * ADT_CELL_SIZE + y + offset_y
                    for x in range(width):
                        cx = j * ADT_CELL_SIZE + x + offset_x
                        if (exists_mask & 1) and 0 <= cy < ADT_GRID_SIZE and 0 <= cx < ADT_GRID_SIZE:
                            liquid_show[cy][cx] = True
                            count += 1
                        exists_mask >>= 1
                
                liquid_entry[i][j] = liquid_type
                
                # Set liquid flags based on sound bank
                if sound_bank == LIQUID_TYPE_WATER:
                    liquid_flags[i][j] |= MAP_LIQUID_TYPE_WATER
                elif sound_bank == LIQUID_TYPE_OCEAN:
                    liquid_flags[i][j] |= MAP_LIQUID_TYPE_OCEAN
                    # Check deep attribute
                    if offset_attributes and (mh2o_offset + 8 + offset_attributes + 16) <= len(data):
                        attrs_offset = mh2o_offset + 8 + offset_attributes
                        deep = struct.unpack_from('<Q', data, attrs_offset + 8)[0]
                        if deep != 0xFFFFFFFFFFFFFFFF:
                            liquid_flags[i][j] |= MAP_LIQUID_TYPE_DARK_WATER
                elif sound_bank == LIQUID_TYPE_MAGMA:
                    liquid_flags[i][j] |= MAP_LIQUID_TYPE_MAGMA
                elif sound_bank == LIQUID_TYPE_SLIME:
                    liquid_flags[i][j] |= MAP_LIQUID_TYPE_SLIME
                
                # Get liquid heights
                pos = 0
                for y in range(height + 1):
                    cy = i * ADT_CELL_SIZE + y + offset_y
                    for x in range(width + 1):
                        cx = j * ADT_CELL_SIZE + x + offset_x
                        if offset_vertex_data and (mh2o_offset + 8 + offset_vertex_data + (pos + 1) * 4) <= len(data):
                            if 0 <= cy <= ADT_GRID_SIZE and 0 <= cx <= ADT_GRID_SIZE:
                                height_val = struct.unpack_from('<f', data, mh2o_offset + 8 + offset_vertex_data + pos * 4)[0]
                                liquid_height[cy][cx] = height_val
                        pos += 1
    
    # Pack liquid data
    first_liquid_type = liquid_entry[0][0]
    first_liquid_flag = liquid_flags[0][0]
    full_type = False
    for y in range(ADT_CELLS_PER_GRID):
        for x in range(ADT_CELLS_PER_GRID):
            if liquid_entry[y][x] != first_liquid_type or liquid_flags[y][x] != first_liquid_flag:
                full_type = True
                break
        if full_type:
            break
    
    # Calculate liquid bounds
    min_x = min_y = 255
    max_x = max_y = 0
    min_liquid_height = 20000.0
    max_liquid_height = -20000.0
    has_liquid = False
    
    for y in range(ADT_GRID_SIZE):
        for x in range(ADT_GRID_SIZE):
            if liquid_show[y][x]:
                has_liquid = True
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                h = liquid_height[y][x]
                min_liquid_height = min(min_liquid_height, h)
                max_liquid_height = max(max_liquid_height, h)
    
    # Fill non-liquid areas with min height
    if has_liquid:
        for y in range(ADT_GRID_SIZE):
            for x in range(ADT_GRID_SIZE):
                if not liquid_show[y][x]:
                    liquid_height[y][x] = CONF_USE_MIN_HEIGHT
                    if CONF_USE_MIN_HEIGHT > min_liquid_height:
                        min_liquid_height = CONF_USE_MIN_HEIGHT
    
    # Get holes
    has_holes = False
    for i in range(ADT_CELLS_PER_GRID):
        for j in range(ADT_CELLS_PER_GRID):
            mcnk_offset = adt.get_mcnk(i, j)
            if mcnk_offset:
                holes_val = struct.unpack_from('<H', data, mcnk_offset + 68)[0]
                holes[i][j] = holes_val
                if holes_val != 0:
                    has_holes = True
    
    # Build output .map file
    # Prepare map header
    map_data = bytearray()
    
    # Calculate offsets
    header_size = 44  # map_fileheader size
    area_header_offset = header_size
    
    # Area header
    area_header = bytearray()
    area_header += MAP_AREA_MAGIC
    if full_area_data:
        area_header += struct.pack('<HH', 0, 0)  # flags=0, gridArea=0
    else:
        area_header += struct.pack('<HH', MAP_AREA_NO_AREA, first_area)
    
    area_data_offset = area_header_offset + len(area_header)
    
    # Height header
    height_flags = 0
    if max_height == min_height:
        height_flags |= MAP_HEIGHT_NO_HEIGHT
    if CONF_ALLOW_FLOAT_TO_INT and (max_height - min_height) < CONF_FLAT_HEIGHT_DELTA_LIMIT:
        height_flags |= MAP_HEIGHT_NO_HEIGHT
    
    if flight_box_max:
        height_flags |= MAP_HEIGHT_HAS_FLIGHT_BOUNDS
    
    height_data = None
    if not (height_flags & MAP_HEIGHT_NO_HEIGHT):
        diff = max_height - min_height
        step = 0
        if CONF_ALLOW_FLOAT_TO_INT:
            if diff < CONF_FLOAT_TO_INT8_LIMIT:
                height_flags |= MAP_HEIGHT_AS_INT8
                step = select_uint8_step(diff)
                uint8_v8 = [[int((v8[i][j] - min_height) * step + 0.5) for j in range(ADT_GRID_SIZE)] for i in range(ADT_GRID_SIZE)]
                uint8_v9 = [[int((v9[i][j] - min_height) * step + 0.5) for j in range(ADT_GRID_SIZE + 1)] for i in range(ADT_GRID_SIZE + 1)]
                height_data = ('int8', uint8_v8, uint8_v9)
            elif diff < CONF_FLOAT_TO_INT16_LIMIT:
                height_flags |= MAP_HEIGHT_AS_INT16
                step = select_uint16_step(diff)
                uint16_v8 = [[int((v8[i][j] - min_height) * step + 0.5) for j in range(ADT_GRID_SIZE)] for i in range(ADT_GRID_SIZE)]
                uint16_v9 = [[int((v9[i][j] - min_height) * step + 0.5) for j in range(ADT_GRID_SIZE + 1)] for i in range(ADT_GRID_SIZE + 1)]
                height_data = ('int16', uint16_v8, uint16_v9)
            else:
                height_data = ('float', v8, v9)
        else:
            height_data = ('float', v8, v9)
    
    # Build liquid header
    liquid_header = bytearray()
    liquid_data = bytearray()
    
    if not has_liquid or (first_liquid_flag == 0 and not full_type):
        # No liquid
        liquid_map_offset = 0
        liquid_map_size = 0
    else:
        liquid_flags_val = 0
        if max_liquid_height == min_liquid_height:
            liquid_flags_val |= MAP_LIQUID_NO_HEIGHT
        if CONF_ALLOW_FLOAT_TO_INT and (max_liquid_height - min_liquid_height) < CONF_FLAT_LIQUID_DELTA_LIMIT:
            liquid_flags_val |= MAP_LIQUID_NO_HEIGHT
        
        if not full_type:
            liquid_flags_val |= MAP_LIQUID_NO_TYPE
        
        liquid_header += MAP_LIQUID_MAGIC
        liquid_header += struct.pack('<B', liquid_flags_val)
        
        if liquid_flags_val & MAP_LIQUID_NO_TYPE:
            liquid_header += struct.pack('<BH', first_liquid_flag, first_liquid_type)
        else:
            liquid_header += struct.pack('<BHH', 0, 0, 0)
            # Add liquid_entry and liquid_flags arrays
            for i in range(ADT_CELLS_PER_GRID):
                for j in range(ADT_CELLS_PER_GRID):
                    liquid_data += struct.pack('<H', liquid_entry[i][j])
            for i in range(ADT_CELLS_PER_GRID):
                for j in range(ADT_CELLS_PER_GRID):
                    liquid_data += struct.pack('<B', liquid_flags[i][j])
        
        liq_offset_x = min_x
        liq_offset_y = min_y
        liq_width = max_x - min_x + 2
        liq_height = max_y - min_y + 2
        liquid_header += struct.pack('<BBBB', liq_offset_x, liq_offset_y, liq_width, liq_height)
        liquid_header += struct.pack('<f', min_liquid_height)
        
        # Add liquid heights if needed
        if not (liquid_flags_val & MAP_LIQUID_NO_HEIGHT):
            for y in range(liq_height):  # height
                for x in range(liq_width):  # width
                    cy = y + liq_offset_y  # offsetY
                    cx = x + liq_offset_x  # offsetX
                    liquid_data += struct.pack('<f', liquid_height[cy][cx])
    
    # Calculate final offsets
    height_header_offset = area_data_offset
    if full_area_data:
        height_header_offset += ADT_CELLS_PER_GRID * ADT_CELLS_PER_GRID * 2  # area_ids
    
    # Write everything
    with open(map_path, 'wb') as f:
        # Map header
        map_header = bytearray()
        map_header += struct.pack('<III', struct.unpack('<I', MAP_MAGIC)[0], MAP_VERSION_MAGIC, build)
        
        area_offset = header_size
        area_size = len(area_header)
        if full_area_data:
            area_size += ADT_CELLS_PER_GRID * ADT_CELLS_PER_GRID * 2
        
        height_offset = area_offset + area_size
        height_size = 20  # height header size
        if height_data:
            if height_data[0] == 'int8':
                height_size += ADT_GRID_SIZE * ADT_GRID_SIZE + (ADT_GRID_SIZE + 1) * (ADT_GRID_SIZE + 1)
            elif height_data[0] == 'int16':
                height_size += (ADT_GRID_SIZE * ADT_GRID_SIZE + (ADT_GRID_SIZE + 1) * (ADT_GRID_SIZE + 1)) * 2
            else:
                height_size += (ADT_GRID_SIZE * ADT_GRID_SIZE + (ADT_GRID_SIZE + 1) * (ADT_GRID_SIZE + 1)) * 4
        
        if flight_box_max:
            height_size += 18 * 2  # 9 shorts * 2 planes
        
        liquid_offset = height_offset + height_size
        liquid_size = len(liquid_header) + len(liquid_data) if liquid_header else 0
        
        holes_offset = 0
        holes_size = 0
        if has_holes:
            holes_offset = liquid_offset + liquid_size if liquid_size > 0 else height_offset + height_size
            holes_size = ADT_CELLS_PER_GRID * ADT_CELLS_PER_GRID * 2
        
        # Write header
        map_header += struct.pack('<IIIII', area_offset, area_size, height_offset, height_size, liquid_offset if liquid_size > 0 else 0)
        map_header += struct.pack('<III', liquid_size, holes_offset, holes_size)
        f.write(map_header)
        
        # Write area
        f.write(area_header)
        if full_area_data:
            for i in range(ADT_CELLS_PER_GRID):
                for j in range(ADT_CELLS_PER_GRID):
                    f.write(struct.pack('<H', area_ids[i][j]))
        
        # Write height
        f.write(struct.pack('<IIff', struct.unpack('<I', MAP_HEIGHT_MAGIC)[0], height_flags, min_height, max_height))
        if height_data:
            if height_data[0] == 'int8':
                for i in range(ADT_GRID_SIZE):
                    f.write(bytes(height_data[1][i]))
                for i in range(ADT_GRID_SIZE + 1):
                    f.write(bytes(height_data[2][i]))
            elif height_data[0] == 'int16':
                for i in range(ADT_GRID_SIZE):
                    for j in range(ADT_GRID_SIZE):
                        f.write(struct.pack('<H', height_data[1][i][j]))
                for i in range(ADT_GRID_SIZE + 1):
                    for j in range(ADT_GRID_SIZE + 1):
                        f.write(struct.pack('<H', height_data[2][i][j]))
            else:
                for i in range(ADT_GRID_SIZE):
                    for j in range(ADT_GRID_SIZE):
                        f.write(struct.pack('<f', height_data[1][i][j]))
                for i in range(ADT_GRID_SIZE + 1):
                    for j in range(ADT_GRID_SIZE + 1):
                        f.write(struct.pack('<f', height_data[2][i][j]))
        
        if flight_box_max:
            f.write(struct.pack('<9h', *flight_box_max))
            f.write(struct.pack('<9h', *flight_box_min))
        
        # Write liquid
        if liquid_size > 0:
            f.write(liquid_header)
            f.write(liquid_data)
        
        # Write holes
        if has_holes:
            for i in range(ADT_CELLS_PER_GRID):
                for j in range(ADT_CELLS_PER_GRID):
                    f.write(struct.pack('<H', holes[i][j]))

def main():
    if len(sys.argv) < 3:
        print("Usage: adt_to_map.py <input.adt> <output.map> [build_number]")
        print("       adt_to_map.py --batch <input_dir> <output_dir> [build_number]")
        sys.exit(1)
    
    dbc_path = Path(sys.argv[0]).parent / "LiquidType.dbc"
    if not dbc_path.exists():
        # Try client dbc folder
        client_dbc = Path(r"D:\Games\wow\World of Warcraft 3.3.5a (ruRu enGB enUs) (12340)\dbc\LiquidType.dbc")
        if client_dbc.exists():
            dbc_path = client_dbc
        else:
            print("Warning: LiquidType.dbc not found, liquid types will be skipped")
            dbc_path = None
    
    if dbc_path:
        load_liquid_type_dbc(dbc_path)
    
    if sys.argv[1] == '--batch':
        input_dir = Path(sys.argv[2])
        output_dir = Path(sys.argv[3])
        build = int(sys.argv[4]) if len(sys.argv) > 4 else 12340
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for adt_file in input_dir.glob('*.adt'):
            map_file = output_dir / adt_file.name.replace('.adt', '.map')
            try:
                convert_adt_to_map(adt_file, map_file, build)
                print(f"Converted: {adt_file.name} -> {map_file.name}")
            except Exception as e:
                print(f"Error converting {adt_file.name}: {e}")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        build = int(sys.argv[3]) if len(sys.argv) > 3 else 12340
        
        convert_adt_to_map(input_file, output_file, build)
        print(f"Converted: {input_file} -> {output_file}")

if __name__ == '__main__':
    main()
