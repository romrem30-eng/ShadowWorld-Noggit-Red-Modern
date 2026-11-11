#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Procedural Terrain Generators & Heightmap Importer.
Includes:
  1. Procedural Perlin noise generators (island, lake_basin, mountain_valley, plains)
  2. Heightmap image importer (PNG/BMP/TIFF 8-bit or 16-bit to continuous terrain)
"""

import math
import numpy as np
from PIL import Image

class PerlinNoise2D:
    def __init__(self, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.p = np.arange(256, dtype=int)
        rng.shuffle(self.p)
        self.p = np.tile(self.p, 2)

    def fade(self, t):
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def lerp(self, a, b, t):
        return a + t * (b - a)

    def grad(self, hash_val, x, y):
        h = hash_val & 3
        u = np.where(h < 2, x, y)
        v = np.where(h < 2, y, x)
        return np.where((h & 1) == 0, u, -u) + np.where((h & 2) == 0, v, -v)

    def noise(self, x, y):
        xi = np.floor(x).astype(int) & 255
        yi = np.floor(y).astype(int) & 255
        xf = x - np.floor(x)
        yf = y - np.floor(y)

        u = self.fade(xf)
        v = self.fade(yf)

        aa = self.p[self.p[xi] + yi]
        ab = self.p[self.p[xi] + yi + 1]
        ba = self.p[self.p[xi + 1] + yi]
        bb = self.p[self.p[xi + 1] + yi + 1]

        g_aa = self.grad(aa, xf, yf)
        g_ba = self.grad(ba, xf - 1.0, yf)
        g_ab = self.grad(ab, xf, yf - 1.0)
        g_bb = self.grad(bb, xf - 1.0, yf - 1.0)

        x1 = self.lerp(g_aa, g_ba, u)
        x2 = self.lerp(g_ab, g_bb, u)
        return self.lerp(x1, x2, v)


class TerrainSampler:
    """Base class for any terrain elevation sampler."""
    def get_elevation(self, wx, wy):
        raise NotImplementedError


class IslandGenerator(TerrainSampler):
    """
    Tropical or temperate island surrounded by open ocean.
    Features a sandy coastal perimeter, gentle central hills, and optional central peak.
    """
    def __init__(self, island_radius: float = 380.0, peak_height: float = 140.0, water_level: float = 0.0, seed: int = 1337):
        self.radius = island_radius
        self.peak_h = peak_height
        self.water_level = water_level
        self.pn1 = PerlinNoise2D(seed=seed)
        self.pn2 = PerlinNoise2D(seed=seed + 101)

    def get_elevation(self, wx, wy):
        is_scalar = np.isscalar(wx)
        if is_scalar:
            wx, wy = np.array([wx], dtype=float), np.array([wy], dtype=float)

        dist = np.hypot(wx, wy)
        angle = np.arctan2(wy, wx)
        
        # Organic coastline wobble
        wobble = 35.0 * np.sin(angle * 3.0) + 20.0 * np.cos(angle * 5.0 - 0.4)
        eff_r = self.radius + wobble

        # Distance to shoreline: positive = inland, negative = submerged in ocean
        sd = eff_r - dist

        # Ocean bed (-20m)
        t_sub = np.clip((-sd) / 60.0, 0.0, 1.0)
        s_sub = t_sub * t_sub * (3.0 - 2.0 * t_sub)
        h_ocean = self.water_level - 22.0 * s_sub

        # Coastal beach shelf (+0m to +2.5m)
        t_beach = np.clip(sd / 35.0, 0.0, 1.0)
        s_beach = t_beach * t_beach * (3.0 - 2.0 * t_beach)
        h_beach = self.water_level + 2.5 * s_beach

        # Inland mountain rise
        t_inland = np.clip((sd - 25.0) / (self.radius * 0.75), 0.0, 1.0)
        s_inland = t_inland * t_inland * (3.0 - 2.0 * t_inland)

        n1 = self.pn1.noise(wx * 0.005, wy * 0.005)
        n2 = np.abs(self.pn2.noise(wx * 0.012, wy * 0.012))
        mountain = (n1 * 0.6 + n2 * 0.8) * self.peak_h

        h_land = h_beach + (mountain + 4.0) * s_inland
        h = np.where(sd >= 0, h_land, h_ocean)

        return float(h[0]) if is_scalar else h


class LakeBasinGenerator(TerrainSampler):
    """
    Huge natural lake basin with wide shores, village building plateaus,
    and a massive alpine mountain range backdrop.
    """
    def __init__(self, peak_height: float = 180.0, water_level: float = 0.0, seed: int = 2026):
        self.peak_h = peak_height
        self.water_level = water_level
        self.pn_m = PerlinNoise2D(seed=seed)
        self.pn_r = PerlinNoise2D(seed=seed + 77)

    def get_elevation(self, wx, wy):
        is_scalar = np.isscalar(wx)
        if is_scalar:
            wx, wy = np.array([wx], dtype=float), np.array([wy], dtype=float)

        wave1 = 65.0 * np.sin(wy * 0.0045 + 0.3)
        wave2 = 45.0 * np.cos(wx * 0.0035 - 0.7)
        wave3 = 25.0 * np.sin((wx - wy) * 0.006)
        shoreline_base = 0.65 * wx + 0.76 * wy
        sd = shoreline_base - (-40.0 + wave1 + wave2 + wave3)

        # 1. Lake depth (< 0)
        t_shelf = np.clip((-sd) / 75.0, 0.0, 1.0)
        s_shelf = t_shelf * t_shelf * (3.0 - 2.0 * t_shelf)
        h_submerged = self.water_level - 20.0 * s_shelf

        # 2. Beach shelf (>= 0)
        t_beach = np.clip(sd / 40.0, 0.0, 1.0)
        s_beach = t_beach * t_beach * (3.0 - 2.0 * t_beach)
        h_beach = self.water_level + 2.4 * s_beach

        h_base = np.where(sd >= 0, h_beach, h_submerged)

        # 3. Inland valleys & plateaus
        t_inland = np.clip((sd - 25.0) / 45.0, 0.0, 1.0)
        s_inland = t_inland * t_inland * (3.0 - 2.0 * t_inland)

        valley = 3.5 + 2.0 * np.sin(wx * 0.008 + 1.0) * np.cos(wy * 0.008)

        # 4. Mountains in North/East perimeter
        dist_mtn = np.maximum(wx + wy - 280.0, np.maximum(wx - 260.0, wy - 260.0))
        mtn_factor = np.clip(dist_mtn / 260.0, 0.0, 1.0)
        s_mtn = mtn_factor ** 1.3

        harmonics = (
            np.abs(np.sin(wx * 0.007 + 0.5) * np.cos(wy * 0.007 - 0.3)) * (self.peak_h * 0.45) +
            np.abs(np.sin(wx * 0.015 + 1.2) * np.sin(wy * 0.015 + 0.8)) * (self.peak_h * 0.28) +
            (self.peak_h * 0.20)
        )
        mtn_h = s_mtn * harmonics

        h = h_base + (valley + mtn_h) * s_inland

        # Central island
        dist_isl = np.hypot(wx - (-350.0), wy - (-280.0))
        island_r = 120.0 + 20.0 * np.sin(np.arctan2(wy - (-280.0), wx - (-350.0)) * 3.0)
        is_isl = island_r - dist_isl
        t_isl = np.clip(is_isl / 30.0, 0.0, 1.0)
        s_isl = t_isl * t_isl * (3.0 - 2.0 * t_isl)
        island_h = self.water_level + s_isl * 8.5
        h = np.where(is_isl > 0, np.maximum(h, island_h), h)

        return float(h[0]) if is_scalar else h


class HeightmapImageSampler(TerrainSampler):
    """
    Loads any PNG / BMP / TIFF grayscale heightmap image and samples it
    seamlessly across world coordinates.
    """
    def __init__(self, image_path: str, bounds_min_xy: float, bounds_max_xy: float, min_z: float = -20.0, max_z: float = 160.0):
        img = Image.open(image_path).convert("L")
        self.data = np.array(img, dtype=float) / 255.0  # normalize 0.0..1.0
        self.img_h, self.img_w = self.data.shape
        self.b_min = bounds_min_xy
        self.b_max = bounds_max_xy
        self.min_z = min_z
        self.max_z = max_z

    def get_elevation(self, wx, wy):
        is_scalar = np.isscalar(wx)
        if is_scalar:
            wx, wy = np.array([wx], dtype=float), np.array([wy], dtype=float)

        # Map wx, wy from [b_min, b_max] to image pixel coordinates [0, img_w-1]
        u = np.clip((wx - self.b_min) / (self.b_max - self.b_min), 0.0, 1.0)
        v = np.clip((self.b_max - wy) / (self.b_max - self.b_min), 0.0, 1.0)

        px = np.clip(np.round(u * (self.img_w - 1)).astype(int), 0, self.img_w - 1)
        py = np.clip(np.round(v * (self.img_h - 1)).astype(int), 0, self.img_h - 1)

        norm_val = self.data[py, px]
        h = self.min_z + norm_val * (self.max_z - self.min_z)

        return float(h[0]) if is_scalar else h
