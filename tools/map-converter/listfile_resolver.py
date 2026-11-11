#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast Retail FileDataID listfile loader with relocated/fuzzy asset resolution."""

from __future__ import annotations

import difflib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def normalize_wow_path(path: str, is_model: bool = False) -> str:
    """Normalize a WoW asset path to lowercase forward slashes (.mdx/.mdl -> .m2)."""
    p = path.strip().replace("\\", "/").lower()
    while "//" in p:
        p = p.replace("//", "/")
    if p.startswith("./"):
        p = p[2:]
    if is_model and (p.endswith(".mdx") or p.endswith(".mdl")):
        p = p[:-4] + ".m2"
    return p


@dataclass
class ResolutionResult:
    fdid: int
    source_path: str
    resolved_path: Optional[str]
    method: str  # "exact", "alias", "basename", "fuzzy", "missing"


@dataclass
class ResolutionStats:
    exact: int = 0
    alias: int = 0
    basename: int = 0
    fuzzy: int = 0
    missing: int = 0
    height_textures_found: int = 0
    fuzzy_mappings: Dict[str, str] = field(default_factory=dict)
    missing_assets: Dict[str, List[str]] = field(default_factory=dict)


class ListfileResolver:
    """Loads Retail CSV listfiles and resolves WotLK/custom paths to FileDataIDs."""

    def __init__(
        self,
        listfile_paths: List[Path],
        alias_map: Optional[Dict[str, object]] = None,
        enable_fuzzy: bool = True,
        fuzzy_cutoff: float = 0.84,
    ) -> None:
        self.exact: Dict[str, int] = {}
        self.fdid_to_path: Dict[int, str] = {}
        self.by_basename: Dict[str, List[Tuple[int, str]]] = {}
        self.by_dir_ext: Dict[Tuple[str, str], List[Tuple[int, str, str]]] = {}
        self.alias_map: Dict[str, object] = {}
        if alias_map:
            for k, v in alias_map.items():
                self.alias_map[normalize_wow_path(k)] = v
        self.enable_fuzzy = enable_fuzzy
        self.fuzzy_cutoff = fuzzy_cutoff
        self._cache: Dict[Tuple[str, bool], ResolutionResult] = {}

        for lf_path in listfile_paths:
            self._load_csv(lf_path)

    def _load_csv(self, csv_path: Path) -> None:
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s or ";" not in s or s.startswith("#"):
                    continue
                fid_str, rel = s.split(";", 1)
                try:
                    fid = int(fid_str)
                except ValueError:
                    continue
                norm = normalize_wow_path(rel)
                if not norm:
                    continue
                self.exact[norm] = fid
                self.fdid_to_path[fid] = norm
                if "/" in norm:
                    parent, base = norm.rsplit("/", 1)
                else:
                    parent, base = "", norm
                self.by_basename.setdefault(base, []).append((fid, norm))
                ext = os.path.splitext(base)[1]
                self.by_dir_ext.setdefault((parent, ext), []).append((fid, base, norm))

    def _score_path_similarity(self, a: str, b: str) -> Tuple[int, float]:
        parts_a = a.split("/")
        parts_b = b.split("/")
        shared_tail = 0
        for pa, pb in zip(reversed(parts_a[:-1]), reversed(parts_b[:-1])):
            if pa == pb:
                shared_tail += 1
            else:
                break
        overlap = len(set(parts_a[:-1]) & set(parts_b[:-1]))
        return (shared_tail * 10 + overlap, difflib.SequenceMatcher(None, a, b).ratio())

    def resolve(
        self, raw_path: str, is_model: bool = False, kind: str = "asset"
    ) -> ResolutionResult:
        norm = normalize_wow_path(raw_path, is_model=is_model)
        cache_key = (norm, is_model)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 1. Explicit alias map
        if norm in self.alias_map:
            target = self.alias_map[norm]
            if isinstance(target, int):
                res = ResolutionResult(target, norm, self.fdid_to_path.get(target), "alias")
                self._cache[cache_key] = res
                return res
            t_norm = normalize_wow_path(str(target), is_model=is_model)
            if t_norm in self.exact:
                res = ResolutionResult(self.exact[t_norm], norm, t_norm, "alias")
                self._cache[cache_key] = res
                return res

        # 2. Exact match
        if norm in self.exact:
            res = ResolutionResult(self.exact[norm], norm, norm, "exact")
            self._cache[cache_key] = res
            return res

        if not self.enable_fuzzy:
            res = ResolutionResult(0, norm, None, "missing")
            self._cache[cache_key] = res
            return res

        # 3. Exact filename relocated across expansion directories
        base = norm.rsplit("/", 1)[-1]
        parent = norm.rsplit("/", 1)[0] if "/" in norm else ""
        candidates = self.by_basename.get(base)
        if candidates:
            best_fid, best_path = max(
                candidates, key=lambda item: self._score_path_similarity(norm, item[1])
            )
            res = ResolutionResult(best_fid, norm, best_path, "basename")
            self._cache[cache_key] = res
            return res

        # 4. Same-directory close typo match (e.g. 8riv_farmbcorn_01.m2 -> 8riv_farmcorn_01.m2)
        ext = os.path.splitext(base)[1]
        dir_entries = self.by_dir_ext.get((parent, ext), [])
        if dir_entries:
            base_names = [entry[1] for entry in dir_entries]
            close = difflib.get_close_matches(base, base_names, n=1, cutoff=self.fuzzy_cutoff)
            if close:
                match_base = close[0]
                for fid, bname, full_p in dir_entries:
                    if bname == match_base:
                        res = ResolutionResult(fid, norm, full_p, "fuzzy")
                        self._cache[cache_key] = res
                        return res

        res = ResolutionResult(0, norm, None, "missing")
        self._cache[cache_key] = res
        return res

    def resolve_height_texture(self, diffuse_path: str) -> int:
        """Find companion *_h.blp heightmap FileDataID for a terrain diffuse texture."""
        norm = normalize_wow_path(diffuse_path)
        if norm.endswith("_h.blp"):
            return self.exact.get(norm, 0)
        if norm.endswith(".blp"):
            h_path = norm[:-4] + "_h.blp"
            return self.exact.get(h_path, 0)
        return 0


def load_alias_file(path: Optional[Path]) -> Dict[str, object]:
    if path is None or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Alias JSON must be an object: {path}")
    return data
