"""Symbol → sector mapping for cross-sectional sector-relative features.

Re-homed from the deleted ``leading_v3`` builder so sector-relative features
(``return_vs_sector`` …) have a real grouping source. Symbols not listed map to
``"Other"`` so an unfamiliar universe never breaks a run; cross-sectional ops
still leave warmup NaN (leakage-safe) rather than back-filling.

This is intentionally a small static map for the VN universe. A future data
pipeline can replace ``build_sector_map`` with a DB-backed lookup without
touching the feature layer.
"""

from __future__ import annotations

from collections.abc import Iterable

DEFAULT_SECTOR = "Other"

SECTOR_MAP: dict[str, str] = {
    # Finance / Ngân hàng
    "ACB": "Finance",
    "BID": "Finance",
    "CTG": "Finance",
    "MBB": "Finance",
    "TCB": "Finance",
    "VIB": "Finance",
    "VCB": "Finance",
    "SHB": "Finance",
    "STB": "Finance",
    "HDB": "Finance",
    "TPB": "Finance",
    "EIB": "Finance",
    "OCB": "Finance",
    "SGB": "Finance",
    "VPB": "Finance",
    "MSB": "Finance",
    "NVB": "Finance",
    "NAB": "Finance",
    "KLB": "Finance",
    "KBC": "Finance",
    "DBC": "Finance",
    "SEA": "Finance",
    "BAF": "Finance",
    "BFC": "Finance",
    "BSI": "Finance",
    # Consumer / Tiêu dùng
    "VNM": "Consumer",
    "SAB": "Consumer",
    "BHN": "Consumer",
    "MSN": "Consumer",
    "MWG": "Consumer",
    "CII": "Consumer",
    "DGW": "Consumer",
    "PNJ": "Consumer",
    "LGC": "Consumer",
    # Energy / Năng lượng
    "GAS": "Energy",
    "PVD": "Energy",
    "PVH": "Energy",
    "BSR": "Energy",
    "POW": "Energy",
    "NT2": "Energy",
    "QTP": "Energy",
    "PVB": "Energy",
    # Materials / Vật liệu
    "HPG": "Materials",
    "NKG": "Materials",
    "DGC": "Materials",
    "HSG": "Materials",
    "TAC": "Materials",
    "THD": "Materials",
    "CVN": "Materials",
    "DRC": "Materials",
    "PAS": "Materials",
    "REE": "Materials",
    "RAL": "Materials",
    "NLG": "Materials",
    "ROS": "Materials",
    # Industrials / Công nghiệp
    "VJC": "Industrials",
    "ACV": "Industrials",
    "HAH": "Industrials",
    "HND": "Industrials",
    "VSC": "Industrials",
    "GMD": "Industrials",
    "ITA": "Industrials",
    # Technology / Công nghệ
    "FPT": "Technology",
    "CMG": "Technology",
    "ICT": "Technology",
    "VGI": "Technology",
    "HUT": "Technology",
    "ITC": "Technology",
    "BBS": "Technology",
    # Real Estate / Bất động sản
    "VHM": "RealEstate",
    "NVL": "RealEstate",
    "DXG": "RealEstate",
    "VRE": "RealEstate",
    "HCM": "RealEstate",
    "KDH": "RealEstate",
    "PDR": "RealEstate",
    "LDG": "RealEstate",
    "SCR": "RealEstate",
    "DIG": "RealEstate",
}


def get_sector(symbol: str) -> str:
    """Sector for one symbol; ``DEFAULT_SECTOR`` if unmapped."""
    return SECTOR_MAP.get(symbol, DEFAULT_SECTOR)


def build_sector_map(symbols: Iterable[str]) -> dict[str, str]:
    """Return a complete {symbol: sector} map covering every given symbol.

    Guarantees no symbol is missing (unmapped → ``DEFAULT_SECTOR``) so the
    resolver's fail-loud "sector_map missing symbols" check never trips on a
    known universe.
    """
    return {s: get_sector(s) for s in symbols}
