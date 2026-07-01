"""Sentinel-2 band metadata for multi-band super-resolution pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# GeoSR MISR ``lr_stack`` channel order (12 bands, all resampled to the LR grid).
GEOSR_LR_STACK_BANDS: tuple[str, ...] = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
)

GEOSR_BAND_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(GEOSR_LR_STACK_BANDS)}

# Earth Search L2A asset keys (stackstac / Element84 catalog).
EARTH_SEARCH_ASSET_BY_BAND: dict[str, str] = {
    "B01": "coastal",
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B05": "rededge1",
    "B06": "rededge2",
    "B07": "rededge3",
    "B08": "nir",
    "B8A": "nir08",
    "B09": "nir09",
    "B11": "swir16",
    "B12": "swir22",
}

NATIVE_GSD_M_BY_BAND: dict[str, float] = {
    "B01": 60.0,
    "B02": 10.0,
    "B03": 10.0,
    "B04": 10.0,
    "B05": 20.0,
    "B06": 20.0,
    "B07": 20.0,
    "B08": 10.0,
    "B8A": 20.0,
    "B09": 60.0,
    "B11": 20.0,
    "B12": 20.0,
}


@dataclass(frozen=True)
class BandPreset:
    name: str
    band_names: tuple[str, ...]
    description: str

    @property
    def num_channels(self) -> int:
        return len(self.band_names)

    def geosr_stack_indices(self) -> tuple[int, ...]:
        return tuple(GEOSR_BAND_TO_INDEX[b] for b in self.band_names)

    def earth_search_assets(self) -> tuple[str, ...]:
        return tuple(EARTH_SEARCH_ASSET_BY_BAND[b] for b in self.band_names)


BAND_PRESETS: dict[str, BandPreset] = {
    "rgb": BandPreset(
        name="rgb",
        band_names=("B04", "B03", "B02"),
        description="Visible RGB (10 m)",
    ),
    "rgb_nir": BandPreset(
        name="rgb_nir",
        band_names=("B04", "B03", "B02", "B08"),
        description="RGB + NIR (10 m)",
    ),
    "10m": BandPreset(
        name="10m",
        band_names=("B02", "B03", "B04", "B08"),
        description="All native 10 m bands",
    ),
    "10m_20m": BandPreset(
        name="10m_20m",
        band_names=(
            "B02",
            "B03",
            "B04",
            "B08",
            "B05",
            "B06",
            "B07",
            "B8A",
            "B11",
            "B12",
        ),
        description="10 m + 20 m bands (20 m resampled to 10 m grid at export)",
    ),
    "all_12": BandPreset(
        name="all_12",
        band_names=GEOSR_LR_STACK_BANDS,
        description="Full 12-band GeoSR stack (mixed native GSD, common LR grid)",
    ),
}


def resolve_band_preset(name: str) -> BandPreset:
    key = str(name).strip().lower()
    if key not in BAND_PRESETS:
        choices = ", ".join(sorted(BAND_PRESETS))
        raise ValueError(f"Unknown band preset {name!r}. Choose from: {choices}")
    return BAND_PRESETS[key]


def parse_band_names_csv(text: str) -> tuple[str, ...]:
    """Parse comma-separated band names, e.g. ``B04,B03,B02,B08``."""
    names = tuple(part.strip().upper() for part in text.split(",") if part.strip())
    if not names:
        raise ValueError("band list must not be empty")
    for name in names:
        if name not in GEOSR_BAND_TO_INDEX and name not in EARTH_SEARCH_ASSET_BY_BAND:
            raise ValueError(f"Unknown Sentinel-2 band name: {name}")
    return names


def band_manifest_dict(band_names: Sequence[str]) -> dict:
    return {
        "band_names": list(band_names),
        "num_channels": len(band_names),
        "geosr_stack_indices": [GEOSR_BAND_TO_INDEX[b] for b in band_names],
        "native_gsd_m": [NATIVE_GSD_M_BY_BAND[b] for b in band_names],
        "earth_search_assets": [EARTH_SEARCH_ASSET_BY_BAND[b] for b in band_names],
    }
