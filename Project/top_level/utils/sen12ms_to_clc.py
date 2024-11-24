from enum import Enum
from pathlib import Path
from typing import Union, Any

import jsonpickle
import numpy as np
import rasterio
from tqdm import tqdm

from Project.top_level.utils.sen12ms_dataLoader import LCBands, ClcDataPath, MainDataPath

LCNoData = 255


class IGBPValues(Enum):
    EvergreenNeedleleafForests = 1
    EvergreenBroadleafForests = 2
    DeciduousNeedleleafForests = 3
    DeciduousBroadleafForests = 4
    MixedForests = 5
    ClosedDenseShrublands = 6
    OpenSparseShrublands = 7
    WoodySavannas = 8
    Savannas = 9
    Grasslands = 10
    PermanentWetlands = 11
    Croplands = 12
    UrbanAndBuiltUpLands = 13
    CroplandNaturalVegetationMosaics = 14
    PermanentSnowAndIce = 15
    Barren = 16
    WaterBodies = 17
    NoData = LCNoData


# Any commented out values are already covered by a previous enumeration class, and we will take that as the true
# value of that pixel.
class LCCSLCValues(Enum):
    # EvergreenNeedleafForests = 11
    # EvergreenBroadleafForests = 12
    # DeciduousNeedleleafForests = 13
    # DeciduousBroadleafForests = 14
    MixedBroadleafNeedleafForests = 15
    MixedBroadleafEvergreenDeciduousForests = 16
    OpenForests = 21
    SparseForests = 22
    DenseHerbaceous = 31
    SparseHerbaceous = 32
    # ClosedDenseShrublands = 41
    # OpenSparseShrublands = 43
    ShrublandGrasslandMosaics = 42
    # PermanentSnowAndIce = 2
    # Barren = 1
    # WaterBodies = 3


class LCCSLUValues(Enum):
    DenseForests = 10
    # OpenForests = 20
    NaturalHerbaceous = 30
    Shrublands = 40
    HerbaceousCroplands = 36
    # UrbanAndBuiltUpLands = 9
    ForestCroplandMosaics = 25
    NaturalHerbaceousCroplandsMosaics = 35
    # PermanentSnowAndIce = 2
    # Barren = 1
    # WaterBodies = 3


class LCCSSHValues(Enum):
    # DenseForests = 10
    # OpenForests = 20
    # Shrublands = 40
    # Grasslands = 30
    WoodyWetlands = 27
    HerbaceousWetlands = 50
    Tundra = 51
    # PermanentSnowAndIce = 2
    # Barren = 1
    # WaterBodies = 3


# Level-1 Values
class CLCValues(Enum):
    """
    Corine Land Cover classification values (Level 1)
    """
    """
    Artificial Surfaces (L2):
        - Urban fabric
        - Industrial, commercial and transport units
        - Mine, dump, and construction sites
        - Artificial, non-agricultural vegetated areas
    """
    ArtificialSurfaces = 1
    """
    Agricultural Areas (L2):
        - Arable land
        - Permanent crops
        - Pastures
        - Heterogeneous agricultural areas
    """
    AgriculturalAreas = 2
    """
    Forest and Semi Natural Areas (L2):
        - Forests
        - Scrub and/or herbaceous vegetation associations
        - Open spaces with little or no vegetation
    """
    ForestAndSemiNaturalAreas = 3
    """
    Wetlands (L2):
        - Inland wetlands
        - Maritime wetlands
    """
    Wetlands = 4
    """
    Water Bodies (L2):
        - Inland waters
        - Marine waters
    """
    WaterBodies = 5
    NoData = LCNoData

    @staticmethod
    def get_item_by_value(value: int) -> str:
        item_by_value: dict[int, str] = {
            1: "ArtificialSurfaces",
            2: "AgriculturalAreas",
            3: "ForestAndSemiNaturalAreas",
            4: "Wetlands",
            5: "WaterBodies",
            LCNoData: "NoData",
        }
        try:
            return item_by_value[value]
        except KeyError as e:
            raise KeyError(f"'{value}' not valid. Choose from '{list(item_by_value.keys())}'.") from e

    @staticmethod
    def as_dict() -> dict[str, int]:
        return {
            "ArtificialSurfaces": 1,
            "AgriculturalAreas": 2,
            "ForestAndSemiNaturalAreas": 3,
            "Wetlands": 4,
            "WaterBodies": 5,
            "NoData": LCNoData,
        }


Sen12MSLCValues = Union[IGBPValues, LCCSLCValues, LCCSLUValues, LCCSSHValues]


"""
Simplified IGBP Landcover
    1. Forest           -> CLCValues.ForestAndSemiNaturalAreas
    2. Shrubland        -> CLCValues.ForestAndSemiNaturalAreas
    3. Savanna          -> CLCValues.ForestAndSemiNaturalAreas
    4. Grassland        -> CLCValues.ForestAndSemiNaturalAreas
    5. Wetlands         -> CLCValues.Wetlands
    6. Croplands        -> CLCValues.AgriculturalAreas
    7. Urban/Built-up   -> CLCValues.ArtificialSurfaces
    8. Snow/Ice         -> CLCValues.NoData
    9. Barren           -> CLCValues.ForestAndSemiNaturalArea
    10. Water           -> CLCValues.WaterBodies
    
HIG-AIML:
    0. Forest           -> CLCValues.ForestAndSemiNaturalAreas
    1. Shrubland        -> CLCValues.ForestAndSemiNaturalAreas
    2. Grassland        -> CLCValues.ForestAndSemiNaturalAreas
    3. Wetlands         -> CLCValues.Wetlands
    4. Croplands        -> CLCValues.AgriculturalAreas
    5. Urban/Built-up   -> CLCValues.ArtificialSurfaces
    6. Barren           -> CLCValues.ForestAndSemiNaturalArea
    7. Water            -> CLCValues.WaterBodies
    255. Invalid        -> CLCValues.NoData
"""



CLCValuesToSen12MSValues: dict[CLCValues, list[Sen12MSLCValues]] = {
    CLCValues.ArtificialSurfaces: [IGBPValues.UrbanAndBuiltUpLands],
    CLCValues.AgriculturalAreas: [
        LCCSLUValues.HerbaceousCroplands,
        IGBPValues.Croplands,
        IGBPValues.CroplandNaturalVegetationMosaics,
        LCCSLUValues.ForestCroplandMosaics,
        LCCSLUValues.NaturalHerbaceousCroplandsMosaics,
    ],
    CLCValues.ForestAndSemiNaturalAreas: [
        IGBPValues.EvergreenNeedleleafForests,
        IGBPValues.EvergreenBroadleafForests,
        IGBPValues.DeciduousNeedleleafForests,
        IGBPValues.DeciduousBroadleafForests,
        LCCSLCValues.MixedBroadleafNeedleafForests,
        LCCSLCValues.MixedBroadleafEvergreenDeciduousForests,
        IGBPValues.MixedForests,
        LCCSLUValues.DenseForests,
        LCCSLCValues.OpenForests,
        LCCSLCValues.SparseForests,
        LCCSLUValues.NaturalHerbaceous,
        LCCSLCValues.DenseHerbaceous,
        LCCSLCValues.SparseHerbaceous,
        LCCSLUValues.Shrublands,
        IGBPValues.ClosedDenseShrublands,
        IGBPValues.OpenSparseShrublands,
        LCCSLCValues.ShrublandGrasslandMosaics,
        IGBPValues.WoodySavannas,
        IGBPValues.Savannas,
        IGBPValues.Grasslands,
        IGBPValues.Barren,
    ],
    CLCValues.Wetlands: [
        IGBPValues.PermanentWetlands,
        LCCSSHValues.WoodyWetlands,
        LCCSSHValues.HerbaceousWetlands,
    ],
    CLCValues.WaterBodies: [IGBPValues.WaterBodies],
    CLCValues.NoData: [
        LCCSSHValues.Tundra,
        IGBPValues.PermanentSnowAndIce,
    ]
}


Sen12MSValuesToCLCValues: dict[int, dict[Sen12MSLCValues, CLCValues]] = {
    0: {
        IGBPValues.UrbanAndBuiltUpLands: CLCValues.ArtificialSurfaces,
        IGBPValues.Croplands: CLCValues.AgriculturalAreas,
        IGBPValues.CroplandNaturalVegetationMosaics: CLCValues.AgriculturalAreas,
        IGBPValues.EvergreenNeedleleafForests: CLCValues.ForestAndSemiNaturalAreas,
        IGBPValues.EvergreenBroadleafForests: CLCValues.ForestAndSemiNaturalAreas,
        IGBPValues.DeciduousNeedleleafForests: CLCValues.ForestAndSemiNaturalAreas,
        IGBPValues.DeciduousBroadleafForests: CLCValues.ForestAndSemiNaturalAreas,
        IGBPValues.MixedForests: CLCValues.ForestAndSemiNaturalAreas,
        IGBPValues.ClosedDenseShrublands: CLCValues.ForestAndSemiNaturalAreas,
        IGBPValues.OpenSparseShrublands: CLCValues.ForestAndSemiNaturalAreas,
        IGBPValues.WoodySavannas: CLCValues.ForestAndSemiNaturalAreas,
        IGBPValues.Savannas: CLCValues.ForestAndSemiNaturalAreas,
        IGBPValues.Grasslands: CLCValues.ForestAndSemiNaturalAreas,
        IGBPValues.Barren: CLCValues.ForestAndSemiNaturalAreas,
        IGBPValues.PermanentWetlands: CLCValues.Wetlands,
        IGBPValues.WaterBodies: CLCValues.WaterBodies,
        IGBPValues.PermanentSnowAndIce: CLCValues.NoData,
        IGBPValues.NoData: CLCValues.NoData,
    },
    1: {
        LCCSLCValues.MixedBroadleafNeedleafForests: CLCValues.ForestAndSemiNaturalAreas,
        LCCSLCValues.MixedBroadleafEvergreenDeciduousForests: CLCValues.ForestAndSemiNaturalAreas,
        LCCSLCValues.OpenForests: CLCValues.ForestAndSemiNaturalAreas,
        LCCSLCValues.SparseForests: CLCValues.ForestAndSemiNaturalAreas,
        LCCSLCValues.DenseHerbaceous: CLCValues.ForestAndSemiNaturalAreas,
        LCCSLCValues.SparseHerbaceous: CLCValues.ForestAndSemiNaturalAreas,
        LCCSLCValues.ShrublandGrasslandMosaics: CLCValues.ForestAndSemiNaturalAreas,
    },
    2: {
        LCCSLUValues.ForestCroplandMosaics: CLCValues.AgriculturalAreas,
        LCCSLUValues.NaturalHerbaceousCroplandsMosaics: CLCValues.AgriculturalAreas,
        LCCSLUValues.HerbaceousCroplands: CLCValues.AgriculturalAreas,
        LCCSLUValues.DenseForests: CLCValues.ForestAndSemiNaturalAreas,
        LCCSLUValues.NaturalHerbaceous: CLCValues.ForestAndSemiNaturalAreas,
        LCCSLUValues.Shrublands: CLCValues.ForestAndSemiNaturalAreas,
    },
    3: {
        LCCSSHValues.WoodyWetlands: CLCValues.Wetlands,
        LCCSSHValues.HerbaceousWetlands: CLCValues.Wetlands,
        LCCSSHValues.Tundra: CLCValues.NoData,
    },
}


def convert_sen12ms_lc_to_clc_lc(
    sen12ms_lc_path: Path,
    clc_out_path: Path,
    calculate_clc_dist: bool = True,
) -> tuple[dict[Sen12MSLCValues, int], dict[CLCValues, int]]:
    # clc_out_path = clc_base_out_path / sen12ms_lc_path.name
    with rasterio.open(str(sen12ms_lc_path)) as src:
        img = src.read()
        meta = src.meta.copy()

    sen12ms_dist: dict[Sen12MSLCValues, int] = {}
    clc_dist: dict[CLCValues, int] = {}

    clc_img: np.ndarray = np.ones(img.shape[1:], dtype=img.dtype) * CLCValues.NoData.value
    for band_idx in LCBands.ALL.value:
        band_idx -= 1
        band = img[band_idx]
        for sen12ms_lc, clc_lc in Sen12MSValuesToCLCValues[band_idx].items():
            lc_idx = np.where(band == sen12ms_lc.value)
            # avoid overwriting data if already using some label for a pixel from a previous LC Sen12MS category
            nodata_mask = clc_img[lc_idx] == 255
            lc_idx = lc_idx[0][nodata_mask], lc_idx[1][nodata_mask]
            sen12ms_dist[sen12ms_lc] = lc_idx[0].shape[0]
            clc_img[lc_idx] = clc_lc.value

    meta['count'] = 1
    with rasterio.open(str(clc_out_path), 'w', **meta) as dest:
        dest.write(clc_img[None, ...])

    if calculate_clc_dist:
        for clc_lc in list(CLCValuesToSen12MSValues.keys()):
            lc_idx = np.where(clc_img == clc_lc.value)
            clc_dist[clc_lc] = lc_idx[0].shape[0]

    return sen12ms_dist, clc_dist


def update_dict_with_int_values(dict0: dict[Any, int], dict1: dict[Any, int]) -> dict[Any, int]:
    new_keys: list[Any] = list(set(list(dict1.keys())).difference(set(list(dict0.keys()))))
    for key in dict0.keys():
        dict0[key] += dict1.get(key, 0)

    for new_key in new_keys:
        dict0[new_key] = dict1[new_key]

    return dict0


def convert_sen12ms_labels_to_clc_labels(sen12ms_data_base_path: Path, clc_data_base_path: Path):
    if sen12ms_data_base_path == clc_data_base_path:
        raise RuntimeError(
            f"sen12ms data path is the same as clc data path: {sen12ms_data_base_path} == {clc_data_base_path}"
        )

    roi_dirs = [path for path in sorted(sen12ms_data_base_path.glob("ROIs*")) if path.is_dir()]
    lc_img_paths: list[Path] = []
    for roi_dir in roi_dirs:
        lc_img_paths.extend(sorted(roi_dir.glob("**/*lc*.tif")))

    sen12ms_dist: dict[Sen12MSLCValues, int] = {}
    clc_dist: dict[CLCValues: int] = {}

    for sen_lc_img_path in tqdm(lc_img_paths, desc="Converting Sen12MS to CLC"):
        relative_path = sen_lc_img_path.relative_to(sen12ms_data_base_path)
        clc_out_path = clc_data_base_path / relative_path
        clc_out_path.parent.mkdir(exist_ok=True, parents=True)
        sen12ms_dist_, clc_dist_ = convert_sen12ms_lc_to_clc_lc(
            sen12ms_lc_path=sen_lc_img_path,
            clc_out_path=clc_out_path,
        )
        sen12ms_dist = update_dict_with_int_values(sen12ms_dist, sen12ms_dist_)
        clc_dist = update_dict_with_int_values(clc_dist, clc_dist_)

    sen12ms_dist_path = sen12ms_data_base_path / "labels-dist.json"
    sen12ms_dist_path.write_text(jsonpickle.dumps(sen12ms_dist))

    clc_dist_path = clc_data_base_path / "labels-dist.json"
    clc_dist_path.write_text(jsonpickle.dumps(clc_dist))


def main():
    if ClcDataPath.exists():
        raise RuntimeError(f"{ClcDataPath} already exists. Are you sure you want to overwrite that data.")

    convert_sen12ms_labels_to_clc_labels(
        sen12ms_data_base_path=MainDataPath,
        clc_data_base_path=ClcDataPath,
    )


if __name__ == '__main__':
    main()
