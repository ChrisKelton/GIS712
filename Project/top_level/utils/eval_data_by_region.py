from pathlib import Path
from typing import Optional, Any, Callable, Union

import fiona
import geopandas as gpd
import jsonpickle
import numpy as np
import pandas as pd
import rasterio.coords
import simplekml
from scipy.stats import entropy
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from tqdm import tqdm

from Project.top_level.utils.sen12ms_dataLoader import SEN12MSDataset, Seasons, S2Bands, LCBands, ClcDataPath
from Project.top_level.utils.sen12ms_to_clc import CLCValues

fiona.drvsupport.supported_drivers['KML'] = 'r'


def hav(theta: float) -> float:
    return (np.sin(theta / 2)) ** 2


def haversine_distance(lon0: float, lat0: float, lon1: float, lat1: float, to_km: bool = True) -> float:
    """
    lat1 = phi_1
    lat0 = phi_0
    lon1 = lambda_1
    lon0 = lambda_0

    d = distance between the two points along a great circle of the sphere
    r = radius of the sphere
    theta = d/r

    del_phi = phi_1 - phi_0
    del_lambda = lambda_1 - lambda_0

    hav(theta) = hav(del_phi) + cos(phi_0) * cos(phi_1) * hav(del_lambda)

    hav(theta) = sin^2(theta / 2) = (1 - cos(theta)) / 2

    d = r*archav(hav(theta)) = 2r*arcsin(sqrt(hav(theta)))
      = 2r*arcsin(sqrt(sin^2(del_phi / 2) + (1 - sin^2(del_phi / 2) - sin^2(phi_m)) * sin^2(del_lambda / 2)))

    x = sin^2(del_phi / 2)
    y = (1 - x - sin^2(phi_m))
    z = sin^2(del_lambda / 2)

    phi_m = (phi_1 + phi_0) / 2

    d = 2r*arcsin(sqrt(x + (y * z)))
    """
    lon0, lat0, lon1, lat1 = map(np.radians, [lon0, lat0, lon1, lat1])

    del_phi: float = lat1 - lat0
    del_lambda: float = lon1 - lon0

    radius_at_poles = 6356.752e3
    radius_at_equator = 6378.137e3

    radius = (radius_at_equator + radius_at_poles) / 2

    phi_m = (lat1 + lat0) / 2
    x = (np.sin(del_phi / 2)) ** 2
    z = (np.sin(del_lambda / 2)) ** 2
    y = (1 - x - ((np.sin(phi_m)) ** 2))

    d = (2 * radius) * np.arcsin(np.sqrt(x + (y * z)))
    if to_km:
        d /= 1e3

    h = hav(d / radius)
    if not(0 <= h <= 1):
        raise RuntimeError(f"h = hav(theta) = hav({d / radius}) = {h}, which does not obey 0 <= h <= 1.")

    return d


def get_grouped_items_idx_by_condition(
    items: list[Any],
    condition: Callable,
    description: Optional[str] = None,
    disable_tqdm: bool = False,
) -> dict[int, Optional[list[int]]]:
    """
    :param items: list of objects
    :param condition: condition to group objects by
    :param description: description for tqdm
    :param disable_tqdm: boolean to turn off tqdm progress output
    :return dictionary of items grouped by condition, where the key will be the smallest index of the grouped items
        does return unique items indices in the keys position with None as the value
    """
    items2 = items.copy()

    same_idx: dict[int, Optional[list[int]]] = {}
    for idx, item in tqdm(enumerate(items), total=len(items), desc=f"{description}", disable=disable_tqdm):
        for idx_, item2 in enumerate(items2):
            if idx != idx_ and idx not in same_idx.get(idx_, []) and condition(item, item2):
                used: bool = False
                for same_key, same_vals in same_idx.items():
                    if idx in same_vals and idx_ not in same_vals:
                        same_idx.setdefault(same_key, []).append(idx_)
                        used = True
                        break
                    if idx_ in same_vals and idx not in same_vals:
                        same_idx.setdefault(same_key, []).append(idx)
                        used = True
                        break
                    if idx in same_vals and idx_ in same_vals:
                        used = True
                        break
                if not used:
                    same_idx.setdefault(idx, []).append(idx_)

    temp = list(same_idx.keys())
    for val in same_idx.values():
        temp.extend(val)
    temp = set(temp)
    unique_idx = list(set(np.arange(0, len(items))).difference(temp))
    if len(unique_idx) > 0:
        for idx in unique_idx:
            same_idx[idx] = None

    return same_idx


def convert_bounding_box_to_polygon(bbox: rasterio.coords.BoundingBox, reverse_coords: bool = False) -> Polygon:
    poly = Polygon(list(zip([bbox.left, bbox.right, bbox.right, bbox.left], [bbox.top, bbox.top, bbox.bottom, bbox.bottom])))
    if reverse_coords:
        x, y = poly.exterior.xy
        coords = list(zip(y, x))
        poly = Polygon(coords)

    return poly


def read_kml(kml_path: Path) -> gpd.GeoDataFrame:
    gdf_list: list[gpd.GeoDataFrame] = []
    for layer in fiona.listlayers(str(kml_path)):
        gdf = gpd.read_file(str(kml_path), driver='KML', layer=layer)
        gdf_list.append(gdf)

    return gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))


def create_kml_from_polygon(poly: Polygon, name: str, out_path: Path, reverse_coords: bool = False):
    """
    Doesn't account for holes in polygon
    """
    x, y = poly.exterior.xy
    if reverse_coords:
        coords = list(zip(y, x))
    else:
        coords = list(zip(x, y))

    kml = simplekml.Kml()
    kml.newpolygon(name=name, outerboundaryis=coords)
    kml.save(str(out_path))


def generate_kmls(base_out_path: Path, overwrite_existing_kmls: bool = False):
    mps_by_season_json_path = base_out_path / "mps-by-season.json"

    if not mps_by_season_json_path.exists():
        dataset = SEN12MSDataset.from_clc_data_path()
        seasons: list[Seasons] = [Seasons.SPRING, Seasons.FALL]

        mps: dict[str, MultiPolygon] = {}
        for season in seasons:
            print(f"Aggregating season '{season}'")
            ids: dict[int, list[int]] = dataset.get_season_ids(season)
            polys: list[Polygon] = []
            for key, vals in tqdm(ids.items(), desc="Scene Ids"):
                for val in tqdm(vals, desc="Patches"):
                    _, bounds = dataset.get_patch(
                        season=season,
                        scene_id=key,
                        patch_id=val,
                        bands=S2Bands.red,
                        convert_bounds_to_lonlat=True,
                    )
                    polys.append(Polygon(list(zip([bounds.left, bounds.right, bounds.right, bounds.left], [bounds.top, bounds.top, bounds.bottom, bounds.bottom]))))

            mp = unary_union(MultiPolygon(polys).geoms)
            # do convex hull here, b/c we do convex hull when creating the KML and the newly unioned multipolygons may
            # now overlap each other, whereas before the individual polygons may not have
            mps[season.value] = unary_union(MultiPolygon([poly.convex_hull for poly in mp.geoms]).geoms)

        mps_by_season_json_path.write_text(jsonpickle.dumps(mps, indent=2))
    else:
        mps = jsonpickle.loads(mps_by_season_json_path.read_text())

    for season, mp in mps.items():
        out_path = base_out_path / season
        if not overwrite_existing_kmls and out_path.exists():
            continue
        out_path.mkdir(exist_ok=True, parents=True)

        print(f"*** Generating KMLs for {season} ***")
        for idx, poly in tqdm(enumerate(mp.geoms), total=len(mp.geoms)):
            create_kml_from_polygon(
                poly=poly.convex_hull,
                name=f"{season}--{idx}",
                out_path=out_path / f"{idx}.kml",
                reverse_coords=True,
            )


LabelDistributionsType = dict[str, dict[str, dict[str, dict[str, float]]]]


def generate_label_distributions_per_season(base_kmls_path: Path, overwrite: bool = False) -> Path:
    label_distributions_json_path = base_kmls_path / "label-distributions-by-season.json"

    if not overwrite and label_distributions_json_path.exists():
        print(f"'{label_distributions_json_path}' already exists. Returning...")
        return label_distributions_json_path

    spring_kml_paths = sorted((base_kmls_path / "ROIs1158_spring").glob("*.kml"))
    fall_kml_paths = sorted((base_kmls_path / "ROIs1970_fall").glob("*.kml"))
    kml_paths: dict[str, list[Path]] = {
        Seasons.SPRING.value: spring_kml_paths,
        Seasons.FALL.value: fall_kml_paths,
    }

    dataset = SEN12MSDataset.from_clc_data_path()
    seasons: list[Seasons] = [Seasons.SPRING, Seasons.FALL]

    # dict[str, dict[str, dict[str, dict[str, dict[str, float]]]] = {
    #   'Spring': {
    #       f'{idx}.kml': {
    #           f'{scene_id}-{patch_id}': {
    #               AgriculturalAreas: %,
    #               ...,
    #           },
    #           ...,
    #       },
    #       ...,
    #   },
    #   'Fall': {
    #       f'{idx}.kml': {
    #           f'{scene_id}-{patch_id}': {
    #               AgriculturalAreas: %,
    #               ...,
    #           },
    #           ...,
    #       },
    #       ...,
    #   },
    # }
    #
    # Season -> KML -> SceneId-PatchId -> CLCLabel -> Percentage
    label_distributions_by_season:LabelDistributionsType = {}
    for season in seasons:
        ids: dict[int, list[int]] = dataset.get_season_ids(season)

        # KML -> SceneId-PatchId -> CLCLabel -> Percentage
        seasonal_distribution: dict[str, dict[str, dict[str, float]]] = {}
        for idx, (scene_id, patch_ids) in tqdm(enumerate(ids.items()), desc="Scene Ids", total=len(ids)):
            print(f"\nScene {idx} / {len(ids)}\n")
            for patch_id in tqdm(patch_ids, desc="Patch Ids", position=0, leave=True):
                l2_band, bounds = dataset.get_patch(
                    season=season,
                    scene_id=scene_id,
                    patch_id=patch_id,
                    bands=LCBands.IGBP,  # Only 1-band in CLC data
                    convert_bounds_to_lonlat=True,
                )
                bounds = convert_bounding_box_to_polygon(bounds, reverse_coords=True)
                vals, cnts = np.unique(l2_band, return_counts=True)
                cnts = cnts / np.sum(cnts)
                cnts *= 100

                vals = list(vals)
                cnts = list(cnts)

                patch_distribution: dict[str, float] = {}
                # iterate over all possible labels to calculate entropy later to determine most entropic labels per kml
                for clc_key, clc_label in CLCValues.as_dict().items():
                    try:
                        vals_idx = vals.index(clc_label)
                        cnt = float(cnts[vals_idx])
                    except ValueError:
                        cnt = 0.
                    patch_distribution[clc_key] = cnt

                kml_path_name: Optional[str] = None
                for kml_path in kml_paths[season.value]:
                    kml_poly = read_kml(kml_path).geometry[0]
                    if kml_poly.overlaps(bounds) or kml_poly.intersects(bounds) or kml_poly.contains(bounds):
                        kml_path_name = kml_path.name
                        break

                seasonal_distribution.setdefault(kml_path_name, {})
                seasonal_distribution[kml_path_name].update({f"{scene_id}-{patch_id}": patch_distribution.copy()})

        label_distributions_by_season[season.value] = seasonal_distribution.copy()

    label_distributions_json_path.write_text(jsonpickle.dumps(label_distributions_by_season, indent=2))

    return label_distributions_json_path


def entropy_from_label_distributions(
    label_distributions_json_path: Path,
    out_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    if out_path is None:
        out_path = label_distributions_json_path.parent / f"{label_distributions_json_path.stem}--w-entropy.json"

    if out_path.exists() and not overwrite:
        return out_path

    label_dists = jsonpickle.loads(label_distributions_json_path.read_text())

    label_dists_w_entropy = label_dists.copy()
    # get entropy per label
    for season, season_dist in label_dists.items():
        for kml, scene_dists in tqdm(season_dist.items(), desc="Scene Ids"):
            for scene, label_dist in scene_dists.items():
                entropy_val = entropy(list(label_dist.values()))
                label_dists_w_entropy[season][kml][scene]["entropy"] = float(entropy_val)
            label_dists_w_entropy[season][kml] = dict(
                sorted(
                    label_dists_w_entropy[season][kml].items(),
                    key=lambda item: item[1]["entropy"],
                    reverse=True,
                )
            )

    out_path.write_text(jsonpickle.dumps(label_dists_w_entropy, indent=2))

    return out_path


def get_highest_entropy_labels_by_general_region(
    json_path: Path,
    base_kml_path: Path,
    labels_per_general_region: int = 5,
    distance_to_combine_regions_km: float = 1000,  # 20,037.5 kilometers is maximum distance between two points on earth
    calculate_entropy: bool = False,
) -> Path:
    """
    @param json_path:
    @param base_kml_path:
    @param labels_per_general_region:
    @param distance_to_combine_regions_km: distance between kml centroids to combine into 1 general region
    @param calculate_entropy:

    @return
    """
    spring_kml_paths = sorted((base_kml_path / Seasons.SPRING.value).glob("*.kml"))
    fall_kml_paths = sorted((base_kml_path / Seasons.FALL.value).glob("*.kml"))

    # kmls are stored as lon/lat
    spring_kmls: dict[str, Polygon] = {}
    for kml_path in spring_kml_paths:
        spring_kmls[kml_path.name] = read_kml(kml_path).geometry[0]

    fall_kmls: dict[str, Polygon] = {}
    for kml_path in fall_kml_paths:
        fall_kmls[kml_path.name] = read_kml(kml_path).geometry[0]

    def kmls_are_general_region(kml0: Polygon, kml1: Polygon) -> bool:
        def get_lonlat_from_polygon_centroid(poly: Polygon) -> tuple[float, float]:
            lon, lat = poly.centroid.xy
            lon = np.asarray(lon)[0]
            lat = np.asarray(lat)[0]

            return lon, lat

        lon0, lat0 = get_lonlat_from_polygon_centroid(kml0)
        lon1, lat1 = get_lonlat_from_polygon_centroid(kml1)

        return haversine_distance(lon0, lat0, lon1, lat1) <= distance_to_combine_regions_km

    def group_kmls(kmls: dict[str, Polygon]) -> list[list[str]]:
        grouped_kmls: dict[int, Optional[list[int]]] = get_grouped_items_idx_by_condition(
            items=list(kmls.values()),
            condition=kmls_are_general_region,
            description="Grouping KMLs",
        )
        # # Extra check since some connected regions don't end up connected and duplicate regions
        # different_kml_groups: list[set[int]] = []
        # for key, val in grouped_kmls.items():
        #     grouped_idxs = [key]
        #     if val is not None:
        #         grouped_idxs.extend(val)
        #     different_kml_groups.append(set(grouped_idxs))
        #
        # groups_to_combine: list[tuple[int, int]] = []
        # for idx0, idx1 in itertools.combinations(range(len(different_kml_groups)), r=2):
        #     group0, group1 = different_kml_groups[idx0], different_kml_groups[idx1]
        #     if len(group0.intersection(group1)) > 0:
        #         groups_to_combine.append((idx0, idx1))

        # TODO: for some reason the same kmls are getting grouped into different kml groups
        grouped_kml_keys: list[list[str]] = []
        kml_keys = list(kmls.keys())
        for key, val in grouped_kmls.items():
            kml_idxs = [key]
            if val is not None:
                kml_idxs.extend(val)
            grouped_kml_keys_: list[str] = []
            for kml_idx in kml_idxs:
                grouped_kml_keys_.append(kml_keys[kml_idx])

            grouped_kml_keys.append(grouped_kml_keys_)

        return grouped_kml_keys

    grouped_spring_kml_paths: list[list[str]] = group_kmls(spring_kmls)
    grouped_fall_kml_paths: list[list[str]] = group_kmls(fall_kmls)

    grouped_kml_paths: dict[str, list[list[str]]] = {
        Seasons.SPRING.value: grouped_spring_kml_paths,
        Seasons.FALL.value: grouped_fall_kml_paths,
    }

    if calculate_entropy:
        json_path = entropy_from_label_distributions(
            label_distributions_json_path=json_path,
            overwrite=True,
        )

    label_dists: LabelDistributionsType = jsonpickle.loads(json_path.read_text())

    '''
    {season: {kmls: {scene_id-patch_id: {label_name: float}}}}
    '''
    top_labels_by_dist: LabelDistributionsType = {}
    for season, dists_by_kml in label_dists.items():
        grouped_kmls = grouped_kml_paths[season]
        labels_by_grouped_kmls: dict[str, dict[str, dict[str, Union[float, str]]]] = {}
        for kml_filenames in grouped_kmls:
            scene_ids_and_patch_ids: dict[str, dict[str, float]] = {}
            for kml_filename in kml_filenames:
                kml_dist = dists_by_kml[kml_filename]
                scene_ids_and_patch_ids_: dict[str, dict[str, Union[float, str]]] = {}
                for scene_id_and_patch_id_key, label_dists in kml_dist.items():
                    label_dists["kml"] = kml_filename
                    scene_ids_and_patch_ids_[scene_id_and_patch_id_key] = label_dists.copy()
                scene_ids_and_patch_ids.update(scene_ids_and_patch_ids_.copy())
            scene_ids_and_patch_ids = dict(
                sorted(
                    scene_ids_and_patch_ids.items(),
                    key=lambda item: item[1]["entropy"],
                    reverse=True,
                )
            )
            top_scene_ids_and_patch_ids_keys = list(scene_ids_and_patch_ids.keys())[:labels_per_general_region]
            general_regions = {}
            for key in top_scene_ids_and_patch_ids_keys:
                general_regions[key] = scene_ids_and_patch_ids[key]
            labels_by_grouped_kmls["-".join([filename.strip(".kml") for filename in kml_filenames])] = general_regions.copy()
        top_labels_by_dist[season] = labels_by_grouped_kmls.copy()

    out_json_path = json_path.parent / f"top-{labels_per_general_region}--{json_path.stem}.json"
    out_json_path.write_text(jsonpickle.dumps(top_labels_by_dist, indent=2))

    return out_json_path


def get_highest_entropy_dist(
    scene_ids_and_patch_ids_labels: dict[str, dict[str, dict[str, float]]],
    *,
    ensure_all_labels_present: bool = True,
    keys_to_ignore: Optional[list[str]] = None,  # only applicable if ensure_all_labels_present is False
) -> Optional[dict[str, Union[float, str]]]:
    # subtract 1 to account for NoData class
    n_labels = len(list(CLCValues.as_dict().keys())) - 1
    key_to_use: Optional[str] = None
    if ensure_all_labels_present:
        for scene_id_and_patch_id_key, label_dist in scene_ids_and_patch_ids_labels.items():
            cnt = 0
            for label_name in CLCValues.as_dict().keys():
                if label_dist[label_name] > 0:
                    cnt += 1
            if cnt == n_labels:
                key_to_use = scene_id_and_patch_id_key
                break
    else:
        if keys_to_ignore is not None:
            possible_keys = list(set(list(scene_ids_and_patch_ids_labels.keys())).difference(set(keys_to_ignore)))
        else:
            possible_keys = list(scene_ids_and_patch_ids_labels.keys())
        if len(possible_keys) == 0:
            return None
        key_to_use = possible_keys[0]

    if key_to_use is None:
        return None

    scene_id_and_patch_id_dist = scene_ids_and_patch_ids_labels[key_to_use]
    scene_id, patch_id = key_to_use.split("-")
    scene_id_and_patch_id_dist["scene_id"] = scene_id
    scene_id_and_patch_id_dist["patch_id"] = patch_id
    return scene_id_and_patch_id_dist


def get_support_set_from_highest_entropy_labels_by_general_region(
    json_path: Path,
    n_labels_in_support_set: int,
    overwrite: bool = False,
) -> Path:
    out_json_path = json_path.parent / f"support-set--{n_labels_in_support_set}.json"
    if not overwrite and out_json_path.exists():
        print(f"'{out_json_path}' exists. Not overwriting...")
        return out_json_path

    top_labels_by_dist = jsonpickle.loads(json_path.read_text())

    # subtract 1 to account for NoData class
    support_sets_by_season: dict[str, list[dict[str, Union[float, str]]]] = {}
    for season, labels_by_grouped_kmls in top_labels_by_dist.items():
        '''
        {label_names: float}
        '''
        support_set_candidates: list[dict[str, Union[float, str]]] = []

        # 1st pass, be strict and ensure that all labels are present in highest entropy label distribution
        for kml_group, scene_ids_and_patch_ids_labels in labels_by_grouped_kmls.items():
            if len(support_set_candidates) > 0:
                support_set_candidates = sorted(
                    support_set_candidates,
                    key=lambda item: item["entropy"],
                    reverse=True,
                )
            highest_entropy_dist = get_highest_entropy_dist(
                scene_ids_and_patch_ids_labels=scene_ids_and_patch_ids_labels
            )
            if highest_entropy_dist is not None:
                if season == Seasons.FALL.value and highest_entropy_dist["kml"] in ["6.kml", "90.kml"]:
                    a = 0
                if len(support_set_candidates) == n_labels_in_support_set:
                    # check if entropy of current candidate is higher than any that have already been stored
                    if support_set_candidates[-1]["entropy"] < highest_entropy_dist["entropy"]:
                        support_set_candidates[-1] = highest_entropy_dist.copy()
                else:
                    # automatically include in support_set_candidates
                    support_set_candidates.append(highest_entropy_dist.copy())

        if len(support_set_candidates) < n_labels_in_support_set:
            # 2nd pass, not strict about ensuring every label is present in highest entropy label distribution
            keys_to_ignore: list[str] = [
                f"{candidate['scene_id']}-{candidate['patch_id']}" for candidate in support_set_candidates
            ]
            for kml_group, scene_ids_and_patch_ids_labels in labels_by_grouped_kmls.items():
                if len(support_set_candidates) > 0:
                    support_set_candidates = sorted(
                        support_set_candidates,
                        key=lambda item: item["entropy"],
                        reverse=True,
                    )
                highest_entropy_dist = get_highest_entropy_dist(
                    scene_ids_and_patch_ids_labels=scene_ids_and_patch_ids_labels,
                    ensure_all_labels_present=False,
                    keys_to_ignore=keys_to_ignore,
                )
                if highest_entropy_dist is not None:
                    if len(support_set_candidates) == n_labels_in_support_set:
                        # check if entropy of current candidate is higher than any that have already been stored
                        if support_set_candidates[-1]["entropy"] < highest_entropy_dist["entropy"]:
                            support_set_candidates[-1] = highest_entropy_dist.copy()
                    else:
                        # automatically include in support_set_candidates
                        support_set_candidates.append(highest_entropy_dist.copy())
        support_sets_by_season[season] = support_set_candidates.copy()

    out_json_path.write_text(jsonpickle.dumps(support_sets_by_season, indent=2))

    return out_json_path


def produce_rel_data_paths_from_season_scene_id_and_patch_id(
    season: str,  # [Seasons.FALL, Seasons.SPRING]
    scene_id: str,
    patch_id: str,
    base_data_path: Optional[Path] = None,
) -> tuple[Path, Path]:
    """
    :returns (rel path to l1, rel path to s2)
    """
    lc_path = Path(f"{season}/lc_{scene_id}/{season}_lc_{scene_id}_p{patch_id}.tif")
    s2_path = Path(f"{season}/s2_{scene_id}/{season}_s2_{scene_id}_p{patch_id}.tif")

    if base_data_path is not None:
        # check if files do exist
        lc_path_ = base_data_path / lc_path
        lc_path_exists = lc_path_.exists()
        s2_path_ = base_data_path / s2_path
        s2_path_exists = s2_path_.exists()

        if not (lc_path_exists and s2_path_exists):
            str_ = ""
            if not lc_path_exists:
                str_ += f"LC Path does not exist at '{lc_path_}'\n"
            if not s2_path_exists:
                str_ += f"S2 Path does not exist at '{s2_path_}'"
            raise RuntimeError(f"{str_}")

    return lc_path, s2_path


def get_datasets_from_support_set(
    total_set_by_season_json_path: Path,
    support_set_json_path: Path,
    base_out_path: Optional[Path] = None,
    base_data_path: Optional[Path] = None,
    overwrite: bool = False,
) -> dict[str, dict[str, Path]]:

    if base_data_path is None and base_out_path is None:
        raise RuntimeError(f"base_data_path is None and base_out_path is None. One or both must be provided.")
    elif base_data_path is not None and base_out_path is None:
        base_out_path = base_data_path / support_set_json_path.stem
    base_out_path.mkdir(exist_ok=True, parents=True)

    total_set = jsonpickle.loads(total_set_by_season_json_path.read_text())
    support_set = jsonpickle.loads(support_set_json_path.read_text())

    csv_paths_by_season: dict[str, dict[str, Path]] = {}
    paths_exist: list[bool] = []
    for season in total_set.keys():
        csv_path = base_out_path / f"{season}--not-support-set.csv"
        if csv_path.exists():
            paths_exist.append(True)
        else:
            paths_exist.append(False)
        csv_paths_by_season.setdefault(season, {})["not-support-set"] = csv_path

        csv_path = base_out_path / f"{season}--support-set.csv"
        if csv_path.exists():
            paths_exist.append(True)
        else:
            paths_exist.append(False)
        csv_paths_by_season.setdefault(season, {})["support-set"] = csv_path

    if not overwrite and all(paths_exist):
        print(f"Datasets that don't include the support set exist. Returning...")
        return csv_paths_by_season

    df_columns: list[str] = [
        "S2",
        "LC",
        "entropy",
        "ArtificialSurfaces",
        "AgriculturalAreas",
        "ForestAndSemiNaturalAreas",
        "Wetlands",
        "WaterBodies",
        "NoData",
    ]
    for season, labels_by_season in total_set.items():
        support_set_by_season = support_set[season]
        csv_path = csv_paths_by_season[season]["support-set"]

        support_set_df: pd.DataFrame = pd.DataFrame(columns=df_columns)
        for support_set_ in support_set_by_season:
            scene_id = support_set_['scene_id']
            patch_id = support_set_['patch_id']
            entropy_ = support_set_['entropy']
            lc_path, s2_path = produce_rel_data_paths_from_season_scene_id_and_patch_id(
                season=season,
                scene_id=scene_id,
                patch_id=patch_id,
                base_data_path=base_data_path,
            )
            support_set_df.loc[len(support_set_df)] = [
                str(s2_path),
                str(lc_path),
                entropy_,
                *[support_set_[label] for label in df_columns[3:]],
            ]
        support_set_df.to_csv(str(csv_path), index=None)

        support_set_scene_and_patch_ids_by_kml: dict[str, list[str]] = {}
        for support_set_ in support_set_by_season:
            support_set_scene_and_patch_ids_by_kml.setdefault(
                support_set_['kml'], []).append(f"{support_set_['scene_id']}-{support_set_['patch_id']}")

        seasonal_dataset: pd.DataFrame = pd.DataFrame(columns=df_columns)
        for kml, labels_by_kml in labels_by_season.items():
            support_set_by_kml = support_set_scene_and_patch_ids_by_kml.get(kml, [])
            for scene_and_patch_id, label_dists in labels_by_kml.items():
                if scene_and_patch_id in support_set_by_kml:
                    continue
                scene_id, patch_id = scene_and_patch_id.split("-")
                lc_path, s2_path = produce_rel_data_paths_from_season_scene_id_and_patch_id(
                    season=season,
                    scene_id=scene_id,
                    patch_id=patch_id,
                    base_data_path=base_data_path,
                )
                entropy_ = entropy(list(label_dists.values()))
                seasonal_dataset.loc[len(seasonal_dataset)] = [
                    str(s2_path),
                    str(lc_path),
                    float(entropy_),
                    *[label_dists[label] for label in df_columns[3:]],
                ]
        csv_path = csv_paths_by_season[season]["not-support-set"]
        seasonal_dataset.to_csv(str(csv_path), index=None)

    return csv_paths_by_season


def main():
    base_out_path = Path("./kmls")
    generate_kmls(base_out_path)
    label_distributions_json_path = generate_label_distributions_per_season(
        base_kmls_path=base_out_path,
        overwrite=False,
    )
    label_distributions_w_entropy_json_path = entropy_from_label_distributions(label_distributions_json_path)
    highest_entropy_per_kml_groups_json_path = get_highest_entropy_labels_by_general_region(
        json_path=label_distributions_w_entropy_json_path,
        base_kml_path=base_out_path,
    )
    support_set_paths: dict[int, Path] = {}
    csv_paths_by_support_set: dict[int, dict[str, Path]] = {}
    for k in tqdm([1, 3, 5], desc="Generating Support Sets"):
        support_set_path = get_support_set_from_highest_entropy_labels_by_general_region(
            json_path=highest_entropy_per_kml_groups_json_path,
            n_labels_in_support_set=k,
        )
        support_set_paths[k] = support_set_path
        csv_paths_by_support_set[k] = get_datasets_from_support_set(
            label_distributions_json_path,
            support_set_path,
            base_data_path=ClcDataPath,
            overwrite=False,
        )


if __name__ == '__main__':
    main()
