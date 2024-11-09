from pathlib import Path

import rasterio
import torch
from rasterio.warp import calculate_default_transform, reproject, Resampling

from Project.model_loaders.arcgis_utils import ArcGISImageClassifier


def load_esri_model() -> ArcGISImageClassifier:
    """
    Output raster from model.child_image_classifier.model(x):
        - band 0: NoData
        - band 1: Artificial Surfaces
        - band 2: Agricultural Areas
        - band 3: Forest and semi natural areas
        - band 4: Wetlands
        - band 5: Water Bodies
        - band 6: NoData

    Expects first 12 bands of Sentinel-2:
        - band 1
        - band 2
        - band 3
        - band 4
        - band 5
        - band 6
        - band 7
        - band 8
        - band 8a
        - band 9
        - band 10
        - band 11
    """
    esri_model_path = Path(__file__).parent.resolve() / "initmodel/esri_v2_label1.pth"
    esri_emd_path = esri_model_path.parent / "esri_v2_label1.emd"

    model = ArcGISImageClassifier()
    kwargs = {
        "model": esri_emd_path,
        "device": 0,
    }
    model.initialize(**kwargs)

    state_dict = torch.load(str(esri_model_path))
    model.child_image_classifier.model.load_state_dict(state_dict)

    return model


def main():
    esri_model = load_esri_model()
    reproject_to_wkid_3857: bool = False
    img_path = Path(
        "/home/ckelton/repos/ncsu-masters/GIS712/Project/data/ROIs1158_spring/s2_1/ROIs1158_spring_s2_1_p30.tif")

    cwd: Path = Path(__file__).parent.resolve()

    if reproject_to_wkid_3857:
        dst_crs = "EPSG:3857"
        tmp_reproj = cwd / f"reprojected--{img_path.name}"
        with rasterio.open(str(img_path)) as src:
            transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
            md = src.meta.copy()
            md.update({'crs': dst_crs, 'transform': transform, 'width': width, 'height': height})
            with rasterio.open(str(tmp_reproj), 'w', **md) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest,
                    )
        img_path = tmp_reproj

    with rasterio.open(str(img_path)) as src:
        img = src.read()
        meta = src.meta.copy()

    cpu_device = torch.device("cpu")
    device = torch.device("cuda") if torch.cuda.is_available() else cpu_device

    # model was trained using first 12 bands of Sentinel-2
    img = torch.Tensor(img[:12][None, ...]).to(device)
    if img.shape[2] != img.shape[3]:
        img = torch.nn.functional.pad(img, (1, 1), mode='constant', value=0)
        img = img[..., :img.shape[2]]
    with torch.no_grad():
        tmp = esri_model(img, normalize=True)

    tmp = tmp.to(cpu_device)
    lc = torch.argmax(tmp, dim=1)

    tmp_out = Path(__file__).parent.resolve() / img_path.name.replace("s2", "lc")
    if reproject_to_wkid_3857:
        tmp_out = tmp_out.parent / f"reprojected--{tmp_out.name}"
    meta['count'] = 1
    with rasterio.open(str(tmp_out), 'w', **meta) as dest:
        dest.write(lc)


if __name__ == '__main__':
    main()
