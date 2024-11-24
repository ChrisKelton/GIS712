import time
from pathlib import Path

import torch

from Project.model_loaders.resnet import resnet50
from Project.top_level.utils.sen12ms_dataLoader import SEN12MSDataset, Seasons, S1Bands, S2Bands, LCBands


def main():
    model = resnet50(pretrained=True)
    data_path = Path(__file__).parent.resolve().parent / "data"
    sen12ms = SEN12MSDataset(str(data_path))

    spring_ids = sen12ms.get_season_ids(Seasons.SPRING)
    cnt_patches = sum([len(pids) for pids in spring_ids.values()])
    print(f"Spring: {len(spring_ids)} scenes with a total of {cnt_patches} patches")

    start = time.time()
    # Load the RGB Bands of the first S2 patch in scene 8
    SCENE_ID = 8
    s2_rgb_patch, bounds = sen12ms.get_patch(
        season=Seasons.SPRING,
        scene_id=SCENE_ID,
        patch_id=spring_ids[SCENE_ID][0],
        bands=S2Bands.RGB,
    )
    print(f"Time Taken {time.time() - start}s")

    print(f"S2 RGB: {s2_rgb_patch.shape} Bounds: {bounds}\n\n")

    # Load a triplet of patches from the first three scenes of Spring - all S1 bands, NDVI S2 bands, and IGBP LC bands
    start = time.time()
    for idx, (scene_id, patch_ids) in enumerate(spring_ids.items()):
        if idx >= 3:
            break
        s1, s2, lc, bounds = sen12ms.get_s1s2lc_triplet(
            season=Seasons.SPRING,
            scene_id=scene_id,
            patch_id=patch_ids[0],
            s1_bands=S1Bands.ALL,
            s2_bands=[S2Bands.red, S2Bands.nir1],
            lc_bands=LCBands.IGBP,
        )
        print(
            f"Scene: {scene_id}, S1: {s1.shape}, S2: {s2.shape}, LC: {lc.shape}, Bounds: {bounds}")

    print("Time Taken {}s".format(time.time() - start))
    print("\n")

    start = time.time()
    # Load all bands of all patches in a specified scene (scene 106)
    s1, s2, lc, _ = sen12ms.get_triplets(Seasons.SPRING, 106, s1_bands=S1Bands.ALL,
                                        s2_bands=[S2Bands.red, S2Bands.green, S2Bands.blue], lc_bands=LCBands.ALL)

    print(f"Scene: 106, S1: {s1.shape}, S2: {s2.shape}, LC: {lc.shape}")
    print("Time Taken {}s".format(time.time() - start))

    data = torch.Tensor(s2[0].astype(float))
    data = data.to(device=torch.device("cuda"))
    model = model.to(device=torch.device("cuda"))
    tmp = model(data[None, ...])
    a = 0


if __name__ == '__main__':
    main()
