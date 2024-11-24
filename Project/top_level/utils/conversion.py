from pathlib import Path
import jsonpickle
from Project.top_level.utils.sen12ms_dataLoader import ClcDataPath
import shutil
import os


def convert_json_with_numpy_contents_to_float(json_path: Path, out_path: Path):
    contents = jsonpickle.loads(json_path.read_text())
    contents_: dict[str, float] = {}
    for key, val in contents.items():
        contents_[key] = float(val)

    out_path.write_text(jsonpickle.dumps(contents_, indent=2))


def main():
    base_json_path = ClcDataPath / "FSL-Training--batch_size-8"
    json_paths = sorted(base_json_path.glob("**/*.json"))
    for json_path in json_paths:
        if json_path.stem == "config.json" or json_path.parent.stem == "archive-jsons":
            continue
        archive_numpy_json_path = json_path.parent / "archive-jsons"
        archive_numpy_json_path.mkdir(exist_ok=True, parents=True)

        archive_numpy_json_path /= json_path.name
        shutil.copy(json_path, archive_numpy_json_path)

        os.remove(str(json_path))

        convert_json_with_numpy_contents_to_float(archive_numpy_json_path, json_path)


if __name__ == '__main__':
    main()
