from pathlib import Path
import argparse
import py7zr


def extract_wdoor(archive_path: str, output_dir: str) -> None:
    archive = Path(archive_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with py7zr.SevenZipFile(archive, mode="r") as z:
        z.extractall(path=out)

    print(f"展開完了: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_path")
    parser.add_argument("--output", default="ai/dataset/floodgate")
    args = parser.parse_args()

    extract_wdoor(args.archive_path, args.output)