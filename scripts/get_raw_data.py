import logging
import shutil
import urllib.request
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from configs.config import data_dir, data_url

# configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def download_zip(url: str) -> BytesIO:
    """
    Download a ZIP file from a given URL and return its content in memory.

    Args:
        url (str): The URL pointing to the ZIP file.

    Returns:
        BytesIO: In-memory binary stream containing the ZIP data.
    """
    logging.info(f"Starting download from {url}")

    with urllib.request.urlopen(url) as response:
        zip_data = response.read()

    logging.info(f"Download completed ({len(zip_data)} bytes)")

    return BytesIO(zip_data)


def unzip(zip_file: BytesIO | str | Path, dir_path: Path) -> None:
    """
    Extract a ZIP file into the given directory, ignoring '__MACOSX' metadata.

    Args:
        zip_file (BytesIO | str): In-memory ZIP data or path to a ZIP file.
        dir_path (Path): Directory to extract contents into.
    """
    logging.info(f"Extracting ZIP file into {dir_path}")
    dir_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    with ZipFile(file=zip_file, mode="r") as zf:
        members = [m for m in zf.namelist() if "__MACOSX" not in m]
        zf.extractall(path=dir_path, members=members)

    logging.info(f"Extraction complete: {len(members)} files extracted")


def change_data_dir_layout(dir_path: Path) -> None:
    """
    Flatten the directory structure by moving all files from subdirectories
    into the main directory, then remove the empty subdirectories.

    Args:
        dir_path (Path): Directory containing the extracted files.
    """

    logging.info(f"Rearranging directory layout in {dir_path}")

    if not dir_path.is_dir():
        raise ValueError(f"{dir_path} is not a valid directory.")

    moved_files = 0
    removed_folders = 0

    for folder in dir_path.iterdir():
        if folder.is_dir():
            for file in folder.iterdir():
                new_path = dir_path / file.name

                # If a file with the same name exists, overwrite it
                if new_path.exists():
                    logging.warning(f"Overwriting existing file: {new_path}")
                    new_path.unlink()

                shutil.move(str(file), str(new_path))
                moved_files += 1
            folder.rmdir()
            removed_folders += 1
    logging.info(
        f"Layout change complete: {moved_files} files moved, {removed_folders} folders removed"
    )


def main():
    """
    Main execution flow:
    1. Download dataset ZIP.
    2. Extract to 'data_dir/raw'.
    3. Flatten directory structure.
    """
    try:
        logging.info("=== Dataset preparation started ===")
        zip_data = download_zip(url=data_url)

        raw_dir = data_dir / "raw"
        unzip(zip_file=zip_data, dir_path=raw_dir)

        for file in raw_dir.iterdir():
            if ".zip" in file.suffix:
                unzip(file, dir_path=raw_dir)
                file.unlink()

        change_data_dir_layout(raw_dir)

        logging.info("=== Dataset preparation finished successfully ===")
        logging.info(f"Data available at: {raw_dir}")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
