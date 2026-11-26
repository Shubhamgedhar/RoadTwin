"""Download RDD2022 dataset countries with resumable HTTP range requests.

Supports downloading and extracting all RDD2022 countries:
India, Japan, Czech, Norway, United_States, China_Drone, China_MotorBike.

With no arguments, downloads all countries. Resumable via HTTP range requests
and automatically refreshes expiring Figshare 403 pre-signed download URLs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Sequence

FIGSHARE_ARTICLE_URL = "https://api.figshare.com/v2/articles/21431547"
ZIP_FILENAME = "RDD2022_released_through_CRDDC2022.zip"

ALL_COUNTRIES = [
    "India",
    "Japan",
    "Czech",
    "Norway",
    "United_States",
    "China_Drone",
    "China_MotorBike",
]


def get_figshare_download_url() -> tuple[str, int]:
    """Query Figshare API to get the fresh download URL and expected size for RDD2022."""
    req = urllib.request.Request(
        FIGSHARE_ARTICLE_URL,
        headers={"User-Agent": "RoadTwin-Dataset-Fetcher/1.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    for f in data.get("files", []):
        if f.get("name") == ZIP_FILENAME:
            return f["download_url"], f["size"]

    raise RuntimeError(f"Could not find {ZIP_FILENAME} in Figshare article {FIGSHARE_ARTICLE_URL}")


def download_with_resume(
    dest_path: Path,
    chunk_size: int = 1024 * 1024 * 4,  # 4 MB
    max_retries: int = 20,
) -> None:
    """Download the file using HTTP range requests with 403 refresh and resume support."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    existing_bytes = dest_path.stat().st_size if dest_path.exists() else 0

    download_url, total_size = get_figshare_download_url()

    if existing_bytes >= total_size:
        print(f"Archive already fully downloaded ({existing_bytes:,} / {total_size:,} bytes).")
        return

    retries = 0
    start_time = time.time()
    last_print = 0.0

    while existing_bytes < total_size:
        headers = {
            "User-Agent": "RoadTwin-Dataset-Fetcher/1.0",
            "Accept-Encoding": "identity",
        }
        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"

        req = urllib.request.Request(download_url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                status = resp.status
                if status not in (200, 206):
                    raise urllib.error.HTTPError(
                        download_url, status, f"Unexpected HTTP status {status}", resp.headers, None
                    )

                with open(dest_path, "ab" if existing_bytes > 0 else "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        existing_bytes += len(chunk)

                        now = time.time()
                        if now - last_print >= 2.0 or existing_bytes >= total_size:
                            last_print = now
                            elapsed = max(0.001, now - start_time)
                            speed_mb = (existing_bytes / elapsed) / (1024 * 1024)
                            pct = (existing_bytes / total_size) * 100
                            print(
                                f"\rDownloading {ZIP_FILENAME}: {existing_bytes / (1024**3):.2f} / "
                                f"{total_size / (1024**3):.2f} GB ({pct:.1f}%) "
                                f"at {speed_mb:.1f} MB/s",
                                end="",
                                flush=True,
                            )
            retries = 0  # reset retries on success chunk
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            retries += 1
            if retries > max_retries:
                print(f"\nExceeded maximum retries ({max_retries}). Last error: {exc}")
                raise
            print(f"\nConnection interrupted ({exc}). Refreshing Figshare URL and resuming in 3s (retry {retries}/{max_retries})...")
            time.sleep(3)
            # Refresh pre-signed URL from Figshare
            download_url, _ = get_figshare_download_url()
            existing_bytes = dest_path.stat().st_size if dest_path.exists() else 0

    print(f"\nDownload finished: {dest_path} ({existing_bytes:,} bytes).")


def verify_zip_integrity(zip_path: Path) -> None:
    """Run zip integrity test to verify no corrupted blocks."""
    print("Verifying zip archive integrity (testzip)...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        first_bad = zf.testzip()
        if first_bad is not None:
            raise RuntimeError(f"Corrupted archive detected! First bad file: {first_bad}")
    print("Zip integrity check PASSED.")


def extract_countries(
    zip_path: Path,
    out_dir: Path,
    requested_countries: Sequence[str],
) -> None:
    """Extract specified countries into data/rdd2022/<Country>/..."""
    with zipfile.ZipFile(zip_path, "r") as outer:
        all_members = outer.namelist()

        # Build mapping of country name -> member path (e.g. 'India' -> 'RDD2022/India.zip')
        country_map = {}
        for member in all_members:
            if member.endswith(".zip"):
                c_name = Path(member).stem
                country_map[c_name] = member

        # Validate requested countries
        invalid = [c for c in requested_countries if c not in country_map]
        if invalid:
            print(f"Error: Requested countries {invalid} not found in archive.")
            print(f"Available countries in archive central directory: {sorted(country_map.keys())}")
            sys.exit(1)

        print(f"Extracting {len(requested_countries)} countries: {list(requested_countries)}...")
        for country in requested_countries:
            member = country_map[country]
            target_country_train = out_dir / country / "train"
            if target_country_train.exists() and any(target_country_train.iterdir()):
                print(f"  Country '{country}' train data already exists at {target_country_train}. Skipping extraction.")
                continue

            print(f"  Extracting {country} from {member}...")
            with outer.open(member) as inner_stream:
                with zipfile.ZipFile(inner_stream) as inner_zip:
                    # Extract train split (with annotations and images) into out_dir
                    inner_zip.extractall(out_dir)

    print("Extraction complete.")



def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract RDD2022 country archives via HTTP range requests."
    )
    parser.add_argument(
        "countries",
        nargs="*",
        default=None,
        help=(
            f"Specific countries to fetch. Defaults to ALL: {ALL_COUNTRIES}"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/rdd2022"),
        help="Target directory for downloaded and extracted files (default: data/rdd2022)",
    )
    args = parser.parse_args(argv)

    countries = args.countries if args.countries else ALL_COUNTRIES
    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / ZIP_FILENAME

    print(f"Target countries: {countries}")
    print(f"Data directory: {data_dir.resolve()}")

    download_with_resume(zip_path)
    verify_zip_integrity(zip_path)
    extract_countries(zip_path, data_dir, countries)


if __name__ == "__main__":
    main()
