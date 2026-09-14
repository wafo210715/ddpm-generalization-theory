"""Fetch the free, no-registration external layers named in the acquisition plan.

Everything here is open-licensed and needs no account, so it can run unattended.
The layers that need a decision or a credential are deliberately excluded and
listed at the bottom of the run.
"""
import pathlib, sys, time
import requests

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "data/external"
UA = {"User-Agent": "hcmc-vitality-research/1.0 (academic)"}

GHSL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"
LAYERS = {
    # factor: building height -> floor-area ratio.  Per-building heights, CC BY 4.0.
    "3dglobfp/Vietnam.rar":
        "https://zenodo.org/api/records/11397015/files/Vietnam.rar/content",
    # coarse height fallback, normalised over the built fraction (AGBH, not ANBH)
    "ghsl/GHS_BUILT_H_AGBH_E2018_R8_C29.zip":
        f"{GHSL}/GHS_BUILT_H_GLOBE_R2023A/GHS_BUILT_H_AGBH_E2018_GLOBE_R2023A_54009_100/V1-0/tiles/"
        "GHS_BUILT_H_AGBH_E2018_GLOBE_R2023A_54009_100_V1_0_R8_C29.zip",
    # factor: land cover -> the land-use-mix substitute
    "worldcover/ESA_WorldCover_10m_2021_v200_N09E105_Map.tif":
        "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
        "ESA_WorldCover_10m_2021_v200_N09E105_Map.tif",
    # factor: population.  Building-weighted, so partly circular -- see the plan.
    "worldpop/vnm_ppp_2020_constrained.tif":
        "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/VNM/"
        "vnm_ppp_2020_constrained.tif",
}
# factor: building age.  Epoch of built-up appearance, 100 m, one file per epoch.
for year in (1975, 1990, 2000, 2010, 2020):
    LAYERS[f"ghsl/GHS_BUILT_S_E{year}_R8_C29.zip"] = (
        f"{GHSL}/GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E{year}_GLOBE_R2023A_54009_100/V1-0/tiles/"
        f"GHS_BUILT_S_E{year}_GLOBE_R2023A_54009_100_V1_0_R8_C29.zip")

NEEDS_A_DECISION = [
    "LandScan Global 1 km - free but requires registration at landscan.ornl.gov",
    "Mapillary imagery - free, but register your own API token first",
    "ShopeeFood / Coc Coc - open APIs, but a terms-of-service call is yours",
    "SDGSAT-1 40 m night lights - portal registration plus a written proposal",
    "Sentinel-2 dry-season composite - compute, not download; do it in Earth Engine",
    "VIIRS annual night lights - simplest via Earth Engine, no bulk file needed",
]


def get(dst: pathlib.Path, url: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  have  {dst.name} ({dst.stat().st_size/1e6:.0f} MB)", flush=True)
        return
    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(3):
        try:
            headers = dict(UA)
            if tmp.exists():                       # resume a partial file
                headers["Range"] = f"bytes={tmp.stat().st_size}-"
            with requests.get(url, headers=headers, stream=True, timeout=120) as r:
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                mode = "ab" if r.status_code == 206 else "wb"
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            tmp.rename(dst)
            print(f"  got   {dst.name} ({dst.stat().st_size/1e6:.0f} MB)", flush=True)
            return
        except Exception as e:
            print(f"  retry {attempt+1} {dst.name}: {type(e).__name__} {str(e)[:80]}", flush=True)
            time.sleep(10 * (attempt + 1))
    print(f"  FAILED {dst.name}", flush=True)


def main():
    only = sys.argv[1:]
    for rel, url in LAYERS.items():
        if only and not any(o in rel for o in only):
            continue
        get(OUT / rel, url)
    print("\nNot fetched here, each needs a credential or a decision:")
    for line in NEEDS_A_DECISION:
        print(f"  - {line}")


if __name__ == "__main__":
    main()
