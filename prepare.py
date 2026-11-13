"""Convert the team's candidate, road-distance and monthly ridership CSVs."""
import argparse
import csv
import json
import math
from pathlib import Path
from io import TextIOWrapper
from zipfile import ZipFile

HUBS = {
    "112900230": "신촌",
    "112000057": "이대", "112900187": "이대",
    "112000410": "아현", "112900121": "아현",
    "112000031": "연대앞", "112000032": "연대앞", "112900188": "연대앞",
}


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def prepare(folder, output):
    candidates = [row for row in rows(folder / "01_region_stations_clean.csv")
                  if row["station_id"].startswith("112")]
    ids = [row["station_id"] for row in candidates]
    selected = set(ids)
    if len(ids) != len(selected) or not set(HUBS) <= selected:
        raise ValueError("Expected unique Seodaemun stop IDs including all configured hub stops")

    edges = {}
    for archive in sorted(folder.glob("11_road_distance_part*.zip")):
        with ZipFile(archive) as zipped:
            for name in zipped.namelist():
                with zipped.open(name) as source:
                    for row in csv.DictReader(TextIOWrapper(source, encoding="utf-8-sig")):
                        origin, destination = row["from_station_id"], row["to_station_id"]
                        if origin not in selected or destination not in selected:
                            continue
                        distance = float(row["road_m"])
                        if not math.isfinite(distance) or distance < 0 or (origin, destination) in edges:
                            raise ValueError(f"Invalid or duplicate directional distance {origin} → {destination}")
                        edges[origin, destination] = distance
    distances = [[0 if origin == destination else edges[origin, destination]
                  for destination in ids] for origin in ids]
    monthly = {}
    days = {}
    splits = {}
    for row in rows(folder / "02_region_ridership_7to9_monthly.csv"):
        month, station = row["month"], row["station_id"]
        if station not in selected:
            continue
        if row["has_source_record"] != "1":
            raise ValueError(f"Missing source ridership: {month}/{station}")
        if station in monthly.setdefault(month, {}):
            raise ValueError(f"Duplicate ridership row: {month}/{station}")
        monthly[month][station] = int(row["alight_07_09"])
        days[month] = int(row["calendar_days"])
        splits[month] = row["split"]
    if not monthly or any(set(values) != set(ids) for values in monthly.values()):
        raise ValueError("Monthly demand coverage does not match candidates")
    stops = [{"id": r["station_id"], "name": r["station_name"],
              "lat": float(r["latitude"]), "lon": float(r["longitude"]),
              "hub": HUBS.get(r["station_id"], "")} for r in candidates]
    data = dict(stops=stops, distances_m=distances,
                months={m: dict(split=splits[m], days=days[m],
                                alight=[monthly[m][i] for i in ids]) for m in sorted(monthly)})
    output.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Folder containing CSV files 01, 02 and road distance ZIP files 11")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("network.json"))
    args = parser.parse_args()
    prepare(args.source, args.out)
