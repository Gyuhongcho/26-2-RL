"""Convert the team's candidate, road-distance and monthly ridership CSVs."""
import argparse
import csv
import json
import math
from io import TextIOWrapper
from zipfile import ZipFile
from pathlib import Path

HUBS = {
    "112900230": "신촌", "113900230": "신촌", "113000137": "신촌", "113000456": "신촌",
    "113900229": "이대", "113000118": "이대",
    "113900302": "아현",
    "112000031": "연대앞", "112900188": "연대앞",
}


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def prepare(folder, output, include_mapo=False):
    candidates = rows(folder / "13_candidate_pool_v2_PROPOSAL.csv")
    ids = [r["station_id"] for r in candidates]
    if len(ids) != len(set(ids)) or not set(HUBS) <= set(ids):
        raise ValueError("Candidate IDs must be unique and include the configured hub stops")
    matrix = rows(folder / "14_candidate_v2_road_distance_matrix_m.csv")
    by_origin = {next(iter(row.values())): row for row in matrix}
    if len(matrix) != len(by_origin) or set(by_origin) != set(ids):
        raise ValueError("Distance matrix origin IDs do not match candidates")
    distances = [[float(by_origin[i][j]) for j in ids] for i in ids]
    for i, row in enumerate(distances):
        for j, distance in enumerate(row):
            if not math.isfinite(distance) or distance < 0 or (i == j and distance != 0):
                raise ValueError(f"Invalid directional distance {ids[i]} → {ids[j]}")
    monthly = {}
    days = {}
    splits = {}
    for row in rows(folder / "02_region_ridership_7to9_monthly.csv"):
        month, station = row["month"], row["station_id"]
        if station not in by_origin:
            continue
        if station in monthly.setdefault(month, {}):
            raise ValueError(f"Duplicate ridership row: {month}/{station}")
        monthly[month][station] = int(row["alight_07_09"])
        days[month] = int(row["calendar_days"])
        splits[month] = row["split"]
    if not monthly or any(set(values) != set(ids) for values in monthly.values()):
        raise ValueError("Monthly demand coverage does not match candidates")
    baseline_ids = list(ids)
    if include_mapo:
        original = set(ids)
        stations = rows(folder / "01_region_stations_clean.csv")
        ridership = rows(folder / "02_region_ridership_7to9_monthly.csv")
        by_station = {}
        for row in ridership:
            by_station.setdefault(row["station_id"], []).append(row)

        for stop in stations:
            station_id = stop["station_id"]
            if station_id in original or not station_id.startswith(("113000", "113900")):
                continue
            records = by_station.get(station_id, [])
            if len(records) != len(monthly) or {r["month"] for r in records} != set(monthly):
                raise ValueError(f"Incomplete monthly coverage for Mapo stop {station_id}")
            ids.append(station_id)
            candidates.append(stop)
            for row in records:
                monthly[row["month"]][station_id] = int(row["alight_07_09"])

        selected = set(ids)
        edges = {(i, j): distances[x][y] for x, i in enumerate(baseline_ids)
                 for y, j in enumerate(baseline_ids)}
        for archive in sorted(folder.glob("11_road_distance_part*.zip")):
            with ZipFile(archive) as zipped:
                for name in zipped.namelist():
                    with zipped.open(name) as source:
                        for row in csv.DictReader(TextIOWrapper(source, encoding="utf-8-sig")):
                            i, j = row["from_station_id"], row["to_station_id"]
                            if i in selected and j in selected and (i not in original or j not in original):
                                distance = float(row["road_m"])
                                if not math.isfinite(distance) or distance < 0:
                                    raise ValueError(f"Invalid road distance {i} → {j}")
                                edges[i, j] = distance
        distances = [[0 if i == j else edges[i, j] for j in ids] for i in ids]

    stops = [{"id": r["station_id"], "name": r["station_name"],
              "lat": float(r["latitude"]), "lon": float(r["longitude"]),
              "hub": HUBS.get(r["station_id"], "")} for r in candidates]
    data = dict(stops=stops, distances_m=distances,
                months={m: dict(split=splits[m], days=days[m],
                                alight=[monthly[m][i] for i in ids]) for m in sorted(monthly)})
    if include_mapo:
        data["baseline_stop_ids"] = baseline_ids
    output.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Folder containing CSV files 02, 13 and 14")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("network.json"))
    parser.add_argument("--include-mapo", action="store_true")
    args = parser.parse_args()
    prepare(args.source, args.out, args.include_mapo)
