"""Convert the team's candidate, road-distance and monthly ridership CSVs."""
import argparse
import csv
import json
import math
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


def prepare(folder, output):
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
    stops = [{"id": r["station_id"], "name": r["station_name"],
              "lat": float(r["latitude"]), "lon": float(r["longitude"]),
              "hub": HUBS.get(r["station_id"], "")} for r in candidates]
    data = dict(stops=stops, distances_m=distances,
                months={m: dict(split=splits[m], days=days[m],
                                alight=[monthly[m][i] for i in ids]) for m in sorted(monthly)})
    output.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Folder containing CSV files 02, 13 and 14")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("network.json"))
    args = parser.parse_args()
    prepare(args.source, args.out)
