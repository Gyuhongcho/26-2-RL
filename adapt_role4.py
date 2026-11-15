"""Convert the team's role4 bundle into trainer input."""
import argparse
import json
from pathlib import Path


def convert(bundle, diagnostic=False):
    stops = bundle["stops"]
    if bundle["ids"] != [s["station_id"] for s in stops]:
        raise ValueError("Bundle indices do not match stop IDs")
    indices = [
        i for i, stop in enumerate(stops)
        if stop["complete_train"] and stop["candidate_eligible_by_policy"]
        and (not diagnostic or stop["diagnostic_subset_eligible"])
    ]
    if not indices:
        raise ValueError("No eligible stops")
    return {
        "source": bundle["schema_version"],
        "requires_osrm": not diagnostic,
        "stops": [
            {"id": stops[i]["station_id"], "name": stops[i]["station_name"],
             "lat": stops[i]["latitude"], "lon": stops[i]["longitude"],
             "hub": stops[i]["hub_id"] or "", "hub_name": stops[i]["hub_name"],
             "demand_daily_train": stops[i]["demand_daily_train"],
             "demand_score_train": stops[i]["demand_score_train"]}
            for i in indices
        ],
        "months": {
            month: {"days": record["days"], "split": record["split"],
                    "alight": [record["alight"][i] for i in indices]}
            for month, record in bundle["months"].items()
        },
        "distances_m": [[bundle["distances_m"][i][j] for j in indices] for i in indices]
        if diagnostic else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    data = convert(json.loads(args.bundle.read_text(encoding="utf-8")), args.diagnostic)
    args.out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
