"""Fetch road geometries for the winning routes of a completed comparison."""
import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROUTER = "https://routing.openstreetmap.de/routed-car/route/v1/driving"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


class LocalRouter:
    """Cache exact ordered waypoint routes shared by all comparison methods."""

    def __init__(self, endpoint, stops):
        from functools import lru_cache
        from urllib.parse import urlsplit
        url = urlsplit(endpoint)
        if url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Use a local OSRM server, for example http://127.0.0.1:5000")
        self.endpoint = endpoint.rstrip("/")
        self.stops = stops
        self.route = lru_cache(maxsize=16000)(self._route)

    def _route(self, path, geometry):
        points = ";".join(f"{self.stops[i]['lon']:.8f},{self.stops[i]['lat']:.8f}" for i in path)
        overview = "full" if geometry else "false"
        url = (f"{self.endpoint}/route/v1/driving/{points}?overview={overview}&steps={str(geometry).lower()}"
               "&geometries=geojson&continue_straight=true")
        from urllib.error import HTTPError
        try:
            with urlopen(url, timeout=15) as response:
                result = json.load(response)
        except HTTPError as error:
            if error.code != 400:
                raise
            result = json.load(error)
        if result.get("code") in {"NoRoute", "NoSegment"}:
            return None
        if result.get("code") != "Ok" or not result.get("routes"):
            raise RuntimeError(f"Local OSRM returned {result.get('code')}")
        return result["routes"][0]


def fetch_comparison(folder):
    manifest = load(folder / "comparison.json")
    methods = manifest.get("methods", ["q-learning", "monte-carlo", "sarsa"])
    if manifest["stage"] != len(methods) + 1:
        raise ValueError("Wait for the comparison to finish before routing its winners.")

    output = {}
    previous = {}
    target = folder / "road_geometry.json"
    if target.is_file():
        previous = load(target)
    last_request = 0
    for method in methods:
        trace = load(folder / method / "trace.json")
        winner = trace["overall_best"]
        stops = [trace["stops"][i] for i in winner["path"]]
        coordinates = [[stop["lon"], stop["lat"]] for stop in stops]
        cached = previous.get(method, {})
        if cached.get("waypoints") == coordinates:
            output[method] = cached
            continue

        encoded = ";".join(f"{lon:.8f},{lat:.8f}" for lon, lat in coordinates)
        request = Request(
            f"{ROUTER}/{encoded}?overview=full&geometries=geojson&steps=false",
            headers={"User-Agent": "Yonsei-Team8-Bus-Route-Study/1.0 (github.com/Gyuhongcho/26-2-RL)"},
        )
        time.sleep(max(0, 1.1 - (time.monotonic() - last_request)))
        last_request = time.monotonic()
        with urlopen(request, timeout=25) as response:
            result = json.load(response)
        if result.get("code") != "Ok" or not result.get("routes"):
            raise ValueError(f"{method}: OSRM returned {result.get('code')}")
        road = result["routes"][0]
        geometry = road["geometry"]
        if geometry["type"] != "LineString" or len(geometry["coordinates"]) < 2:
            raise ValueError(f"{method}: expected a road LineString")
        output[method] = {
            "waypoints": coordinates,
            "geometry": geometry,
            "distance_m": road["distance"],
            "max_snap_m": max(waypoint["distance"] for waypoint in result["waypoints"]),
        }
        print(f"{method}: model {winner['distance_m']:.0f} m, OSRM {road['distance']:.0f} m", flush=True)

    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_folder", type=Path)
    fetch_comparison(parser.parse_args().run_folder)
