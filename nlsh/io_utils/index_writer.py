import json
from typing import Dict, List

def write_partitions_file(blocks: List[int], path: str):
    with open(path, "w") as f:
        for node_id, part in enumerate(blocks):
            f.write(f"{node_id},{part}\n")

def write_inverted_csv(parts_map: Dict[int, List[int]], path: str):
    with open(path, "w") as f:
        for part in sorted(parts_map.keys()):
            nodes = parts_map[part]
            f.write(",".join([str(part)] + [str(n) for n in nodes]) + "\n")

def write_meta(meta: Dict, path: str):
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
