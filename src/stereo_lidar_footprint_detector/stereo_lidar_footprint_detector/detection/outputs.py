"""JSON output shared by the offline runner and ROS adapter."""

import json
from pathlib import Path


def obstacle_collection(obstacles, timestamp_ns, frame_id, frame_index=None):
    collection = dict(obstacles)
    collection.update(timestamp_ns=int(timestamp_ns), frame_id=str(frame_id), units="m")
    if frame_index is not None:
        collection["frame_index"] = int(frame_index)
    return collection


def write_obstacles(path, collection):
    # Serialize first so nonfinite data cannot leave a partially written file.
    payload = json.dumps(collection, indent=2, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n")
