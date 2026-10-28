"""Bounded, timestamp-matched review storage and snapshot export."""

from dataclasses import dataclass
from pathlib import Path
import json
import threading
import time
import uuid

import cv2
import numpy as np


PANELS = ('camera', 'yolo', 'lidar', 'bev')


@dataclass(frozen=True)
class ReviewFrame:
    timestamp_ns: int
    images: dict
    frame_ids: dict

    @property
    def size(self):
        return sum(len(data) for data in self.images.values())


class ReviewBuffer:
    """Thread-safe compressed storage; the paused selection is never evicted."""

    def __init__(self, memory_bytes, timeout_s=10.0, max_pending=30):
        if memory_bytes <= 0 or timeout_s <= 0 or max_pending <= 0:
            raise ValueError('Review buffer limits must be positive')
        self.limit = memory_bytes
        self.timeout_s = timeout_s
        self.max_pending = max_pending
        self.lock = threading.RLock()
        self.pending = {}
        self.frames = {}
        self.selected = None
        self.live = True
        self.waiting_next = False
        self.dropped = 0
        self.last_missing = ''

    def _expire(self, now):
        for stamp, item in list(self.pending.items()):
            if now - item['arrival'] > self.timeout_s:
                self.last_missing = ', '.join(p for p in PANELS if p not in item['images'])
                del self.pending[stamp]
                self.dropped += 1

    def _bytes(self):
        return (sum(f.size for f in self.frames.values())
                + sum(len(data) for item in self.pending.values()
                      for data in item['images'].values()))

    def add(self, timestamp, panel, encoded, frame_id='', now=None):
        if panel not in PANELS:
            raise ValueError(f'Unknown panel: {panel}')
        now = time.monotonic() if now is None else now
        with self.lock:
            self._expire(now)
            if timestamp in self.frames:
                previous = self.frames[timestamp]
                if previous.size - len(previous.images[panel]) + len(encoded) > self.limit:
                    self.dropped += 1
                    return
                # Late shoreline inference republishes an existing BEV frame.
                # Replace the immutable snapshot so paused views and exports update.
                self.pending[timestamp] = dict(arrival=now,
                    images=dict(previous.images), frame_ids=dict(previous.frame_ids))
                del self.frames[timestamp]
            item = self.pending.setdefault(timestamp, dict(arrival=now, images={}, frame_ids={}))
            item['images'][panel] = encoded
            item['frame_ids'][panel] = frame_id
            if sum(map(len, item['images'].values())) > self.limit:
                del self.pending[timestamp]
                self.dropped += 1
                return
            if all(p in item['images'] for p in PANELS):
                self.frames[timestamp] = ReviewFrame(timestamp, item['images'], item['frame_ids'])
                del self.pending[timestamp]
                if self.live:
                    self.selected = max(self.frames)
                elif self.waiting_next:
                    newer = sorted(t for t in self.frames if self.selected is None or t > self.selected)
                    if newer:
                        self.selected = newer[0]
                        self.waiting_next = False
            while len(self.pending) > self.max_pending:
                del self.pending[min(self.pending, key=lambda t: self.pending[t]['arrival'])]
                self.dropped += 1
            while self._bytes() > self.limit:
                candidates = sorted(t for t in self.frames if t != self.selected)
                if candidates:
                    del self.frames[candidates[0]]
                elif self.pending:
                    del self.pending[min(self.pending, key=lambda t: self.pending[t]['arrival'])]
                else:
                    break
                self.dropped += 1

    def toggle_pause(self):
        with self.lock:
            self.set_live(not self.live)

    def set_live(self, enabled=True):
        with self.lock:
            self.live = enabled
            self.waiting_next = False
            if enabled and self.frames:
                self.selected = max(self.frames)

    def step(self, direction):
        with self.lock:
            self.live = False
            self.waiting_next = False
            stamps = sorted(self.frames)
            if not stamps:
                self.waiting_next = direction > 0
                return
            if self.selected is None:
                self.selected = stamps[0]
                return
            candidates = [t for t in stamps if (t - self.selected) * direction > 0]
            if candidates:
                self.selected = min(candidates) if direction > 0 else max(candidates)
            else:
                self.waiting_next = direction > 0

    def view(self):
        with self.lock:
            self._expire(time.monotonic())
            stamps = sorted(self.frames)
            frame = self.frames.get(self.selected)
            status = dict(live=self.live, waiting_next=self.waiting_next,
                          count=len(stamps), index=stamps.index(self.selected) + 1 if frame else 0,
                          pending=len(self.pending), dropped=self.dropped,
                          memory_bytes=self._bytes(), last_missing=self.last_missing)
            return frame, status


def decode_panel(data):
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot decode review image')
    return image


def save_frame(frame, directory, topics, jpeg_quality):
    """Write the exact selected frame, all four panels and reproducible metadata."""
    destination = Path(directory).expanduser() / f'{frame.timestamp_ns}_{uuid.uuid4().hex[:8]}'
    destination.mkdir(parents=True, exist_ok=False)
    dimensions, tiles = {}, []
    for panel in PANELS:
        image = decode_panel(frame.images[panel])
        if not cv2.imwrite(str(destination / f'{panel}.png'), image):
            raise OSError(f'Failed to save {panel}')
        dimensions[panel] = list(image.shape[:2])
        tile = np.full((480, 640, 3), 25, dtype=np.uint8)
        factor = min(640 / image.shape[1], 450 / image.shape[0])
        resized = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        h, w = resized.shape[:2]
        tile[30:30+h, (640-w)//2:(640-w)//2+w] = resized
        cv2.putText(tile, f'{panel} | {frame.timestamp_ns}', (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (230, 230, 230), 1)
        tiles.append(tile)
    composite = np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))
    if not cv2.imwrite(str(destination / 'comparison.png'), composite):
        raise OSError('Failed to save comparison')
    metadata = dict(timestamp_ns=frame.timestamp_ns, topics=topics,
                    frame_ids=frame.frame_ids, dimensions_hw=dimensions,
                    buffer_encoding='JPEG', jpeg_quality=jpeg_quality,
                    note='PNG exports decode the compressed review buffer; JPEG compression is lossy.')
    (destination / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return destination
