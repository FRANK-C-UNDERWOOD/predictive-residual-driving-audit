"""Causal raw-event reader for the downloaded DSEC development sequences.

The signed-stack adapter is explicit and is not a claim of GEN1 reproduction.
No labels are used to compute event values or an inference-time gate score.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np


class DSECPilotSequence:
    def __init__(self, root: str | Path, sequence: str):
        import h5py
        import hdf5plugin  # noqa: F401 -- registers the publisher's Blosc filter

        self.sequence = sequence
        self.path = Path(root) / "train" / sequence
        self.event_path = self.path / "events/left/events.h5"
        self.label_path = self.path / "object_detections/left/tracks.npy"
        self.file = h5py.File(self.event_path, "r")
        self.events = self.file["events"]
        self.offset_us = int(np.asarray(self.file["t_offset"]).item())
        self.labels = np.load(self.label_path, allow_pickle=False)
        self.annotation_times = np.unique(self.labels["t"]).astype(np.int64)
        self.ms_to_idx = np.asarray(self.file["ms_to_idx"], dtype=np.int64)
        self.height, self.width = 480, 640
        self.event_count = len(self.events["t"])
        self.first_local_us = int(self.events["t"][0])
        self.last_local_us = int(self.events["t"][-1])

    def close(self):
        self.file.close()

    def info(self):
        return dict(
            sequence=self.sequence, event_path=str(self.event_path),
            label_path=str(self.label_path), offset_us=self.offset_us,
            event_count=self.event_count, label_boxes=len(self.labels),
            annotation_times=len(self.annotation_times),
            event_range_absolute_us=[self.offset_us + self.first_local_us,
                                     self.offset_us + self.last_local_us],
            annotation_range_absolute_us=[int(self.annotation_times[0]),
                                          int(self.annotation_times[-1])],
            class_ids=np.unique(self.labels["class_id"]).tolist(),
            track_ids=len(np.unique(self.labels["track_id"])),
            convention="Raw distorted event coordinates; labels already in event view",
        )

    def read_window(self, end_absolute_us: int, duration_us: int = 180000):
        """Return events in (end-duration, end], never using later observations."""
        end = int(end_absolute_us) - self.offset_us
        start = end - int(duration_us)
        if start < self.first_local_us or end > self.last_local_us:
            raise ValueError("Window does not have complete recorded event support")
        lo_ms = max(0, start // 1000)
        hi_ms = min(len(self.ms_to_idx) - 1, end // 1000 + 1)
        lo = int(self.ms_to_idx[lo_ms]) if lo_ms < len(self.ms_to_idx) else self.event_count
        hi = int(self.ms_to_idx[hi_ms])
        if hi_ms == len(self.ms_to_idx) - 1 and end >= hi_ms * 1000:
            hi = self.event_count
        times = np.asarray(self.events["t"][lo:hi], dtype=np.int64)
        left, right = np.searchsorted(times, [start, end], side="right")
        lower, upper = lo + int(left), lo + int(right)
        values = {key: np.asarray(self.events[key][lower:upper]) for key in ("x", "y", "p", "t")}
        if len(values["t"]) and not (
            int(values["t"].min()) > start and int(values["t"].max()) <= end
        ):
            raise ValueError("Causal event-window boundary violated")
        return values, start

    def signed_stack(self, end_absolute_us: int, size=256, steps=3, channels=3, bin_us=20000,
                     return_counts=False):
        import cv2

        duration = steps * channels * bin_us
        events, start = self.read_window(end_absolute_us, duration)
        x = events["x"].astype(np.int64)
        y = events["y"].astype(np.int64)
        if len(x) and (x.min() < 0 or x.max() >= self.width or y.min() < 0 or y.max() >= self.height):
            raise ValueError("Unexpected DSEC pixel geometry")
        polarity = events["p"].astype(np.int16)
        if not np.isin(polarity, [0, 1]).all():
            raise ValueError("Expected DSEC 0/1 polarity")
        bins = (events["t"].astype(np.int64) - start - 1) // bin_us
        frames = np.zeros((steps * channels, self.height, self.width), dtype=np.int32)
        np.add.at(frames, (bins, y, x), 2 * polarity - 1)
        frames = np.sign(frames).astype(np.float32)
        ratio = min(size / self.height, size / self.width)
        resized_h, resized_w = int(round(ratio * self.height)), int(round(ratio * self.width))
        pad_h, pad_w = (size - resized_h) / 2, (size - resized_w) / 2
        top, bottom = int(round(pad_h - .1)), int(round(pad_h + .1))
        left, right = int(round(pad_w - .1)), int(round(pad_w + .1))
        # Match the upstream Resize_frame nearest-neighbor letterbox convention.
        resized = np.stack([
            np.pad(cv2.resize(f, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST),
                   ((top, bottom), (left, right))) for f in frames
        ])
        tensor = resized.reshape(1, steps, channels, size, size)
        record = dict(sequence=self.sequence, end_absolute_us=int(end_absolute_us),
                      start_absolute_us=int(end_absolute_us) - duration,
                      input_shape=list(tensor.shape), raw_events=len(x),
                      nonzero_input=int(np.count_nonzero(tensor)), bin_us=bin_us,
                      native_size=[self.height, self.width], resize_ratio=ratio,
                      pad_top_left=[top, left],
                      encoding="chronological signed polarity sum; upstream-style nearest letterbox")
        if return_counts:
            gx = np.clip(((x * ratio + left) / 8).astype(np.int64), 0, size//8-1)
            gy = np.clip(((y * ratio + top) / 8).astype(np.int64), 0, size//8-1)
            side = size // 8
            counts8 = np.bincount(gy * side + gx, minlength=side*side).reshape(side, side)
            counts = {8: counts8}
            for stride in (16, 32):
                factor = stride // 8
                counts[stride] = counts8.reshape(side//factor, factor, side//factor, factor).sum((1, 3))
            return tensor, record, counts
        return tensor, record
