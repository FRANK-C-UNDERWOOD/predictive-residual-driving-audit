"""Raw GEN1 input adapter for the pinned SpikeFPN convention.

No DSEC geometry or calibration is reused. GT is never used to encode events.
The upstream's native clipped/integer labels and exact timestamp windows are
kept separate from optional Prophesee benchmark filtering.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2

WIDTH, HEIGHT, INPUT_SIZE = 304, 240, 256
BIN_US, BINS = 20000, 9
WINDOW_US = BIN_US * BINS
DAT_DTYPE = np.dtype([('t', '<u4'), ('packed', '<u4')])


def dat_header(path):
    path = Path(path)
    height = width = None
    comments = []
    with path.open('rb') as f:
        while True:
            start = f.tell()
            prefix = f.read(2)
            f.seek(start)
            if prefix != b'% ':
                break
            line = f.readline(65537)
            if len(line) > 65536 or not line.endswith(b'\n'):
                raise ValueError('Malformed DAT comment header')
            comments.append(line.decode('latin-1').strip())
            words = line.split()
            if len(words) >= 3 and words[1] == b'Height':
                height = int(words[2])
            if len(words) >= 3 and words[1] == b'Width':
                width = int(words[2])
        if comments:
            data = f.read(2)
            if len(data) != 2:
                raise ValueError('Missing DAT type/size')
            event_type, event_size = data
        else:
            event_type, event_size = 0, 8
        offset = f.tell()
    if event_size != 8 or (path.stat().st_size - offset) % 8:
        raise ValueError('Expected complete 8-byte DAT records')
    if height not in (None, HEIGHT) or width not in (None, WIDTH):
        raise ValueError('Not native GEN1 304x240 geometry')
    return dict(offset=offset, event_type=event_type, event_size=event_size,
                height=height, width=width, comments=comments,
                events=(path.stat().st_size-offset)//8)


class Gen1Events:
    def __init__(self, path):
        self.path = Path(path)
        self.header = dat_header(path)
        n = self.header['events']
        self.records = (np.memmap(path, mode='r', dtype=DAT_DTYPE,
                                 offset=self.header['offset'], shape=(n,))
                        if n else np.empty(0, dtype=DAT_DTYPE))
        self.validation = self.validate()

    def validate(self, chunk_size=1_000_000):
        previous = None
        for start in range(0, len(self.records), chunk_size):
            part = self.records[start:start+chunk_size]
            t, p = part['t'], part['packed']
            if (previous is not None and int(t[0]) < previous) or np.any(t[1:] < t[:-1]):
                raise ValueError('Nonmonotone/wrapped DAT timestamps; refuse binary search')
            if np.any((p & 16383) >= WIDTH) or np.any(((p >> 14) & 16383) >= HEIGHT):
                raise ValueError('Event coordinate outside GEN1 sensor')
            previous = int(t[-1])
        return dict(events=len(self.records), monotone=True,
                    first_us=int(self.records['t'][0]) if len(self.records) else None,
                    last_us=int(self.records['t'][-1]) if len(self.records) else None)

    def bounds(self, query_us):
        query_us = int(query_us)
        if query_us < 0:
            raise ValueError('Negative query timestamp')
        start = query_us-WINDOW_US+1
        t = self.records['t']
        lo = int(np.searchsorted(t, max(0, start), side='left'))
        hi = int(np.searchsorted(t, query_us, side='right'))
        return lo, hi, start

    def read_window(self, query_us):
        lo, hi, start = self.bounds(query_us)
        part = self.records[lo:hi]
        p = part['packed']
        return np.column_stack(((p & 16383).astype(np.int64),
                                ((p >> 14) & 16383).astype(np.int64),
                                part['t'].astype(np.int64),
                                2*((p >> 28) & 1).astype(np.int64)-1)), start

    def signed_frames(self, query_us):
        events, start = self.read_window(query_us)
        return signed_frames(events, start), dict(query_us=int(query_us),
                start_inclusive_us=start, events=len(events),
                pre_sensor_history=start < 0)

    def close(self):
        if isinstance(self.records, np.memmap):
            self.records._mmap.close()


def signed_frames(events, start_inclusive_us):
    events = np.asarray(events)
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError('Expected x,y,t,signed-polarity events')
    if not np.isfinite(events).all() or np.any(events != np.floor(events)):
        raise ValueError('Events must have finite integer coordinates/timestamps/polarity')
    x, y, t, p = events.astype(np.int64).T
    b = (t-int(start_inclusive_us))//BIN_US
    if (np.any(x < 0) or np.any(x >= WIDTH) or np.any(y < 0) or np.any(y >= HEIGHT)
            or np.any(b < 0) or np.any(b >= BINS) or not np.isin(p, [-1, 1]).all()):
        raise ValueError('Event outside declared support')
    counts = np.zeros((BINS, HEIGHT, WIDTH), dtype=np.int64)
    np.add.at(counts, (b, y, x), p)
    return np.sign(counts).astype(np.int8).reshape(3, 3, HEIGHT, WIDTH)


def encode_frames(frames):
    frames = np.asarray(frames)
    if frames.shape != (3, 3, HEIGHT, WIDTH) or not np.isin(frames, [-1, 0, 1]).all():
        raise ValueError('Expected 3x3 signed native GEN1 frames')
    ratio = INPUT_SIZE/WIDTH
    resized = cv2.resize(frames.reshape(9, HEIGHT, WIDTH).astype(np.float32).transpose(1,2,0),
                         (INPUT_SIZE, round(HEIGHT*ratio)), interpolation=cv2.INTER_NEAREST)
    # 240*256/304 rounds to 202, leaving 27 rows above and below.
    image = cv2.copyMakeBorder(resized, 27, 27, 0, 0, cv2.BORDER_CONSTANT, value=0)
    return np.ascontiguousarray(image.transpose(2,0,1).reshape(1,3,3,256,256))


def restore_boxes(normalized_xyxy):
    boxes = np.asarray(normalized_xyxy)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        raise ValueError('Expected finite normalized xyxy boxes')
    # test_gad.py multiplies by an int64 shape array, promoting float32
    # predictions before scaling; multiplying by scalar 256 rounds earlier.
    return (boxes*np.array([[INPUT_SIZE]*4],dtype=np.int64)-np.array([0.,27.,0.,27.]))/(INPUT_SIZE/WIDTH)


def load_labels(path):
    labels = np.load(path, allow_pickle=False)
    names = set(labels.dtype.names or ())
    if not {'x','y','w','h','class_id'} <= names:
        raise ValueError('Missing GEN1 structured label fields')
    time_key = 't' if 't' in names else 'ts' if 'ts' in names else None
    if time_key is None:
        raise ValueError('Missing annotation timestamp')
    if labels.ndim != 1 or np.any(labels[time_key][1:] < labels[time_key][:-1]):
        raise ValueError('Labels must be one-dimensional and timestamp ordered')
    for name in ['x','y','w','h',time_key]:
        if not np.isfinite(labels[name]).all():
            raise ValueError('Nonfinite labels')
    if np.any(labels['w'] < 0) or np.any(labels['h'] < 0) or np.any(labels[time_key] < 0):
        raise ValueError('Negative box extent or timestamp')
    if not np.isin(labels['class_id'], [0,1]).all():
        raise ValueError('Unexpected class: GEN1 car=0, pedestrian=1')
    return labels, time_key


def upstream_labels(rows):
    """Original preprocessing clips to width/height-1 and truncates to int64."""
    a = np.column_stack([rows[k] for k in ['class_id','x','y','w','h']]).astype(np.float64)
    x1 = np.clip(a[:,1], 0, WIDTH-1)
    y1 = np.clip(a[:,2], 0, HEIGHT-1)
    x2 = np.clip(a[:,1]+a[:,3], 0, WIDTH-1)
    y2 = np.clip(a[:,2]+a[:,4], 0, HEIGHT-1)
    return np.column_stack([a[:,0],x1,y1,x2-x1,y2-y1]).astype(np.int64)


def sequence_pairs(root, split='val'):
    """Return complete event/label pairs; never silently skip an orphan."""
    folder = Path(root)/split
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    events = {p.name[:-7]: p for p in folder.glob('*_td.dat')}
    labels = {p.name[:-9]: p for p in folder.glob('*_bbox.npy')}
    if not events:
        raise FileNotFoundError('No raw GEN1 event files in '+str(folder))
    if events.keys() != labels.keys():
        raise ValueError('Unpaired GEN1 files: '+str(sorted(events.keys() ^ labels.keys())))
    return [(key, events[key], labels[key]) for key in sorted(events)]
