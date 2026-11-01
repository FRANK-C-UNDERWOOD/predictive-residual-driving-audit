"""Outcome-independent annotation factors for a driving-perception audit."""
from collections import defaultdict
import math
import numpy as np

def annotation_factors(grid, labels, cfg):
    grid = list(map(int, grid))
    if len(grid) < 2 or any(b <= a for a, b in zip(grid, grid[1:])):
        raise ValueError('Strictly ordered native query grid required')
    index = {t: i for i, t in enumerate(grid)}
    median_gap = float(np.median(np.diff(grid)))
    diagonal = math.hypot(cfg['width'], cfg['height'])
    previous, counts, first_seen, seen = {}, defaultdict(int), {}, set()
    result = {}
    for row in sorted(labels, key=lambda r: (int(r['t']), int(r['track_id']))):
        if str(int(row['class_id'])) not in cfg['class_map']:
            continue
        t, identity = int(row['t']), int(row['track_id'])
        if t not in index:
            raise ValueError('Annotation timestamp not on native query grid')
        key = (t, identity)
        if key in seen:
            raise ValueError('Duplicate identity at one query')
        seen.add(key)
        # Preserve the parent's float32 x+w/y+h convention before clipping.
        box = [max(0., float(row['x'])), max(0., float(row['y'])),
               min(float(cfg['width']), float(row['x'] + row['w'])),
               min(float(cfg['height']), float(row['y'] + row['h']))]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        first_seen.setdefault(identity, t)
        counts[identity] += 1
        left_censored = first_seen[identity] == grid[0]
        stage = ('left_censored' if left_censored else 'early'
                 if counts[identity] <= cfg['early_annotated_appearances'] else 'established')
        p = previous.get(identity)
        valid = (p is not None and p['grid_index'] + 1 == index[t]
                 and t - p['time_us'] <= cfg['gap_factor'] * median_gap)
        speed = (math.dist(center, p['center']) / ((t - p['time_us']) / 1e6) / diagonal
                 if valid else None)
        motion = ('motion_unknown' if speed is None else 'motion_low'
                  if speed <= cfg['motion_low_max_diagonal_per_second'] else 'motion_high'
                  if speed > cfg['motion_high_min_diagonal_per_second'] else 'motion_middle')
        fraction = (box[2] - box[0]) * (box[3] - box[1]) / (cfg['width'] * cfg['height'])
        size = 'small' if fraction <= cfg['small_area_fraction'] else 'large'
        cls = cfg['class_map'][str(int(row['class_id']))]
        record = dict(time_us=t, identity=identity, grid_index=index[t], box=box,
                      cls=cls, area_fraction=fraction, size=size, stage=stage,
                      appearance_number=counts[identity], first_annotated_us=first_seen[identity],
                      left_censored=left_censored, motion=motion,
                      motion_diagonal_per_second=speed)
        record['groups'] = ['all', size, stage, motion, 'car' if cls == 0 else 'pedestrian']
        if stage != 'left_censored':
            record['groups'].append(size + '_' + stage)
        result[key] = record
        previous[identity] = dict(time_us=t, grid_index=index[t], center=center)
    return result, median_gap

def group_identities(rows, groups):
    result = {g: set() for g in groups}
    for row in rows:
        for group in row['groups']:
            result[group].add(row['identity'])
    return result
