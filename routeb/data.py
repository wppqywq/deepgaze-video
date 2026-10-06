"""Versioned video prompts over the existing audited materialized inputs."""
import copy
import json
from pathlib import Path

from routeb.evaluation import Inputs

PROMPT_VERSION = 'video-next-fixation-v2'


def frame_paths(sample, exported):
    frames = [frame for frame in sample['input']['frames'] if frame is not None]
    if len(frames) != len(exported['images']):
        raise ValueError('Frame/image count mismatch')
    return {frame['index']: path for frame, path in zip(frames, exported['images'])}


def record(sample, variant, paths):
    if variant not in ('d1', 'd4', 'r4'):
        raise ValueError('Unknown visual variant')
    frames = sample['input']['frames']
    if len(frames) != 4 or frames[-1] is None:
        raise ValueError('Four frame slots and a current frame are required')
    lines = [
        'Predict the next fixation location during video viewing.',
        'Times are seconds relative to the prediction cutoff, t=0.',
        'Coordinates use a 100 x 100 screen grid including black bars; x rightward, y downward.',
        'Video:',
    ]
    images = []
    for slot in ([3] if variant == 'd1' else range(4)):
        frame = frames[slot]
        if frame is None:
            lines.append(f'Frame {slot + 1}: missing.')
        else:
            if frame['relative_seconds'] > 0:
                raise ValueError('Future frame in input')
            content = frames[-1] if variant == 'r4' else frame
            images.append(str(paths[content['index']]))
            lines.append(f"Frame {slot + 1}, t={frame['relative_seconds']:.6f}: <image>")
    lines.append('Completed fixations, oldest to newest:')
    for event in sample['input']['history']:
        if event is None:
            lines.append('missing')
            continue
        onset, offset = event['start_relative_seconds'], event['end_relative_seconds']
        if not onset <= offset <= 0:
            raise ValueError('Incomplete or reversed fixation in history')
        x, y = event['cell']
        lines.append(f'onset={onset:.6f}, offset={offset:.6f}, xy=({x:02d}, {y:02d})')
    lines.append('Output only the next coordinate: [(XX, YY)], using two digits 00-99 for each coordinate.')
    x, y = sample['target']['cell']
    return {'images': images, 'conversations': [
        {'from': 'human', 'value': '\n'.join(lines)},
        {'from': 'gpt', 'value': f'[({x:02d}, {y:02d})]'}]}


class VideoInputs(Inputs):
    """Keep immutable input audits; construct the new template in memory only."""

    def rows(self, variant, split):
        original = super().rows('d4', split)
        return [record(self.samples[sample_id], variant,
                       frame_paths(self.samples[sample_id], exported))
                for sample_id, exported in zip(self.ids[split], original)]

    def baseline_rows(self, variant):
        """Use the same frame slots/content for B1; preserve all original metadata."""
        rows, paths = [], {}
        for split in ('train', 'validation'):
            originals = super().rows('d4', split)
            for sample_id, exported in zip(self.ids[split], originals):
                source = self.samples[sample_id]
                for index, path in frame_paths(source, exported).items():
                    path = Path(path)
                    key = (source['video'], index)
                    if key in paths and paths[key] != path:
                        raise ValueError('Conflicting image paths for the same video frame')
                    paths[key] = path
                row = copy.deepcopy(source)
                frames = row['input']['frames']
                if variant == 'd1':
                    frames[:3] = [None] * 3
                elif variant == 'r4':
                    for frame in frames:
                        if frame is not None:
                            frame['index'] = frames[-1]['index']
                elif variant != 'd4':
                    raise ValueError('Unknown baseline variant')
                rows.append(row)
        return rows, paths


class FovlandingInputs:
    """Read paired development inputs verified by the full processor scan."""

    def __init__(self, outputs, scan_report):
        import numpy as np
        from fovlanding import read_jsonl
        from gpu_train import file_hash, require

        self.outputs = Path(outputs)
        self.scan = json.loads(Path(scan_report).read_text())
        require(self.scan['status'] == 'Full fovlanding processor scan passed', 'Full scan required')
        for name, expected in self.scan['data_sha256'].items():
            require(file_hash(self.outputs / name) == expected, 'Audited input changed: ' + name)
        self.targets = list(read_jsonl(self.outputs / 'targets.jsonl'))
        self.samples = {}
        self.ids = {'train': [], 'validation': []}
        films = {'train': set(), 'validation': set()}
        for target in self.targets:
            key, audit = target['target_id'], target['audit']
            require(key not in self.samples and audit['split'] in self.ids, 'Duplicate target or invalid split')
            self.ids[audit['split']].append(key)
            films[audit['split']].add(audit['source_film_id'])
            self.samples[key] = {'target': {'cell': target['target']['xy_grid']},
                'video': audit['video_id'], 'source_group': audit['source_film_id'],
                'observer': audit['observer_id']}
        require(not films['train'] & films['validation'], 'Source-film leakage across splits')
        require({key: len(value) for key, value in self.ids.items()} == self.scan['split_counts'], 'Target population differs')
        frame_manifest = json.loads((self.outputs / 'frames.json').read_text())
        self.paths = {}
        for video, frames in frame_manifest.items():
            self.paths[video] = {}
            for frame in frames:
                path = self.outputs / frame['path']
                require(file_hash(path) == frame['sha256'], 'Decoded frame changed')
                self.paths[video][frame['index']] = str(path)
        self.prior = np.load(self.outputs / 'training-prior.npy', allow_pickle=False)
        require(self.prior.shape == (100, 100) and bool(np.isfinite(self.prior).all()) and
                bool((self.prior > 0).all()) and abs(float(self.prior.sum()) - 1) < 1e-10, 'Invalid training-only prior')
        self.provenance = {'scan_sha256': file_hash(scan_report), 'data_sha256': self.scan['data_sha256'],
                           'scope': 'development_assumptions', 'assumptions': self.scan['assumptions']}

    def rows(self, variant, split):
        from fovlanding import serialize_target
        if variant not in ('d1', 'd4', 'r4') or split not in self.ids:
            raise ValueError('Unknown variant or split')
        return [serialize_target(target, variant.upper(), self.paths[target['audit']['video_id']])
                for target in self.targets if target['audit']['split'] == split]
