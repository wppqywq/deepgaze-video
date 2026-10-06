"""Build causal fovlanding-v1 targets from standardized trial records."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from math import floor, isclose, isfinite
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REQUIRED_PROVENANCE = (
    "project_git_commit",
    "upstream_git_commit",
    "dataset_manifest_sha256",
    "split_manifest_sha256",
    "detector_version",
    "detector_parameters_file",
    "audited_trial_metadata_file",
)
REQUIRED_PROTOCOL = {
    "name": "fovlanding-v1",
    "quantization": "floor100-v1",
    "event_rule": "merge-fixa-purs-v1",
    "task": "free_viewing",
    "cutoff_rule": "next_saccade_onset",
    "gaze_input_rule": "sample_time_strictly_less_than_cutoff",
    "video_input_rule": "frame_pts_less_than_or_equal_to_cutoff",
    "target_rule": "first_valid_raw_sample_at_next_foveation_start",
}
OUTPUT_TARGETS = "targets.jsonl"
OUTPUT_SUMMARY = "build-summary.json"


@dataclass(frozen=True)
class Episode:
    start_sample: int
    stop_sample: int
    start_ms: float
    end_ms: float
    event_ids: tuple[int, ...]
    labels: tuple[str, ...]


def load_config(path: Path, trials_path: Path | None = None) -> dict[str, Any]:
    config = json.loads(path.read_text())
    preflight_config(config)
    audit_path = config['provenance'].get('data_validation_file')
    if not audit_path:
        raise ValueError('Missing data_validation_file; formal builds require a verified data audit.')
    audit = json.loads(Path(audit_path).read_text())
    scope = config['provenance'].get('validation_scope', 'formal')
    if scope == 'development_assumptions':
        if (audit.get('development_pilot', {}).get('eligible') is not True
                or config['provenance'].get('assumptions') != audit.get('assumptions')):
            raise ValueError('Development input lacks a matching empirical audit and assumptions.')
    elif scope != 'formal' or audit.get('formal_target_build', {}).get('valid_now') is not True:
        raise ValueError('Data audit blocks formal target construction.')
    for key in ('dataset_manifest_sha256', 'split_manifest_sha256'):
        if audit.get(key) != config['provenance'][key]:
            raise ValueError('Data audit provenance differs: ' + key)
    if trials_path is not None:
        digest = hashlib.sha256()
        with trials_path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
                digest.update(chunk)
        if digest.hexdigest() != audit.get('standardized_trials_sha256'):
            raise ValueError('Trial input differs from the verified data audit.')
    return config


def preflight_config(config: Mapping[str, Any]) -> None:
    protocol = config.get("protocol", {})
    mismatched = [
        key for key, expected in REQUIRED_PROTOCOL.items()
        if protocol.get(key) != expected
    ]
    if protocol.get("target_grid") != [100, 100]:
        mismatched.append("target_grid")
    if mismatched:
        raise ValueError("Invalid protocol settings: " + ", ".join(mismatched))
    provenance = config.get("provenance", {})
    missing = [key for key in REQUIRED_PROVENANCE if not provenance.get(key)]
    if missing:
        raise ValueError("Missing required provenance: " + ", ".join(missing))
    for key in ("dataset_manifest_sha256", "split_manifest_sha256"):
        value = str(provenance[key])
        if len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value):
            raise ValueError(f"{key} must be a SHA256 digest.")
    for key in ("project_git_commit", "upstream_git_commit"):
        value = str(provenance[key])
        if len(value) not in (40, 64) or any(
            character not in "0123456789abcdefABCDEF" for character in value
        ):
            raise ValueError(f"{key} must be a full commit hash.")
    if not isinstance(provenance.get("final_test_accessed"), bool):
        raise ValueError("final_test_accessed must be true or false.")
    settings = config.get("input", {})
    if settings.get("frame_count") != 4 or settings.get("history_window_ms") != 1000:
        raise ValueError("fovlanding-v1 requires four frame slots spanning one second.")
    if not 1 <= settings.get("history_events_max", 0) <= 4:
        raise ValueError("fovlanding-v1 allows at most four history events.")
    if settings.get("require_full_history_window") is not True:
        raise ValueError("fovlanding-v1 requires a complete history window.")


def quantize_xy(x: float, y: float, display_rect: Sequence[float]) -> tuple[int, int]:
    if len(display_rect) != 4:
        raise ValueError("display_rect must contain left, top, width, and height.")
    left, top, width, height = map(float, display_rect)
    values = (x, y, left, top, width, height)
    if not all(isfinite(value) for value in values) or width <= 0 or height <= 0:
        raise ValueError("Coordinates and display geometry must be finite and positive.")
    u = (x - left) / width
    v = (y - top) / height
    if not 0 <= u < 1 or not 0 <= v < 1:
        raise ValueError("Gaze is outside the video display rectangle.")
    return floor(100 * u), floor(100 * v)


def frame_plan(
    pts_ms: Sequence[float],
    cutoff_ms: float,
    condition: str,
    window_ms: float,
    frame_count: int,
) -> list[dict[str, Any]]:
    pts = [float(value) for value in pts_ms]
    if not pts or not isfinite(cutoff_ms) or not all(isfinite(value) for value in pts):
        raise ValueError("PTS and cutoff must be finite and PTS must be nonempty.")
    if any(left >= right for left, right in zip(pts, pts[1:])):
        raise ValueError("PTS must be strictly increasing.")
    if condition not in ("D1", "D4", "R4"):
        raise ValueError("Unknown input condition.")

    def at_or_before(value: float) -> int:
        index = bisect_right(pts, value) - 1
        if index < 0:
            raise ValueError("No causal frame is available.")
        return index

    current = at_or_before(cutoff_ms)
    if condition == "D1":
        return [{
            "requested_ms": cutoff_ms,
            "slot_frame_pts_ms": pts[current],
            "source_frame_index": current,
            "source_frame_pts_ms": pts[current],
            "relative_ms": pts[current] - cutoff_ms,
            "synthetic_repeat": False,
        }]
    if frame_count < 2 or window_ms <= 0 or pts[0] > cutoff_ms - window_ms:
        raise ValueError("A complete causal history window is required.")
    requests = [
        cutoff_ms - window_ms + index * window_ms / (frame_count - 1)
        for index in range(frame_count)
    ]
    requests[-1] = cutoff_ms
    historical = [at_or_before(value) for value in requests]
    result = []
    for requested, slot_index in zip(requests, historical):
        source_index = current if condition == "R4" else slot_index
        result.append({
            "requested_ms": requested,
            "slot_frame_pts_ms": pts[slot_index],
            "source_frame_index": source_index,
            "source_frame_pts_ms": pts[source_index],
            "relative_ms": pts[slot_index] - cutoff_ms,
            "synthetic_repeat": condition == "R4",
        })
    return result


def _validated_events(
    events: Sequence[Mapping[str, Any]],
    sample_count: int,
) -> list[dict[str, Any]]:
    result = []
    previous_stop = 0
    for event_id, source in enumerate(events):
        label = str(source["label"])
        start = int(source["start_sample"])
        stop = int(source["stop_sample"])
        start_ms = float(source["start_ms"])
        end_ms = float(source["end_ms"])
        if not label:
            raise ValueError("Detector labels must be nonempty.")
        if not 0 <= start < stop <= sample_count or start < previous_stop:
            raise ValueError("Events overlap or have invalid sample bounds.")
        if not isfinite(start_ms) or not isfinite(end_ms) or start_ms >= end_ms:
            raise ValueError("Event times must be finite and increasing.")
        result.append({
            "event_id": event_id,
            "label": label,
            "start_sample": start,
            "stop_sample": stop,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "usable_samples": bool(source.get("usable_samples", True)),
        })
        previous_stop = stop
    return result


def merge_foveations(
    events: Sequence[Mapping[str, Any]],
    sample_count: int,
    event_config: Mapping[str, Sequence[str]],
) -> tuple[list[Episode], list[dict[str, Any]]]:
    foveation_labels = set(event_config["foveation_labels"])
    checked = _validated_events(events, sample_count)
    episodes = []
    for event in checked:
        if event["label"] not in foveation_labels or not event['usable_samples']:
            continue
        if (episodes and episodes[-1].stop_sample == event["start_sample"]
                and isclose(episodes[-1].end_ms, event["start_ms"], rel_tol=0, abs_tol=1e-6)):
            previous = episodes.pop()
            episodes.append(Episode(
                previous.start_sample,
                event["stop_sample"],
                previous.start_ms,
                event["end_ms"],
                previous.event_ids + (event["event_id"],),
                previous.labels + (event["label"],),
            ))
        else:
            episodes.append(Episode(
                event["start_sample"],
                event["stop_sample"],
                event["start_ms"],
                event["end_ms"],
                (event["event_id"],),
                (event["label"],),
            ))
    return episodes, checked


def _transition(
    current: Episode,
    following: Episode,
    events: Sequence[Mapping[str, Any]],
    event_config: Mapping[str, Sequence[str]],
) -> tuple[bool, list[Mapping[str, Any]], str]:
    between = [
        event for event in events
        if event["start_sample"] >= current.stop_sample
        and event["stop_sample"] <= following.start_sample
    ]
    if not between or current.stop_sample != between[0]["start_sample"]:
        return False, between, "noncontiguous_before_transition"
    if any(not event['usable_samples'] for event in between):
        return False, between, 'invalid_transition_event'
    if between[-1]["stop_sample"] != following.start_sample:
        return False, between, "noncontiguous_after_transition"
    if any(
        left["stop_sample"] != right["start_sample"]
        for left, right in zip(between, between[1:])
    ):
        return False, between, "noncontiguous_within_transition"
    boundaries = [(current.end_ms, between[0]["start_ms"]),
                  (between[-1]["end_ms"], following.start_ms)]
    boundaries.extend((left["end_ms"], right["start_ms"]) for left, right in zip(between, between[1:]))
    if any(not isclose(left, right, rel_tol=0, abs_tol=1e-6) for left, right in boundaries):
        return False, between, "noncontiguous_transition_times"
    saccades = set(event_config["saccade_labels"])
    psos = set(event_config["pso_labels"])
    if between[0]["label"] not in saccades:
        return False, between, "transition_does_not_start_with_saccade"
    if any(event["label"] not in psos for event in between[1:]):
        return False, between, "unexpected_transition_label"
    return True, between, ""


def _sample_valid(sample: Mapping[str, Any]) -> bool:
    return bool(sample.get("valid")) and all(
        isfinite(float(sample[key])) for key in ("time_ms", "x", "y")
    )


def _history_record(
    episode: Episode,
    samples: Sequence[Mapping[str, Any]],
    cutoff_ms: float,
    display_rect: Sequence[float],
    max_raw_gap_ms: float,
) -> dict[str, Any]:
    selected = [
        (index, samples[index])
        for index in range(episode.start_sample, episode.stop_sample)
        if float(samples[index]["time_ms"]) < cutoff_ms
    ]
    if not selected or any(not _sample_valid(sample) for _, sample in selected):
        raise ValueError("Foveation history contains missing or invalid samples.")
    if any(float(right['time_ms']) - float(left['time_ms']) > max_raw_gap_ms
           for (_, left), (_, right) in zip(selected, selected[1:])):
        raise ValueError('Foveation history crosses a raw timestamp gap.')
    first_index, first = selected[0]
    last_index, last = selected[-1]
    return {
        "start_relative_ms": float(first["time_ms"]) - cutoff_ms,
        "end_boundary_relative_ms": min(episode.end_ms, cutoff_ms) - cutoff_ms,
        "last_sample_relative_ms": float(last["time_ms"]) - cutoff_ms,
        "start_xy_grid": list(quantize_xy(float(first["x"]), float(first["y"]), display_rect)),
        "end_xy_grid": list(quantize_xy(float(last["x"]), float(last["y"]), display_rect)),
        "_audit": {
            "start_ms": float(first["time_ms"]),
            "end_boundary_ms": min(episode.end_ms, cutoff_ms),
            "last_sample_ms": float(last["time_ms"]),
            "sample_start": first_index,
            "sample_stop": last_index + 1,
            "source_event_ids": list(episode.event_ids),
            "source_labels": list(episode.labels),
        },
    }


def build_trial_targets(
    trial: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    samples = trial["samples"]
    sample_times = [float(sample["time_ms"]) for sample in samples]
    if not samples or not all(isfinite(value) for value in sample_times):
        raise ValueError("Sample times must be finite and nonempty.")
    if any(left >= right for left, right in zip(sample_times, sample_times[1:])):
        raise ValueError("Sample times must be strictly increasing.")
    episodes, events = merge_foveations(trial["events"], len(samples), config["events"])
    settings = config["input"]
    max_raw_gap_ms = float(settings.get('max_raw_gap_ms', float('inf')))
    display_rect = trial["display_rect"]
    exclusions: Counter[str] = Counter()
    targets = []
    links: list[tuple[bool, list[Mapping[str, Any]], str]] = []
    for current, following in zip(episodes, episodes[1:]):
        links.append(_transition(current, following, events, config["events"]))
    for index, (valid_link, transition_events, reason) in enumerate(links):
        if not valid_link:
            exclusions[reason] += 1
            continue
        following = episodes[index + 1]
        transition_times = sample_times[episodes[index].stop_sample - 1:following.start_sample + 1]
        if any(right - left > max_raw_gap_ms for left, right in zip(transition_times, transition_times[1:])):
            exclusions['raw_transition_timestamp_gap'] += 1
            continue
        if any(not _sample_valid(sample) for sample in
               samples[episodes[index].stop_sample:following.start_sample]):
            exclusions["invalid_transition_samples"] += 1
            continue
        cutoff_ms = float(transition_events[0]["start_ms"])
        target_sample = samples[following.start_sample]
        if float(target_sample['time_ms']) >= trial.get('video_end_ms', float('inf')):
            exclusions['target_after_video_end'] += 1
            continue
        if float(target_sample["time_ms"]) <= cutoff_ms:
            exclusions["target_not_after_cutoff"] += 1
            continue
        if not _sample_valid(target_sample):
            exclusions["invalid_target_start_sample"] += 1
            continue
        try:
            target_grid = quantize_xy(
                float(target_sample["x"]),
                float(target_sample["y"]),
                display_rect,
            )
        except ValueError:
            exclusions["target_outside_display"] += 1
            continue
        history_episodes = episodes[:index + 1][-int(settings["history_events_max"]):]
        try:
            history_records = [
                _history_record(episode, samples, cutoff_ms, display_rect, max_raw_gap_ms)
                for episode in history_episodes
            ]
            plans = {
                condition: frame_plan(
                    trial["frame_pts_ms"],
                    cutoff_ms,
                    condition,
                    float(settings["history_window_ms"]),
                    int(settings["frame_count"]),
                )
                for condition in ("D1", "R4", "D4")
            }
        except ValueError as error:
            exclusions[str(error)] += 1
            continue
        history = [
            {key: value for key, value in record.items() if key != "_audit"}
            for record in history_records
        ]
        target_id = f"{trial['trial_id']}-transition-{transition_events[0]['event_id']:06d}"
        targets.append({
            "schema_version": config["protocol"]["name"],
            "target_id": target_id,
            "model_input": {
                "cutoff_ms": cutoff_ms,
                "frame_plans": plans,
                "history": history,
                "task": config["protocol"]["task"],
            },
            "target": {"xy_grid": list(target_grid)},
            "audit": {
                "target_start_ms": float(target_sample["time_ms"]),
                "target_sample_index": following.start_sample,
                "trial_id": trial["trial_id"],
                "source_file": trial.get("source_file"),
                "source_film_id": trial["source_film_id"],
                "video_id": trial["video_id"],
                "observer_id": trial["observer_id"],
                "split": trial["split"],
                "quantization": config["protocol"]["quantization"],
                "event_rule": config["protocol"]["event_rule"],
                "transition_event_ids": [event["event_id"] for event in transition_events],
                "transition_labels": [event["label"] for event in transition_events],
                "target_source_event_ids": list(following.event_ids),
                "target_source_labels": list(following.labels),
                "history_sources": [record["_audit"] for record in history_records],
            },
        })
    return targets, exclusions


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on line {line_number}.") from error


def serialize_target(
    target: Mapping[str, Any], condition: str, image_paths: Mapping[int, Path | str],
) -> dict[str, Any]:
    """Serialize only the approved inputs and the final supervised coordinate."""
    source = target['model_input']
    if source['task'] != 'free_viewing' or condition not in ('D1', 'D4', 'R4'):
        raise ValueError('Unknown task or visual condition.')
    lines = ['Task: free viewing.', 'Predict the landing position after the next saccade.',
             'Times are milliseconds relative to the input cutoff.',
             'Video frames, oldest to newest:']
    images = []
    for index, frame in enumerate(source['frame_plans'][condition], 1):
        if frame['source_frame_pts_ms'] > source['cutoff_ms'] or frame['relative_ms'] > 0:
            raise ValueError('Future frame in input.')
        images.append(str(image_paths[frame['source_frame_index']]))
        lines.append(f"Frame {index}: {frame['relative_ms']:.3f} ms. <image>")
    lines.append('Past foveation events, oldest to newest:')
    for event in source['history']:
        start, last = event['start_relative_ms'], event['last_sample_relative_ms']
        end = event['end_boundary_relative_ms']
        if not start <= last < 0 or not last <= end <= 0:
            raise ValueError('Future or inconsistent gaze history.')
        sx, sy = event['start_xy_grid']
        ex, ey = event['end_xy_grid']
        lines.append(f'Start {start:.3f} ms; end {end:.3f} ms; '
                     f'start ({sx:02d}, {sy:02d}); end ({ex:02d}, {ey:02d}).')
    lines.extend(['Coordinates use integers 00-99 in the video display rectangle, x rightward, y downward.',
                  'Return only the next position in the format (XX, YY).'])
    x, y = target['target']['xy_grid']
    return {'images': images, 'conversations': [
        {'from': 'human', 'value': '\n'.join(lines)},
        {'from': 'gpt', 'value': f'({x:02d}, {y:02d})'}]}


def write_build(
    trials: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Use a new output directory.")
    all_targets = []
    exclusions: Counter[str] = Counter()
    trial_count = 0
    for trial in trials:
        trial_count += 1
        targets, rejected = build_trial_targets(trial, config)
        all_targets.extend(targets)
        exclusions.update(rejected)
    ids = [target["target_id"] for target in all_targets]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Build produced no targets or duplicate target IDs.")
    output.mkdir(parents=True)
    with (output / OUTPUT_TARGETS).open("x") as stream:
        for target in all_targets:
            stream.write(json.dumps(target, allow_nan=False, separators=(",", ":")) + "\n")
    summary = {
        "status": "fovlanding-v1 target build complete",
        "trials": trial_count,
        "targets": len(all_targets),
        "exclusions": dict(sorted(exclusions.items())),
        "protocol": config["protocol"],
        "provenance": config["provenance"],
    }
    (output / OUTPUT_SUMMARY).write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run fovlanding builds inside Slurm.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config, args.trials)
    summary = write_build(read_jsonl(args.trials), config, args.output)
    print(json.dumps(summary, allow_nan=False))


if __name__ == "__main__":
    main()
