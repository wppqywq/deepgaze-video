"""Contract tests for the fovlanding-v1 data layer."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from fovlanding import (
    build_trial_targets,
    frame_plan,
    load_config,
    preflight_config,
    quantize_xy,
    serialize_target,
    write_build,
)

CONFIG = {
    "protocol": {
        "name": "fovlanding-v1",
        "quantization": "floor100-v1",
        "event_rule": "merge-fixa-purs-v1",
        "task": "free_viewing",
        "cutoff_rule": "next_saccade_onset",
        "gaze_input_rule": "sample_time_strictly_less_than_cutoff",
        "video_input_rule": "frame_pts_less_than_or_equal_to_cutoff",
        "target_rule": "first_valid_raw_sample_at_next_foveation_start",
        "target_grid": [100, 100],
    },
    "events": {
        "foveation_labels": ["FIXA", "PURS"],
        "saccade_labels": ["SACC", "ISAC"],
        "pso_labels": ["HPSO", "LPSO", "IHPS", "ILPS"],
    },
    "input": {
        "frame_count": 4,
        "history_window_ms": 1000.0,
        "history_events_max": 4,
        "require_full_history_window": True,
    },
    "provenance": {
        "project_git_commit": "a" * 40,
        "upstream_git_commit": "b" * 40,
        "dataset_manifest_sha256": "c" * 64,
        "split_manifest_sha256": "d" * 64,
        "detector_version": "1.1.2",
        "detector_parameters_file": "detector.json",
        "audited_trial_metadata_file": "trials.csv",
        "final_test_accessed": True,
    },
}


def trial():
    labels = [
        ("FIXA", 0, 2),
        ("PURS", 2, 4),
        ("SACC", 4, 5),
        ("HPSO", 5, 6),
        ("FIXA", 6, 8),
        ("ISAC", 8, 9),
        ("PURS", 9, 11),
    ]
    samples = []
    for index in range(11):
        samples.append({
            "time_ms": 1100.0 + index * 10.0,
            "x": 100.0 + index * 10.0,
            "y": 200.0 + index * 5.0,
            "valid": True,
        })
    events = [{
        "label": label,
        "start_sample": start,
        "stop_sample": stop,
        "start_ms": samples[start]["time_ms"],
        "end_ms": samples[stop]["time_ms"] if stop < len(samples) else 1210.0,
    } for label, start, stop in labels]
    return {
        "trial_id": "trial-1",
        "video_id": "clip-1",
        "source_film_id": "film-1",
        "observer_id": "observer-1",
        "split": "train",
        "display_rect": [0.0, 0.0, 1000.0, 1000.0],
        "frame_pts_ms": [float(value) for value in range(0, 1300, 100)],
        "samples": samples,
        "events": events,
    }


class FovlandingContracts(unittest.TestCase):
    def test_neural_ar_uses_prompt_gaze_and_training_normalization(self):
        from types import SimpleNamespace
        import numpy as np
        import torch
        from routeb.baseline import fovlanding_tensors, export_heatmaps
        from baseline import Population, cell_log_mass
        rows = []
        for split in ('train', 'validation'):
            row = copy.deepcopy(build_trial_targets(trial(), CONFIG)[0][0])
            row['target_id'] = split
            row['audit'].update(split=split, source_film_id=split, video_id=split)
            rows.append(row)
        paths = {split: {f['source_frame_index']: f'{split}-{f["source_frame_index"]}.png'
            for f in rows[i]['model_input']['frame_plans']['D4']}
            for i, split in enumerate(('train', 'validation'))}
        inputs = SimpleNamespace(targets=rows, paths=paths, ids={'train': ['train'], 'validation': ['validation']})
        _, requested = fovlanding_tensors(inputs, 'd4')
        cache = {key: np.full(2048, key[1], np.float32) for key in requested}
        data, norm, _ = fovlanding_tensors(inputs, 'd4', cache)
        self.assertEqual(data[0].shape, (2, 4, 6))
        model = Population(history_dimensions=6)
        loss = -cell_log_mass(model(*data[:5]), data[5]).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        rows[1]['target']['xy_grid'] = [99, 99]
        rows[1]['audit']['observer_id'] = 'not an input'
        again, _, _ = fovlanding_tensors(inputs, 'd4', cache)
        for original, changed in zip(data[:5], again[:5]):
            torch.testing.assert_close(original, changed)
        changed_cache = {key: value + (100 if key[0] == 'validation' else 0) for key, value in cache.items()}
        _, norm2, _ = fovlanding_tensors(inputs, 'd4', changed_cache)
        for key in norm:
            np.testing.assert_array_equal(norm[key], norm2[key])
        repeated, _, _ = fovlanding_tensors(inputs, 'r4', cache)
        torch.testing.assert_close(repeated[4], data[4])
        torch.testing.assert_close(repeated[2][:, 0], repeated[2][:, -1])
        exported_rows, _ = fovlanding_tensors(inputs, 'd4')
        with tempfile.TemporaryDirectory() as temporary:
            report = export_heatmaps(model, again, np.array([1]), exported_rows, Path(temporary), 'cpu')
            with np.load(Path(temporary) / report['file'], allow_pickle=False) as saved:
                self.assertEqual(saved['target_ids'].tolist(), ['validation'])
                self.assertEqual(saved['probabilities'].shape, (1, 100, 100))
                self.assertAlmostEqual(float(saved['probabilities'].sum()), 1., places=6)

    def test_mixture_cell_mass_normalizes_and_has_finite_tail_gradients(self):
        import numpy as np
        import torch
        from baseline import cell_log_mass, parameters
        from stage0 import gmm_grid
        cells = torch.tensor([[x, y] for y in range(100) for x in range(100)])
        output = torch.linspace(-1, 1, 25, dtype=torch.float64).reshape(1, 5, 5)
        masses = cell_log_mass(output.expand(10000, -1, -1), cells).exp()
        weights, means, scales = [v[0].numpy() for v in parameters(output)]
        np.testing.assert_allclose(masses.numpy().reshape(100, 100), gmm_grid(weights, means, scales), rtol=1e-9)
        self.assertAlmostEqual(float(masses.sum()), 1., places=10)
        extreme = torch.tensor([[[0., 12., 12., -12., -12.]]], requires_grad=True)
        loss = -cell_log_mass(extreme, torch.tensor([[0, 0]])).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(extreme.grad).all())

    def test_seed_summary_keeps_films_as_resampling_units(self):
        from routeb.comparison import summarize_seeds
        reports = [{'seed': seed, 'selection': 'validation_macro_film_best', 'by_film_bits': {
            film: {'d1': -11., 'static_d1': -12., 'r4': -10., 'd4': -10. + difference}
            for film, difference in zip(('a', 'b'), (-.1 + seed * .1, .1 + seed * .1))}}
            for seed in range(3)]
        result = summarize_seeds(reports)
        self.assertEqual(result['films'], 2)
        contrast = result['paired_contrasts']['d4-r4']
        self.assertAlmostEqual(contrast['macro_bits'], .1)
        self.assertAlmostEqual(contrast['seed_std_bits'], .1)
        self.assertAlmostEqual(contrast['by_film_seed_mean_bits']['a'], 0.)
        with self.assertRaisesRegex(ValueError, 'distinct'):
            summarize_seeds([reports[0]] * 3)
        changed = copy.deepcopy(reports)
        changed[1]['selection'] = 'same_optimizer_update'
        with self.assertRaisesRegex(ValueError, 'selection'):
            summarize_seeds(changed)

    def test_ar_cell_probabilities_match_existing_grid_including_tails(self):
        import numpy as np
        from stage0 import gaussian_cell_log_mass, gmm_grid
        cells = np.array([[x, y] for y in range(100) for x in range(100)])
        for mean, scale in (([.4, .6], [.15, .25]), ([-.5, 1.5], [.2, .3])):
            scores = gaussian_cell_log_mass(np.tile(mean, (10000, 1)), scale, cells)
            grid = gmm_grid(np.ones(1), np.array([mean]), np.array([scale]))
            np.testing.assert_allclose(np.exp(scores).reshape(100, 100), grid, atol=1e-12, rtol=1e-8)
            self.assertAlmostEqual(float(np.exp(scores).sum()), 1., places=10)

    def test_ar_features_ignore_targets_and_reject_future_history(self):
        import numpy as np
        from scripts.compare_video import history_features
        source = build_trial_targets(trial(), CONFIG)[0][0]
        expected = history_features([source], 4)
        source['target']['xy_grid'] = [99, 99]
        source['audit'] = {'target_start_ms': 98765, 'observer_id': 'secret'}
        np.testing.assert_array_equal(history_features([source], 4), expected)
        source['model_input']['history'][0]['last_sample_relative_ms'] = 0
        with self.assertRaises(ValueError):
            history_features([source], 4)

    def test_ar_fit_and_selection_do_not_use_validation_targets(self):
        from types import SimpleNamespace
        from scripts.compare_video import history_baselines
        rows = []
        for index in range(10):
            row = copy.deepcopy(build_trial_targets(trial(), CONFIG)[0][0])
            row['target_id'] = str(index)
            row['audit']['source_film_id'] = f'film-{index % 2}'
            row['model_input']['history'][-1]['end_xy_grid'] = [10 + index * 7, 25 + index * 3]
            row['target']['xy_grid'] = [20 + index * 5, 30 + index * 2]
            rows.append(row)
        inputs = SimpleNamespace(targets=rows, ids={'train': [str(i) for i in range(8)], 'validation': ['8', '9']})
        _, before = history_baselines(inputs)
        for row in rows[8:]:
            row['target']['xy_grid'] = [99, 0]
        _, after = history_baselines(inputs)
        self.assertEqual(before, after)

    def test_persistence_scale_keeps_systematic_displacement(self):
        import numpy as np
        from scripts.compare_video import fit_history_gaussian
        features = np.array([[.105, .205], [.205, .305], [.305, .405]])
        cells = np.array([[20, 40], [30, 50], [40, 60]])
        # Constant nonzero displacement has zero centered std, but nonzero error.
        fit = fit_history_gaussian(features, cells)
        np.testing.assert_allclose(fit['scale'], [.1, .2])

    def test_full_history_uses_only_serialized_gaze_information(self):
        import numpy as np
        from scripts.compare_video import history_features
        source = build_trial_targets(trial(), CONFIG)[0][0]
        event = source['model_input']['history'][-1]
        actual = history_features([source], 4, True)
        self.assertEqual(actual.shape, (1, 28))
        np.testing.assert_allclose(actual[0, :4], (np.array(event['end_xy_grid'] + event['start_xy_grid']) + .5) / 100)
        np.testing.assert_allclose(actual[0, 4:6], [float(f"{event[key]:.3f}") / 1000
            for key in ('start_relative_ms', 'end_boundary_relative_ms')])
        source['target']['xy_grid'] = [99, 99]
        source['audit'] = {}
        # Last sample time validates causality but is not a prompt feature.
        event['last_sample_relative_ms'] = event['start_relative_ms']
        np.testing.assert_array_equal(history_features([source], 4, True), actual)
        endpoint_only = history_features([source], 4)
        event['start_xy_grid'] = [(event['start_xy_grid'][0] + 1) % 100, 0]
        self.assertFalse(np.array_equal(history_features([source], 4, True), actual))
        np.testing.assert_array_equal(history_features([source], 4), endpoint_only)

    def test_history_rejects_invalid_boundaries_and_coordinates(self):
        from scripts.compare_video import history_features
        for key, value in [('start_relative_ms', float('nan')), ('end_boundary_relative_ms', 1.),
                           ('start_relative_ms', 0.), ('start_xy_grid', [100, 0]),
                           ('end_xy_grid', [1.5, 2])]:
            for full in (False, True):
                with self.subTest(key=key, full=full):
                    source = build_trial_targets(trial(), CONFIG)[0][0]
                    source['model_input']['history'][-1][key] = value
                    with self.assertRaises(ValueError):
                        history_features([source], 4, full)

    def test_film_macro_does_not_weight_large_films_more(self):
        from routeb.evaluation import macro_film_bits
        self.assertEqual(macro_film_bits({'a': {'targets': 100, 'mean_bits': -8},
                                         'b': {'targets': 1, 'mean_bits': -12}}), -10)
        with self.assertRaises(ValueError):
            macro_film_bits({'a': {'mean_bits': float('nan')}})

    def test_warmup_cosine_schedule_and_restored_update(self):
        from routeb.train import learning_rate_multiplier
        config = {'scheduler': 'cosine', 'max_optimizer_steps': 1000, 'warmup_fraction': .05}
        self.assertAlmostEqual(learning_rate_multiplier(0, config), .02)
        self.assertEqual(learning_rate_multiplier(49, config), 1)
        self.assertEqual(learning_rate_multiplier(50, config), 1)
        self.assertAlmostEqual(learning_rate_multiplier(525, config), .5)
        self.assertEqual(learning_rate_multiplier(1000, config), 0)

    def test_checkpoint_selection_uses_macro_and_earliest_tie(self):
        from routeb.comparison import select_evaluation
        evaluations = [
            {'epoch': 1, 'mean_bits': -8, 'selection_bits': -10},
            {'epoch': 2, 'mean_bits': -9, 'selection_bits': -9},
            {'epoch': 3, 'mean_bits': -7, 'selection_bits': -9}]
        self.assertEqual(select_evaluation(evaluations, 3, True, 'selection_bits')['epoch'], 2)
        self.assertEqual(select_evaluation(evaluations, 3, False, 'selection_bits')['epoch'], 3)

    def test_fovlanding_retention_preserves_selected_validation_and_recovery(self):
        from routeb.retention import prune_intermediate
        protocol = {'architecture': 'single-gaze-adapter-fovlanding-v1'}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for update, checks in ((0, 0), (100, 0), (100, 1), (200, 1), (200, 2), (300, 2), (300, 3)):
                path = root / f'checkpoint-{update:06d}-v{checks:03d}'
                path.mkdir()
                (path / 'manifest.json').write_text(json.dumps({'update': update, 'metadata': {
                    'protocol': protocol, 'progress': {'validated_epochs': checks,
                        'best_checkpoint': 'checkpoints/checkpoint-000100-v001'}}}))
            removed = prune_intermediate(root, protocol, 100)
            self.assertEqual(set(removed), {'checkpoint-000100-v000', 'checkpoint-000200-v001'})
            self.assertTrue((root / 'checkpoint-000100-v001').exists())

    def test_materialized_inputs_reject_film_leakage_and_changed_frames(self):
        import numpy as np
        from routeb.data import FovlandingInputs
        source = build_trial_targets(trial(), CONFIG)[0]
        for index, target in enumerate(source):
            target['audit']['split'] = 'train' if index == 0 else 'validation'
            target['audit']['source_film_id'] = f'film-{index}'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'frame.png').write_bytes(b'frame-content-for-hash-check')
            indices = {frame['source_frame_index'] for target in source for frame in target['model_input']['frame_plans']['D4']}
            frame_hash = hashlib.sha256((root / 'frame.png').read_bytes()).hexdigest()
            (root / 'frames.json').write_text(json.dumps({'clip-1': [
                {'index': index, 'path': 'frame.png', 'sha256': frame_hash} for index in indices]}))
            np.save(root / 'training-prior.npy', np.full((100, 100), .0001))
            (root / 'targets.jsonl').write_text(''.join(json.dumps(target) + '\n' for target in source))

            def scan():
                (root / 'scan.json').write_text(json.dumps({'status': 'Full fovlanding processor scan passed',
                    'max_length': 4096, 'split_counts': {'train': 1, 'validation': 1}, 'assumptions': {},
                    'data_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                        for name in ('targets.jsonl', 'frames.json', 'training-prior.npy')}}))
            scan()
            inputs = FovlandingInputs(root, root / 'scan.json')
            self.assertEqual(inputs.rows('d4', 'validation')[0]['conversations'],
                             inputs.rows('r4', 'validation')[0]['conversations'])
            (root / 'frame.png').write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'frame changed'):
                FovlandingInputs(root, root / 'scan.json')
            source[1]['audit']['source_film_id'] = source[0]['audit']['source_film_id']
            (root / 'targets.jsonl').write_text(''.join(json.dumps(target) + '\n' for target in source))
            scan()
            with self.assertRaisesRegex(RuntimeError, 'Source-film leakage'):
                FovlandingInputs(root, root / 'scan.json')

    def test_quantization_rejects_outside_instead_of_clipping(self):
        self.assertEqual(quantize_xy(700.0, 440.0, [0.0, 0.0, 1000.0, 1000.0]), (70, 44))
        with self.assertRaisesRegex(ValueError, "outside"):
            quantize_xy(1000.0, 440.0, [0.0, 0.0, 1000.0, 1000.0])

    def test_frame_plans_are_causal_and_r4_preserves_slot_times(self):
        pts = [0.0, 100.0, 200.0, 300.0]
        d4 = frame_plan(pts, 300.0, "D4", 300.0, 4)
        r4 = frame_plan(pts, 300.0, "R4", 300.0, 4)
        self.assertEqual([row["source_frame_index"] for row in d4], [0, 1, 2, 3])
        self.assertEqual([row["source_frame_index"] for row in r4], [3, 3, 3, 3])
        self.assertEqual([row["relative_ms"] for row in d4], [row["relative_ms"] for row in r4])
        self.assertTrue(all(row["source_frame_pts_ms"] <= 300.0 for row in d4 + r4))

    def test_build_merges_fixa_purs_and_keeps_target_out_of_input(self):
        targets, exclusions = build_trial_targets(trial(), CONFIG)
        self.assertEqual(len(targets), 2)
        self.assertEqual(exclusions, {})
        first = targets[0]
        history = first["model_input"]["history"][0]
        self.assertNotIn("source_labels", history)
        self.assertNotIn("sample_start", history)
        self.assertEqual(history["start_relative_ms"], -40.0)
        self.assertEqual(first["audit"]["history_sources"][0]["source_labels"], ["FIXA", "PURS"])
        self.assertEqual(first["audit"]["transition_labels"], ["SACC", "HPSO"])
        self.assertEqual(first["target"]["xy_grid"], [16, 23])
        self.assertNotIn("target_start_ms", first["model_input"])
        self.assertLess(first["model_input"]["cutoff_ms"], first["audit"]["target_start_ms"])
        self.assertEqual(first["model_input"]["frame_plans"]["D1"][-1],
                         first["model_input"]["frame_plans"]["D4"][-1])

    def test_invalid_first_target_sample_is_not_replaced(self):
        source = trial()
        source["samples"][9]["valid"] = False
        targets, exclusions = build_trial_targets(source, CONFIG)
        self.assertEqual(len(targets), 1)
        self.assertEqual(exclusions["invalid_target_start_sample"], 1)
        self.assertEqual(targets[0]["audit"]["target_sample_index"], 6)

    def test_gap_breaks_transition(self):
        source = trial()
        del source["events"][2]
        targets, exclusions = build_trial_targets(source, CONFIG)
        self.assertEqual(len(targets), 1)
        self.assertEqual(exclusions["noncontiguous_before_transition"], 1)
        self.assertEqual(len(targets[0]["model_input"]["history"]), 2)

    def test_missing_raw_sample_breaks_even_labeled_saccade(self):
        source = trial()
        source["samples"][4]["valid"] = False
        targets, exclusions = build_trial_targets(source, CONFIG)
        self.assertEqual(len(targets), 1)
        self.assertEqual(exclusions["invalid_transition_samples"], 1)
        self.assertEqual(targets[0]["audit"]["target_sample_index"], 9)

    def test_time_gap_breaks_even_index_adjacent_transition(self):
        source = trial()
        source['events'][3]['end_ms'] -= 1
        targets, exclusions = build_trial_targets(source, CONFIG)
        self.assertEqual(len(targets), 1)
        self.assertEqual(exclusions['noncontiguous_transition_times'], 1)

    def test_raw_gap_and_detector_invalidity_break_transitions(self):
        config = copy.deepcopy(CONFIG)
        config['input']['max_raw_gap_ms'] = 10
        source = trial()
        source['samples'][5]['time_ms'] += 1
        targets, rejected = build_trial_targets(source, config)
        self.assertEqual(len(targets), 1)
        self.assertEqual(rejected['raw_transition_timestamp_gap'], 1)
        source = trial()
        source['events'][2]['usable_samples'] = False
        targets, rejected = build_trial_targets(source, config)
        self.assertEqual(len(targets), 1)
        self.assertEqual(rejected['invalid_transition_event'], 1)

    def test_target_beyond_video_duration_is_excluded(self):
        source = trial()
        source['video_end_ms'] = 1190
        targets, rejected = build_trial_targets(source, CONFIG)
        self.assertEqual(len(targets), 1)
        self.assertEqual(rejected['target_after_video_end'], 1)

    def test_serializer_matches_d4_r4_and_ignores_audit(self):
        target = build_trial_targets(trial(), CONFIG)[0][0]
        paths = {index: f'/frames/{index}.png' for index in range(13)}
        d4 = serialize_target(target, 'D4', paths)
        r4 = serialize_target(target, 'R4', paths)
        d1 = serialize_target(target, 'D1', paths)
        self.assertEqual(d4['conversations'], r4['conversations'])
        self.assertEqual(len(d4['images']), 4)
        self.assertEqual(r4['images'], d1['images'] * 4)
        self.assertEqual(d4['images'][-1], d1['images'][0])
        target['audit'] = {'observer_id': 'secret', 'target_start_ms': 1234567}
        self.assertEqual(serialize_target(target, 'D4', paths), d4)
        self.assertEqual(d4['conversations'][-1]['value'], '(16, 23)')

    def test_serializer_rejects_future_gaze(self):
        target = build_trial_targets(trial(), CONFIG)[0][0]
        target['model_input']['history'][0]['last_sample_relative_ms'] = 0
        with self.assertRaisesRegex(ValueError, 'gaze history'):
            serialize_target(target, 'D1', {11: '/frame.png'})

    def test_internal_transition_gap_is_rejected(self):
        source = trial()
        source["events"][3]["start_sample"] = 6
        source["events"][3]["stop_sample"] = 7
        source["events"][3]["start_ms"] = source["samples"][6]["time_ms"]
        source["events"][3]["end_ms"] = source["samples"][7]["time_ms"]
        source["events"][4]["start_sample"] = 7
        source["events"][4]["start_ms"] = source["samples"][7]["time_ms"]
        targets, exclusions = build_trial_targets(source, CONFIG)
        self.assertEqual(len(targets), 1)
        self.assertEqual(exclusions["noncontiguous_within_transition"], 1)

    def test_preflight_rejects_unresolved_provenance(self):
        config = copy.deepcopy(CONFIG)
        config["provenance"]["detector_version"] = None
        with self.assertRaisesRegex(ValueError, "detector_version"):
            preflight_config(config)

    def test_preflight_rejects_placeholder_hashes(self):
        config = copy.deepcopy(CONFIG)
        config["provenance"]["dataset_manifest_sha256"] = "dataset"
        with self.assertRaisesRegex(ValueError, "SHA256"):
            preflight_config(config)

    def test_formal_load_rejects_blocked_data_audit(self):
        config = copy.deepcopy(CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            audit = Path(temporary) / 'audit.json'
            audit.write_text(json.dumps({'formal_target_build': {'valid_now': False}}))
            config['provenance']['data_validation_file'] = str(audit)
            path = Path(temporary) / 'config.json'
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'blocks formal'):
                load_config(path)

    def test_formal_load_is_bound_to_the_audited_input(self):
        config = copy.deepcopy(CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'trials.jsonl'
            source.write_text('{}\n')
            audit = Path(temporary) / 'audit.json'
            audit.write_text(json.dumps({
                'formal_target_build': {'valid_now': True},
                'dataset_manifest_sha256': config['provenance']['dataset_manifest_sha256'],
                'split_manifest_sha256': config['provenance']['split_manifest_sha256'],
                'standardized_trials_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            }))
            config['provenance']['data_validation_file'] = str(audit)
            path = Path(temporary) / 'config.json'
            path.write_text(json.dumps(config))
            self.assertEqual(load_config(path, source), config)
            source.write_text('{"changed":true}\n')
            with self.assertRaisesRegex(ValueError, 'Trial input differs'):
                load_config(path, source)

    def test_development_scope_requires_matching_assumptions(self):
        config = copy.deepcopy(CONFIG)
        config['provenance'].update(validation_scope='development_assumptions', assumptions={'screen': [2560, 1440]})
        with tempfile.TemporaryDirectory() as temporary:
            audit = Path(temporary) / 'audit.json'
            record = {'formal_target_build': {'valid_now': False}, 'development_pilot': {'eligible': True},
                'assumptions': config['provenance']['assumptions'],
                'dataset_manifest_sha256': config['provenance']['dataset_manifest_sha256'],
                'split_manifest_sha256': config['provenance']['split_manifest_sha256']}
            audit.write_text(json.dumps(record))
            config['provenance']['data_validation_file'] = str(audit)
            path = Path(temporary) / 'config.json'
            path.write_text(json.dumps(config))
            self.assertEqual(load_config(path), config)
            config['provenance']['assumptions'] = {'screen': [1920, 1080]}
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'matching empirical'):
                load_config(path)

    def test_failed_build_does_not_reserve_output_directory(self):
        source = trial()
        source["events"] = []
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "build"
            with self.assertRaisesRegex(ValueError, "no targets"):
                write_build([source], CONFIG, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
