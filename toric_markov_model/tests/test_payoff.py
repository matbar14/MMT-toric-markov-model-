import unittest

import numpy as np
import torch

from tests.support import market_frame
from toric_markov_model.data.payoff import PayoffData, causal_context, execution_outcomes, validate_market, walk_forward_segments
from toric_markov_model.execution import ExecutionConfig, exit_on_bar
from toric_markov_model.model.payoff import ToricPayoffModel
from toric_markov_model.train.payoff import choose_actions, policy_actions, select_policy, simulate, train_toric


class PayoffTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_same_bar_stop_first_and_open_gap(self):
        self.assertEqual(exit_on_bar(1, 100, 100, 103, 98, 102, 0.01, 0.02, 1, 4),
                         (True, 99.0, "STOP_LOSS"))
        self.assertEqual(exit_on_bar(-1, 100, 104, 105, 98, 102, 0.01, 0.02, 2, 4),
                         (True, 104.0, "STOP_LOSS"))

    def test_positive_terminal_label_can_be_a_realized_loss(self):
        frame = market_frame(3)
        frame["spot_open"], frame["spot_high"], frame["spot_low"], frame["spot_close"] = 100, 103, 98, 102
        returns, exits = execution_outcomes(frame, ExecutionConfig(horizon=1, slippage=0))
        self.assertTrue((frame.spot_close / frame.spot_open - 1 > 0.003).all())
        np.testing.assert_allclose(returns[:, 0], -0.01 - 0.001 * 1.99)
        np.testing.assert_array_equal(exits[:, 0], np.arange(3))

    def test_flat_market_costs_are_negative_for_both_sides(self):
        frame = market_frame(6)
        for field in ("open", "high", "low", "close"):
            frame[f"spot_{field}"] = 100.0
        config = ExecutionConfig(horizon=2)
        returns, exits = execution_outcomes(frame, config)
        self.assertTrue((returns[:5] < 0).all())
        self.assertTrue(np.isnan(returns[-1]).all())
        np.testing.assert_array_equal(exits[:5, 0], np.arange(1, 6))
        for column, side in enumerate((1, -1)):
            ratio = (1 - side * config.slippage) / (1 + side * config.slippage)
            self.assertAlmostEqual(returns[0, column], side * (ratio - 1) - config.fee * (1 + ratio))

    def test_simulator_matches_counterfactual_labels_including_short_costs(self):
        data = PayoffData.from_frame(market_frame(250))
        for side, column in ((1, 0), (-1, 1)):
            entries = np.arange(120, 150)
            actions = np.zeros(len(entries), dtype=int)
            actions[0] = side
            run = simulate(data, entries, actions)
            trade = run["trades"][0]
            self.assertEqual(trade["exit"], data.exit_indices[120, column])
            self.assertAlmostEqual(trade["net_return"], data.net_returns[120, column], places=12)
            self.assertAlmostEqual(run["equity"][-1], 1 + 0.2 * data.net_returns[120, column], places=12)
            self.assertAlmostEqual(np.prod(1 + run["daily_returns"]), run["equity"][-1], places=12)

    def test_cooldown_and_one_position_at_a_time(self):
        data = PayoffData.from_frame(market_frame(250), ExecutionConfig(horizon=1, cooldown=1))
        entries = np.arange(120, 125)
        run = simulate(data, entries, np.ones(5))
        self.assertEqual([trade["entry"] for trade in run["trades"]], [120, 122, 124])

    def test_context_and_candidates_do_not_read_entry_or_future(self):
        frame = market_frame(260)
        original, names, candidates = causal_context(frame)
        changed = frame.copy()
        changed.loc[200:, "spot_close"] *= 2
        changed.loc[200:, "spot_cvd"] += 1e6
        altered, altered_names, altered_candidates = causal_context(changed)
        self.assertEqual(names, altered_names)
        np.testing.assert_array_equal(original[:200], altered[:200])
        np.testing.assert_array_equal(candidates[:200], altered_candidates[:200])
        data = PayoffData.from_frame(frame)
        window = data.windows(np.array([200]), 16)[0]
        np.testing.assert_array_equal(window, original[184:200])

    def test_cvd_origin_does_not_change_features_or_pattern_candidates(self):
        frame = market_frame(260)
        original, _, candidates = causal_context(frame)
        frame.spot_cvd += 1e6
        frame.futures_cvd -= 1e6
        altered, _, altered_candidates = causal_context(frame)
        np.testing.assert_allclose(original, altered, atol=1e-5)
        np.testing.assert_array_equal(candidates, altered_candidates)

    def test_walk_forward_context_and_full_horizons_are_disjoint(self):
        data = PayoffData.from_frame(market_frame(1000))
        previous_eval_end = None
        for segment in walk_forward_segments(1000):
            entries = {name: data.entries(start, end, 16) for name, (start, end) in segment.items()}
            for previous, following in (("train", "selection"), ("selection", "calibration"), ("calibration", "evaluation")):
                self.assertLess(entries[previous][-1] + data.config.horizon - 1, entries[following][0] - 16)
            if previous_eval_end is not None:
                self.assertEqual(segment["evaluation"][0], previous_eval_end)
            previous_eval_end = segment["evaluation"][1]
        self.assertEqual(previous_eval_end, 1000)

    def test_negative_tied_or_unavailable_edges_do_not_force_orders(self):
        scores = np.array([[-0.1, -0.2], [0.01, 0.01], [0.01, -0.01], [-0.01, 0.01]])
        np.testing.assert_array_equal(choose_actions(scores, 0), [0, 0, 1, -1])
        np.testing.assert_array_equal(choose_actions(scores, 0, np.zeros_like(scores, dtype=bool)), np.zeros(4))
        with self.assertRaises(ValueError):
            choose_actions(np.full((2, 2), np.nan), 0)

    def test_no_edge_policy_stays_disabled(self):
        data = PayoffData.from_frame(market_frame(500))
        entries = data.entries(120, 400, 16)
        scores = np.full((len(entries), 2), -0.001)
        policy = select_policy(data, entries, scores)
        self.assertFalse(policy["enabled"])
        self.assertFalse(policy["production_approved"])
        np.testing.assert_array_equal(policy_actions(data, entries, scores, policy), np.zeros(len(entries)))

    def test_payoff_head_is_trained_and_inverse_scaled(self):
        model = ToricPayoffModel(4, max_len=4, dim_angles=8, num_states=4, dropout=0)
        with self.assertRaises(RuntimeError):
            model(torch.zeros(2, 4, 4))
        model.fit_statistics(torch.randn(30, 4), torch.randn(30, 2) * 0.01)
        features = torch.randn(2, 4, 4)
        loss = model(features).square().mean()
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertFalse(hasattr(model, "pattern_head"))
        self.assertFalse(hasattr(model, "non_hold_gate_head"))
        model.eval()
        torch.testing.assert_close(model.predict_payoffs(features), model(features) * model.target_std + model.target_mean)

    def test_training_statistics_and_selection_do_not_use_evaluation(self):
        data = PayoffData.from_frame(market_frame(500))
        entries = dict(train=data.entries(0, 280, 8), selection=data.entries(280, 370, 8))
        model, report = train_toric(data, entries, seq_len=8, epochs=1)
        self.assertEqual(report["best_epoch"], 1)
        np.testing.assert_allclose(model.target_mean.numpy(), data.net_returns[entries["train"]].mean(0), atol=1e-8)
        predicted = model.predict_payoffs(torch.from_numpy(data.windows(entries["selection"], 8)))
        self.assertTrue(torch.isfinite(predicted).all())
        data.features[370:] = 1e6
        data.net_returns[370:] = 1e6
        repeated, repeated_report = train_toric(data, entries, seq_len=8, epochs=1)
        self.assertEqual(report, repeated_report)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, repeated.state_dict()[name], rtol=0, atol=0)

    def test_toric_can_learn_a_known_synthetic_payoff_signal(self):
        torch.manual_seed(42)
        features = torch.randn(64, 4, 4)
        long_return = 0.01 * features[:, -1, 0]
        targets = torch.stack((long_return - 0.002, -long_return - 0.002), 1)
        model = ToricPayoffModel(4, max_len=4, dim_angles=8, num_states=4, dropout=0)
        model.fit_statistics(features.reshape(-1, 4), targets)
        normalized = (targets - model.target_mean) / model.target_std
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
        before = (model(features) - normalized).square().mean().item()
        for step in range(100):
            optimizer.zero_grad()
            loss = (model(features) - normalized).square().mean()
            loss.backward()
            optimizer.step()
        after = (model(features) - normalized).square().mean().item()
        self.assertLess(after, before * 0.1)

    def test_invalid_market_and_execution_are_rejected(self):
        with self.assertRaises(ValueError):
            ExecutionConfig(fee=float("nan"))
        with self.assertRaises(ValueError):
            ExecutionConfig(horizon=0)
        with self.assertRaises(ValueError):
            validate_market(market_frame(250).drop(200))
        with self.assertRaises(ValueError):
            validate_market(market_frame(250).iloc[::-1])
