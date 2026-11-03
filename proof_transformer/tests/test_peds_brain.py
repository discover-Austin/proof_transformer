from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import peds_brain  # noqa: E402
from proof_transformer.peds.brain import (  # noqa: E402
    append_brain_journal,
    claim_next_codex_task,
    choose_next_action,
    create_codex_task,
    derive_current_task_queue,
    evaluate_task_result,
    generate_codex_task,
    initialize_brain,
    list_codex_actuator_queue,
    list_task_queue,
    load_brain_state,
    load_latest_task_result,
    mark_task_running,
    observe_latest_evidence,
    propose_next_hypotheses,
    record_codex_task_result,
    record_decision,
    update_active_policy_from_survivor,
)


class PEDSBrainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_initialize_brain_creates_state_and_journals(self) -> None:
        state = initialize_brain(objective="Improve Codex patch reliability", root=self.root, max_iterations=3)

        self.assertEqual(state["system"], "PEDS")
        self.assertEqual(state["current_objective"], "Improve Codex patch reliability")
        self.assertTrue((self.root / "brain_state.json").exists())
        self.assertTrue((self.root / "brain_journal.jsonl").exists())
        self.assertTrue((self.root / "goals.jsonl").exists())
        self.assertTrue((self.root / "hypotheses.jsonl").exists())
        self.assertTrue((self.root / "decisions.jsonl").exists())
        self.assertTrue((self.root / "active_policy.jsonl").exists())
        self.assertTrue((self.root / "task_queue.jsonl").exists())
        self.assertTrue((self.root / "task_results.jsonl").exists())
        self.assertTrue((self.root / "codex_actuator_queue.jsonl").exists())
        self.assertTrue((self.root / "tasks").exists())
        self.assertTrue((self.root / "actuator_runs").exists())
        self.assertIn("last_codex_task", state)
        self.assertIn("last_codex_task_result", state)

    def test_append_brain_journal_writes_valid_jsonl(self) -> None:
        initialize_brain(objective="Improve Codex patch reliability", root=self.root)

        append_brain_journal("observation", "Observed a compact report.", evidence={"ok": True}, root=self.root)

        rows = [json.loads(line) for line in (self.root / "brain_journal.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(rows)
        self.assertEqual(rows[-1]["event_type"], "observation")
        self.assertEqual(rows[-1]["evidence"]["ok"], True)

    def test_observe_latest_evidence_summarizes_campaign_and_autopilot_reports(self) -> None:
        initialize_brain(objective="Improve Codex patch reliability", root=self.root)

        evidence = observe_latest_evidence(
            campaign_report=self._campaign_report(mutation_targets=["psi", "theta"]),
            autopilot_summary=self._autopilot_summary(active_policy=False),
            root=self.root,
        )

        self.assertTrue(evidence["has_campaign_report"])
        self.assertTrue(evidence["has_survivor"])
        self.assertEqual(evidence["campaign"]["total_winning_attempts"], 2)
        self.assertEqual(evidence["campaign"]["mutation_diversity"], 2)
        self.assertTrue(evidence["safety"]["clean"])
        self.assertIn("codex_task", evidence)
        self.assertIn("codex_task_result", evidence)
        self.assertIn("actuator_providers", evidence)
        self.assertIn("selected_provider", evidence["actuator_providers"])
        self.assertIn("agentic_capability", evidence)
        self.assertEqual(evidence["agentic_capability"]["status"], "not_reported")

    def test_propose_next_hypotheses_returns_deterministic_hypotheses(self) -> None:
        state = initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        evidence = observe_latest_evidence(
            campaign_report=self._campaign_report(mutation_targets=["psi"]),
            autopilot_summary=self._autopilot_summary(active_policy=False),
            root=self.root,
        )

        first = propose_next_hypotheses(state, evidence)
        second = propose_next_hypotheses(state, evidence)

        self.assertEqual(first, second)
        self.assertIn("mutation_diversity", {item["kind"] for item in first})
        self.assertIn("agentic_capability_curriculum", {item["kind"] for item in first})
        self.assertTrue(all(item.get("hypothesis_id") for item in first))

    def test_choose_next_action_picks_run_campaign_when_no_evidence_exists(self) -> None:
        state = initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        evidence = observe_latest_evidence(campaign_report=None, autopilot_summary=None, root=self.root)

        decision = choose_next_action(state, evidence, [])

        self.assertEqual(decision["action"], "run_campaign_5x2")

    def test_choose_next_action_stops_on_unsafe_evidence(self) -> None:
        state = initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        campaign = self._campaign_report(mutation_targets=["psi"])
        campaign["safety"] = {"network_attempts": ["http://example.invalid"], "model_modules_loaded": [], "fake_weights_written": False}
        evidence = observe_latest_evidence(
            campaign_report=campaign,
            autopilot_summary=self._autopilot_summary(active_policy=False),
            root=self.root,
        )

        decision = choose_next_action(state, evidence, [])

        self.assertEqual(decision["action"], "stop_and_report")
        self.assertEqual(decision["stop_reason"], "unsafe_evidence")

    def test_choose_next_action_activates_survivor_when_no_active_policy_exists(self) -> None:
        state = initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        evidence = observe_latest_evidence(
            campaign_report=self._campaign_report(mutation_targets=["psi", "theta"]),
            autopilot_summary=self._autopilot_summary(active_policy=False),
            root=self.root,
        )

        decision = choose_next_action(state, evidence, [])

        self.assertEqual(decision["action"], "activate_best_survivor")

    def test_external_autopilot_policy_does_not_count_as_brain_active_policy(self) -> None:
        state = initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        evidence = observe_latest_evidence(
            campaign_report=self._campaign_report(mutation_targets=["psi", "theta"]),
            autopilot_summary=self._autopilot_summary(active_policy=True),
            root=self.root,
        )

        decision = choose_next_action(state, evidence, [])

        self.assertFalse(evidence["active_policy"]["found"])
        self.assertTrue(evidence["external_active_policy"]["found"])
        self.assertEqual(decision["action"], "activate_best_survivor")

    def test_choose_next_action_picks_mutation_stress_for_high_success_narrow_diversity(self) -> None:
        state = initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        update_active_policy_from_survivor(self._profile(), root=self.root)
        evidence = observe_latest_evidence(
            campaign_report=self._campaign_report(mutation_targets=["psi"]),
            autopilot_summary=self._autopilot_summary(active_policy=True),
            root=self.root,
        )

        evidence["artifact_quality"]["status"] = "strong"
        evidence["active_policy_usage"]["campaign_start_supported"] = True
        decision = choose_next_action(state, evidence, [])

        self.assertEqual(decision["action"], "run_mutation_stress")

    def test_after_active_policy_exists_does_not_choose_activation_again(self) -> None:
        state = initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        update_active_policy_from_survivor(self._profile(), root=self.root)
        evidence = observe_latest_evidence(
            campaign_report=self._campaign_report(mutation_targets=["psi", "theta"]),
            autopilot_summary=self._autopilot_summary(active_policy=True),
            root=self.root,
        )

        decision = choose_next_action(state, evidence, [])

        self.assertNotEqual(decision["action"], "activate_best_survivor")

    def test_active_policy_start_path_wired_allows_success_decision(self) -> None:
        state = initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        update_active_policy_from_survivor(self._profile(), root=self.root)
        evidence = observe_latest_evidence(
            campaign_report=self._campaign_report(mutation_targets=["psi", "theta"]),
            autopilot_summary=self._autopilot_summary(active_policy=True),
            root=self.root,
        )
        evidence["artifact_quality"]["status"] = "strong"

        decision = choose_next_action(state, evidence, [])

        self.assertTrue(evidence["active_policy_usage"]["campaign_start_supported"])
        self.assertTrue(evidence["active_policy_usage"]["autopilot_start_supported"])
        self.assertFalse(evidence["active_policy_usage"]["needs_codex_task"])
        self.assertEqual(decision["action"], "stop_and_report")
        self.assertEqual(decision["stop_reason"], "success")

    def test_agentic_capability_gap_queues_curriculum_task_before_success(self) -> None:
        state = initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        update_active_policy_from_survivor(self._profile(), root=self.root)
        evidence = observe_latest_evidence(
            campaign_report=self._campaign_report(mutation_targets=["psi", "theta"]),
            autopilot_summary=self._autopilot_summary(active_policy=True, agentic_coverage=True),
            root=self.root,
        )
        evidence["artifact_quality"]["status"] = "strong"

        decision = choose_next_action(state, evidence, [])

        self.assertTrue(evidence["agentic_capability"]["needs_curriculum_task"])
        self.assertEqual(decision["action"], "expand_agentic_capability_curriculum")
        self.assertEqual(decision["hypothesis_kind"], "agentic_capability_curriculum")

    def test_accepted_agentic_task_result_updates_capability_coverage(self) -> None:
        state = initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        state["last_codex_task_result"] = {
            "task_result_id": "task_result_agentic",
            "status": "completed",
            "ok": True,
            "evidence": {
                "evidence_observed": {
                    "agentic_capability_action": "expand_agentic_capability_curriculum",
                    "dashboard_eval_passed": True,
                }
            },
        }
        peds_brain.write_brain_state(state, root=self.root)
        summary = self._autopilot_summary(active_policy=True, agentic_coverage=True)
        summary["agentic_capability_coverage"]["pressured_capabilities"] = ["repo_navigation"]
        summary["agentic_capability_coverage"]["untested_capabilities"] = ["ui_or_browser_operation"]

        evidence = observe_latest_evidence(
            campaign_report=self._campaign_report(mutation_targets=["psi", "theta"]),
            autopilot_summary=summary,
            root=self.root,
        )

        self.assertEqual(evidence["agentic_capability"]["status"], "broad_curriculum_pressured")
        self.assertFalse(evidence["agentic_capability"]["needs_curriculum_task"])
        self.assertIn("ui_or_browser_operation", evidence["agentic_capability"]["pressured_capabilities"])
        self.assertEqual(evidence["agentic_capability"]["latest_curriculum_task_result_id"], "task_result_agentic")

    def test_active_policy_with_unknown_artifact_quality_picks_artifact_inspection(self) -> None:
        state = initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        update_active_policy_from_survivor(self._profile(), root=self.root)
        evidence = observe_latest_evidence(
            campaign_report=self._campaign_report(mutation_targets=["psi", "theta"]),
            autopilot_summary=self._autopilot_summary(active_policy=True),
            root=self.root,
        )

        decision = choose_next_action(state, evidence, [])

        self.assertEqual(evidence["artifact_quality"]["status"], "unknown")
        self.assertEqual(decision["action"], "inspect_artifact_quality")

    def test_active_policy_with_missing_useful_outputs_generates_outputs(self) -> None:
        state = initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        update_active_policy_from_survivor(self._profile(), root=self.root)
        evidence = observe_latest_evidence(
            campaign_report=self._campaign_report(mutation_targets=["psi", "theta"]),
            autopilot_summary=self._autopilot_summary(active_policy=True, useful=False),
            root=self.root,
        )

        decision = choose_next_action(state, evidence, [])

        self.assertFalse(evidence["useful_artifacts"])
        self.assertEqual(decision["action"], "generate_useful_outputs")

    def test_record_decision_appends_decisions(self) -> None:
        initialize_brain(objective="Improve Codex patch reliability", root=self.root)

        record = record_decision({"action": "run_campaign_5x2", "reason": "missing evidence"}, root=self.root)

        rows = (self.root / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 1)
        self.assertEqual(json.loads(rows[0])["decision_id"], record["decision_id"])

    def test_update_active_policy_from_survivor_writes_append_only_policy(self) -> None:
        initialize_brain(objective="Improve Codex patch reliability", root=self.root)

        first = update_active_policy_from_survivor(self._profile(), root=self.root)
        second = update_active_policy_from_survivor(self._profile(candidate_id="candidate_two"), root=self.root)
        state = load_brain_state(self.root)

        lines = (self.root / "active_policy.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(state["current_active_policy_id"], second["active_policy_id"])
        self.assertEqual(first["candidate_id"], "candidate_best")

    def test_brain_campaign_command_passes_active_policy_file(self) -> None:
        initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        update_active_policy_from_survivor(self._profile(), root=self.root)

        command = peds_brain._command_for_action("run_campaign_5x2", "Improve Codex patch reliability", self.root)

        self.assertIn("--active-policy-file", command)
        self.assertIn(str(self.root / "active_policy.jsonl"), command)

    def test_cli_activate_profile_updates_brain_active_policy(self) -> None:
        initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        profile_path = self.root / "survivor_profile.json"
        profile_path.write_text(json.dumps(self._profile(candidate_id="candidate_new")), encoding="utf-8")

        payload = self._run_cli("--activate-profile", str(profile_path))
        state = load_brain_state(self.root)

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["active_policy"]["candidate_id"], "candidate_new")
        self.assertEqual(state["current_active_candidate_id"], "candidate_new")
        self.assertEqual(len((self.root / "active_policy.jsonl").read_text(encoding="utf-8").splitlines()), 1)

    def test_cli_status_returns_json(self) -> None:
        initialize_brain(objective="Improve Codex patch reliability", root=self.root)

        payload = self._run_cli("--status")

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["state"]["system"], "PEDS")

    def test_cli_think_returns_decision_json(self) -> None:
        initialize_brain(objective="Improve Codex patch reliability", root=self.root)

        payload = self._run_cli("--think")

        self.assertTrue(payload["ok"])
        self.assertIn("decision", payload)
        self.assertIn(payload["decision"]["action"], {"run_campaign_5x2", "run_autopilot_use_latest", "activate_best_survivor", "run_mutation_stress", "stop_and_report"})

    def test_code_editing_action_generates_codex_task_artifact(self) -> None:
        initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        decision = {"action": "inspect_artifact_quality", "reason": "Artifacts are weak.", "hypothesis_kind": "artifact_quality"}
        evidence = observe_latest_evidence(
            campaign_report=self._campaign_report(mutation_targets=["psi", "theta"]),
            autopilot_summary=self._autopilot_summary(active_policy=True, weak_prompt=True),
            root=self.root,
        )
        hypotheses = propose_next_hypotheses(load_brain_state(self.root), evidence)

        result = peds_brain.execute_decision(decision=decision, evidence=evidence, hypotheses=hypotheses, root=self.root)

        self.assertTrue(result["ok"])
        path = Path(result["codex_task"]["task_path"])
        self.assertTrue(path.exists())
        self.assertEqual(load_brain_state(self.root)["last_codex_task"], str(path))
        queue = list_codex_actuator_queue(root=self.root)
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["status"], "pending")
        self.assertEqual(queue[0]["task_path"], str(path))
        self.assertNotEqual(path.name, "_codex_task.md")
        self.assertRegex(path.name, r"\d{8}T\d{12}Z_inspect_artifact_quality_task_[0-9a-f]{16}_codex_task\.md")
        content = path.read_text(encoding="utf-8")
        for heading in [
            "# PEDS-Decided Codex Task",
            "## Brain objective",
            "## Evidence observed",
            "## Hypothesis",
            "## Decision",
            "## Task",
            "## Files likely touched",
            "## Constraints",
            "## Validation commands",
            "## Expected useful output",
            "## Final report contract",
        ]:
            self.assertIn(heading, content)

    def test_agentic_curriculum_action_generates_broad_capability_task(self) -> None:
        initialize_brain(objective="Improve all agentic capabilities", root=self.root)
        decision = {
            "action": "expand_agentic_capability_curriculum",
            "reason": "Broad capability coverage is incomplete.",
            "hypothesis_kind": "agentic_capability_curriculum",
        }
        evidence = observe_latest_evidence(
            campaign_report=self._campaign_report(mutation_targets=["psi", "theta"]),
            autopilot_summary=self._autopilot_summary(active_policy=True, agentic_coverage=True),
            root=self.root,
        )
        hypotheses = propose_next_hypotheses(load_brain_state(self.root), evidence)

        result = peds_brain.execute_decision(decision=decision, evidence=evidence, hypotheses=hypotheses, root=self.root)
        content = Path(result["codex_task"]["task_path"]).read_text(encoding="utf-8")

        self.assertTrue(result["ok"])
        self.assertEqual(result["action"], "expand_agentic_capability_curriculum")
        self.assertIn("broad agentic capabilities", content)
        self.assertIn("src/proof_transformer/peds/survivor_outputs.py", content)

    def test_prior_task_without_result_waits_for_codex_task_result(self) -> None:
        state = initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        update_active_policy_from_survivor(self._profile(), root=self.root)
        decision = {"action": "inspect_artifact_quality", "reason": "Artifacts are unknown.", "hypothesis_kind": "artifact_quality"}
        evidence = observe_latest_evidence(
            campaign_report=self._campaign_report(mutation_targets=["psi", "theta"]),
            autopilot_summary=self._autopilot_summary(active_policy=True),
            root=self.root,
        )
        generate_codex_task(decision=decision, evidence=evidence, root=self.root)

        observed = observe_latest_evidence(
            campaign_report=self._campaign_report(mutation_targets=["psi", "theta"]),
            autopilot_summary=self._autopilot_summary(active_policy=True),
            root=self.root,
        )
        next_decision = choose_next_action(state, observed, [])

        self.assertTrue(observed["codex_task"]["pending_result"])
        self.assertEqual(next_decision["action"], "wait_for_codex_task_result")

    def test_wait_action_backfills_codex_queue_for_existing_task(self) -> None:
        initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        task_path = self.root / "tasks" / "20260607T000000000000Z_inspect_artifact_quality_codex_task.md"
        task_path.parent.mkdir(parents=True, exist_ok=True)
        task_path.write_text("# PEDS-Decided Codex Task\n", encoding="utf-8")
        decision = {"action": "wait_for_codex_task_result", "task_path": str(task_path), "reason": "pending", "hypothesis_kind": "artifact_quality"}

        result = peds_brain.execute_decision(decision=decision, evidence={}, hypotheses=[], root=self.root)
        queue = list_codex_actuator_queue(root=self.root)

        self.assertTrue(result["ok"])
        self.assertEqual(queue[-1]["status"], "pending")
        self.assertEqual(queue[-1]["task_path"], str(task_path))

    def test_codex_can_claim_next_queued_task(self) -> None:
        initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        decision = {"action": "inspect_artifact_quality", "reason": "Artifacts are unknown.", "hypothesis_kind": "artifact_quality"}
        evidence = observe_latest_evidence(
            campaign_report=self._campaign_report(mutation_targets=["psi", "theta"]),
            autopilot_summary=self._autopilot_summary(active_policy=True),
            root=self.root,
        )
        generated = generate_codex_task(decision=decision, evidence=evidence, root=self.root)

        claimed = claim_next_codex_task(root=self.root)
        queue = list_codex_actuator_queue(root=self.root)

        self.assertIsNotNone(claimed)
        self.assertEqual(claimed["task_path"], generated["task_path"])
        self.assertEqual(queue[-1]["status"], "claimed")

    def test_mark_running_and_current_queue_view_separates_completed_from_pending(self) -> None:
        initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        decision = {"action": "inspect_artifact_quality", "reason": "Artifacts are unknown.", "hypothesis_kind": "artifact_quality"}
        evidence = observe_latest_evidence(
            campaign_report=self._campaign_report(mutation_targets=["psi", "theta"]),
            autopilot_summary=self._autopilot_summary(active_policy=True),
            root=self.root,
        )
        generated = create_codex_task(decision=decision, evidence=evidence, root=self.root)

        claimed = claim_next_codex_task(root=self.root)
        running = mark_task_running(claimed, root=self.root)
        result_path = self.root / "task_result.json"
        result_path.write_text(json.dumps({"task_id": generated["task_id"], "task_path": generated["task_path"], "status": "completed", "ok": True, "summary": "done"}), encoding="utf-8")
        record = record_codex_task_result(result_path, root=self.root)
        queue = derive_current_task_queue(root=self.root)

        self.assertEqual(running["status"], "running")
        self.assertEqual(record["status"], "completed")
        self.assertEqual(queue[0]["status"], "completed")
        self.assertEqual([item for item in queue if item["status"] == "pending"], [])
        self.assertEqual(list_task_queue(root=self.root), queue)
        self.assertEqual(load_latest_task_result(root=self.root)["task_id"], generated["task_id"])
        self.assertEqual(evaluate_task_result(record)["action"], "accept_result")

    def test_manual_task_result_state_is_not_success(self) -> None:
        initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        update_active_policy_from_survivor(self._profile(), root=self.root)
        decision = {"action": "inspect_artifact_quality", "reason": "Artifacts are unknown.", "hypothesis_kind": "artifact_quality"}
        evidence = observe_latest_evidence(
            campaign_report=self._campaign_report(mutation_targets=["psi", "theta"]),
            autopilot_summary=self._autopilot_summary(active_policy=True),
            root=self.root,
        )
        generated = generate_codex_task(decision=decision, evidence=evidence, root=self.root)
        result_path = self.root / "manual_result.json"
        result_path.write_text(json.dumps({"task_id": generated["task_id"], "task_path": generated["task_path"], "status": "manual_actuator_required", "summary": "manual needed", "manual_command": "codex exec ..."}), encoding="utf-8")

        record = record_codex_task_result(result_path, root=self.root)
        queue = derive_current_task_queue(root=self.root)
        decision = choose_next_action(load_brain_state(self.root), observe_latest_evidence(campaign_report=self._campaign_report(mutation_targets=["psi", "theta"]), autopilot_summary=self._autopilot_summary(active_policy=True), root=self.root), [])

        self.assertEqual(record["status"], "manual_actuator_required")
        self.assertFalse(record["ok"])
        self.assertEqual(queue[0]["status"], "manual_actuator_required")
        self.assertEqual(decision["action"], "manual_actuator_required")

    def test_later_manual_result_ingestion_updates_queue_and_continues(self) -> None:
        initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        update_active_policy_from_survivor(self._profile(), root=self.root)
        decision = {"action": "inspect_artifact_quality", "reason": "Artifacts are unknown.", "hypothesis_kind": "artifact_quality"}
        evidence = observe_latest_evidence(
            campaign_report=self._campaign_report(mutation_targets=["psi", "theta"]),
            autopilot_summary=self._autopilot_summary(active_policy=True),
            root=self.root,
        )
        generated = generate_codex_task(decision=decision, evidence=evidence, root=self.root)
        manual_path = self.root / "manual_required.json"
        manual_path.write_text(json.dumps({"task_id": generated["task_id"], "task_path": generated["task_path"], "status": "manual_actuator_required", "ok": False, "summary": "manual needed", "manual_command": "codex exec ..."}), encoding="utf-8")
        record_codex_task_result(manual_path, root=self.root)
        self.assertEqual(derive_current_task_queue(root=self.root)[0]["status"], "manual_actuator_required")

        completed_path = self.root / "manual_completed.json"
        completed_path.write_text(json.dumps({"task_id": generated["task_id"], "task_path": generated["task_path"], "status": "completed", "ok": True, "summary": "manual task completed with validation"}), encoding="utf-8")
        completed = record_codex_task_result(completed_path, root=self.root)
        observed = observe_latest_evidence(campaign_report=self._campaign_report(mutation_targets=["psi", "theta"]), autopilot_summary=self._autopilot_summary(active_policy=True), root=self.root)
        next_decision = choose_next_action(load_brain_state(self.root), observed, [])

        self.assertEqual(completed["status"], "completed")
        self.assertEqual(derive_current_task_queue(root=self.root)[0]["status"], "completed")
        self.assertEqual(next_decision["action"], "accept_result")

    def test_execute_actuator_true_selects_execute_actuator_task_for_pending_task(self) -> None:
        state = initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        update_active_policy_from_survivor(self._profile(), root=self.root)
        decision = {"action": "inspect_artifact_quality", "reason": "Artifacts are unknown.", "hypothesis_kind": "artifact_quality"}
        evidence = observe_latest_evidence(
            campaign_report=self._campaign_report(mutation_targets=["psi", "theta"]),
            autopilot_summary=self._autopilot_summary(active_policy=True),
            root=self.root,
        )
        generated = generate_codex_task(decision=decision, evidence=evidence, root=self.root)
        observed = observe_latest_evidence(campaign_report=self._campaign_report(mutation_targets=["psi", "theta"]), autopilot_summary=self._autopilot_summary(active_policy=True), root=self.root)

        next_decision = choose_next_action(state, observed, [], execute_actuator=True)

        self.assertEqual(next_decision["action"], "execute_actuator_task")
        self.assertEqual(next_decision["task_path"], generated["task_path"])

    def test_duplicate_queue_events_do_not_create_contradictory_current_state(self) -> None:
        initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        task_path = self.root / "tasks" / "duplicate_codex_task.md"
        task_path.write_text("# task", encoding="utf-8")
        rows = [
            {"task_id": "task_duplicate", "queue_id": "task_duplicate", "timestamp": "2026-06-07T00:00:00Z", "event": "queued", "status": "pending", "task_path": str(task_path), "action": "inspect_artifact_quality"},
            {"task_id": "task_duplicate", "queue_id": "task_duplicate", "timestamp": "2026-06-07T00:00:01Z", "event": "queued", "status": "pending", "task_path": str(task_path), "action": "inspect_artifact_quality"},
            {"task_id": "task_duplicate", "queue_id": "task_duplicate", "timestamp": "2026-06-07T00:00:02Z", "event": "completed", "status": "completed", "task_path": str(task_path), "action": "inspect_artifact_quality"},
        ]
        (self.root / "task_queue.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

        queue = derive_current_task_queue(root=self.root)

        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["status"], "completed")
        self.assertEqual([item for item in queue if item["status"] == "pending"], [])

    def test_legacy_codex_queue_compatibility_does_not_create_pending_after_completed(self) -> None:
        initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        (self.root / "task_queue.jsonl").unlink()
        task_path = self.root / "tasks" / "legacy_codex_task.md"
        task_path.write_text("# task", encoding="utf-8")
        rows = [
            {"queue_id": "legacy_task", "timestamp": "2026-06-07T00:00:00Z", "event": "queued", "status": "pending", "task_path": str(task_path), "action": "inspect_artifact_quality"},
            {"queue_id": "legacy_task", "timestamp": "2026-06-07T00:00:02Z", "event": "completed", "status": "completed", "task_path": str(task_path), "action": "inspect_artifact_quality"},
        ]
        (self.root / "codex_actuator_queue.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

        queue = derive_current_task_queue(root=self.root)

        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["task_id"], "legacy_task")
        self.assertEqual(queue[0]["status"], "completed")

    def test_task_result_can_be_recorded_and_observed(self) -> None:
        initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        update_active_policy_from_survivor(self._profile(), root=self.root)
        decision = {"action": "inspect_artifact_quality", "reason": "Artifacts are unknown.", "hypothesis_kind": "artifact_quality"}
        evidence = observe_latest_evidence(
            campaign_report=self._campaign_report(mutation_targets=["psi", "theta"]),
            autopilot_summary=self._autopilot_summary(active_policy=True),
            root=self.root,
        )
        generate_codex_task(decision=decision, evidence=evidence, root=self.root)
        result_path = self.root / "task_result.json"
        result_path.write_text(json.dumps({"ok": True, "summary": "Task completed with validation evidence."}), encoding="utf-8")

        record = record_codex_task_result(result_path, root=self.root)
        observed = observe_latest_evidence(
            campaign_report=self._campaign_report(mutation_targets=["psi", "theta"]),
            autopilot_summary=self._autopilot_summary(active_policy=True),
            root=self.root,
        )
        next_decision = choose_next_action(load_brain_state(self.root), observed, [])

        self.assertEqual(record["status"], "completed")
        self.assertEqual(list_codex_actuator_queue(root=self.root)[-1]["status"], "completed")
        self.assertTrue(observed["codex_task_result"]["found"])
        self.assertFalse(observed["codex_task"]["pending_result"])
        self.assertEqual(next_decision["action"], "accept_result")

    def test_cli_record_task_result_returns_json(self) -> None:
        initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        result_path = self.root / "task_result.md"
        result_path.write_text("Summary: completed and ready for PEDS review.", encoding="utf-8")

        payload = self._run_cli("--record-task-result", str(result_path))

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["task_result"]["status"], "completed")
        self.assertEqual(payload["state"]["last_codex_task_result"]["path"], str(result_path))

    def test_cli_claim_codex_task_returns_queued_task(self) -> None:
        initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        decision = {"action": "inspect_artifact_quality", "reason": "Artifacts are unknown.", "hypothesis_kind": "artifact_quality"}
        evidence = observe_latest_evidence(
            campaign_report=self._campaign_report(mutation_targets=["psi", "theta"]),
            autopilot_summary=self._autopilot_summary(active_policy=True),
            root=self.root,
        )
        generated = generate_codex_task(decision=decision, evidence=evidence, root=self.root)

        payload = self._run_cli("--claim-codex-task")

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["codex_task"]["task_path"], generated["task_path"])

    def test_code_editing_action_generates_unique_task_artifacts(self) -> None:
        initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        decision = {"action": "inspect_artifact_quality", "reason": "Artifacts are weak.", "hypothesis_kind": "artifact_quality"}
        evidence = observe_latest_evidence(
            campaign_report=self._campaign_report(mutation_targets=["psi", "theta"]),
            autopilot_summary=self._autopilot_summary(active_policy=True, weak_prompt=True),
            root=self.root,
        )
        hypotheses = propose_next_hypotheses(load_brain_state(self.root), evidence)

        first = peds_brain.execute_decision(decision=decision, evidence=evidence, hypotheses=hypotheses, root=self.root)
        second = peds_brain.execute_decision(decision=decision, evidence=evidence, hypotheses=hypotheses, root=self.root)

        self.assertNotEqual(first["codex_task"]["task_path"], second["codex_task"]["task_path"])
        self.assertEqual(len(list((self.root / "tasks").glob("*_codex_task.md"))), 2)

    def test_brain_journal_records_post_activation_decision(self) -> None:
        initialize_brain(objective="Improve Codex patch reliability", root=self.root)
        update_active_policy_from_survivor(self._profile(), root=self.root)

        record_decision({"action": "inspect_artifact_quality", "reason": "Artifact quality is unknown.", "hypothesis_kind": "artifact_quality"}, root=self.root)

        rows = [json.loads(line) for line in (self.root / "brain_journal.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any(row["event_type"] == "decision" and (row.get("decision") or {}).get("action") == "inspect_artifact_quality" for row in rows))

    def test_step_updates_brain_state_last_decision_and_next_action(self) -> None:
        initialize_brain(objective="Improve Codex patch reliability", root=self.root, max_iterations=5)
        update_active_policy_from_survivor(self._profile(), root=self.root)
        original_campaign_loader = peds_brain.build_latest_campaign_report
        original_autopilot_loader = peds_brain.load_latest_autopilot_summary
        peds_brain.build_latest_campaign_report = lambda: self._campaign_report(mutation_targets=["psi", "theta"])
        peds_brain.load_latest_autopilot_summary = lambda: self._autopilot_summary(active_policy=True)
        try:
            result = peds_brain.run_brain_step(root=self.root)
        finally:
            peds_brain.build_latest_campaign_report = original_campaign_loader
            peds_brain.load_latest_autopilot_summary = original_autopilot_loader

        state = load_brain_state(self.root)
        self.assertTrue(result["ok"])
        self.assertEqual(result["decision"]["action"], "inspect_artifact_quality")
        self.assertEqual(state["last_decision_id"], result["decision"]["decision_id"])
        self.assertEqual(state["next_action"]["action"], "inspect_artifact_quality")

    def test_max_iterations_prevents_unbounded_loops(self) -> None:
        initialize_brain(objective="Improve Codex patch reliability", root=self.root, max_iterations=0)

        result = peds_brain.run_brain_loop(root=self.root, max_iterations=5)

        self.assertTrue(result["ok"])
        self.assertEqual(result["steps_run"], 1)
        self.assertEqual(result["state"]["stop_reason"], "max_iterations_reached")

    def _run_cli(self, *args: str) -> dict:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "peds_brain.py"), *args, "--root", str(self.root), "--json"],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, "PYTHONPATH": str(SRC)},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        return json.loads(completed.stdout)

    def _campaign_report(self, *, mutation_targets: list[str]) -> dict:
        mutations = [{"mutation_target": target, "diagnosis_category": "no_generated_child"} for target in mutation_targets]
        return {
            "found": True,
            "campaign_id": "campaign_brain",
            "report_path": str(self.root / "campaign_report.json"),
            "campaign_root": str(self.root / "campaign"),
            "aggregate": {
                "total_generations_completed": 2,
                "total_candidate_attempts": 4,
                "total_winning_attempts": 2,
                "total_rejected_attempts": 2,
                "corrections_attempted": 2,
                "corrections_followed_by_winner": 2,
                "correction_success_rate": 1.0,
                "collapse_detected": False,
                "trained_rate": 1.0,
                "adapter_loaded_rate": 1.0,
            },
            "mutations": mutations,
            "safety": {"network_attempts": [], "model_modules_loaded": [], "fake_weights_written": False},
        }

    def _autopilot_summary(
        self,
        *,
        active_policy: bool,
        weak_prompt: bool = False,
        useful: bool = True,
        agentic_coverage: bool = False,
    ) -> dict:
        policy = {"active_policy_id": "policy_one", "candidate_id": "candidate_best"} if active_policy else None
        summary = {
            "ok": True,
            "summary_path": str(self.root / "autopilot_summary.json"),
            "output_dir": str(self.root / "outputs") if useful else None,
            "candidate_id": "candidate_best",
            "survivor_selected": True,
            "useful_output_ready": useful,
            "artifacts_written": useful,
            "artifacts": {"codex_prompt": str(self.root / "outputs" / "codex_prompt.txt")} if useful else {},
            "active_policy": policy,
            "latest_active_policy": policy,
            "codex_prompt_preview": "short" if weak_prompt else "A concrete PEDS survivor prompt with enough detail to guide local repo work." * 5,
            "safety": {"network_attempts": [], "model_modules_loaded": [], "fake_weights_written": False},
        }
        if agentic_coverage:
            summary["agentic_capability_coverage"] = {
                "status": "broadening_needed",
                "intent": "Improve agentic capability across task classes, not only PEDS self-improvement mechanics.",
                "pressured_capabilities": ["code_editing", "repo_navigation"],
                "untested_capabilities": ["tool_orchestration", "ui_or_browser_operation"],
                "next_eval_tasks": [
                    "Add a local tool_orchestration eval with task queue evidence.",
                    "Add a local ui_or_browser_operation eval with dashboard evidence.",
                ],
                "coverage_warning": "Do not claim all-agentic improvement until each capability has local verifier evidence.",
            }
            summary["agentic_next_eval_tasks"] = summary["agentic_capability_coverage"]["next_eval_tasks"]
        return summary

    def _profile(self, *, candidate_id: str = "candidate_best") -> dict:
        return {
            "candidate_id": candidate_id,
            "objective": "Improve Codex patch reliability",
            "selection_reason": "latest winning attempt with clean safety",
            "safety": {"network_attempts": [], "model_modules_loaded": [], "fake_weights_written": False},
            "evidence_summary": {"total_winning_attempts": 2},
        }


if __name__ == "__main__":
    unittest.main()
