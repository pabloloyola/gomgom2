#!/usr/bin/env python3
"""Run the pre-refinement fixed-pair LLM simulator transfer gate.

This is the first script that requires a real OpenAI-compatible model endpoint.
It deliberately disables response caching so repeat-consistency is measured from
independent requests rather than cache hits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from eipg.experiments.llm_transfer import TransferGateThresholds, evaluate_transfer_gate
from eipg.simulators.llm_choice import TextPersona, TextPersonaChoiceSimulator
from eipg.simulators.openai_compatible import OpenAICompatibleChatClient, OpenAICompatibleConfig


def _slate(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def diagnostic_pairs() -> list[tuple[str, TextPersona, pd.DataFrame, int]]:
    """Return fixed, deliberately clear persona-slate pairs and expected direction."""
    budget = TextPersona(
        "budget",
        "budget_sensitive",
        "A strongly price-conscious shopper. When products are broadly comparable, they usually choose the lowest-priced option and need a clear benefit to pay more.",
    )
    quality = TextPersona(
        "quality",
        "quality_oriented",
        "A quality-oriented shopper who willingly pays a moderate premium for clearly higher quality and reliability, while still noticing price.",
    )
    sustain = TextPersona(
        "sustain",
        "sustainability_oriented",
        "A sustainability-oriented shopper who strongly values environmentally responsible products and accepts a moderate price premium for a clearly more sustainable option.",
    )

    common = [
        {"context_id": "price_clear", "alternative_id": 0, "price": 1.0, "quality": 0.60, "sustain": 0.50, "novelty": 0.40, "brand": 0.0},
        {"context_id": "price_clear", "alternative_id": 1, "price": 1.8, "quality": 0.62, "sustain": 0.50, "novelty": 0.40, "brand": 0.0},
        {"context_id": "price_clear", "alternative_id": 2, "price": 2.4, "quality": 0.61, "sustain": 0.50, "novelty": 0.40, "brand": 0.0},
    ]
    quality_upgrade = [
        {"context_id": "quality_upgrade", "alternative_id": 0, "price": 1.5, "quality": 0.35, "sustain": 0.50, "novelty": 0.40, "brand": 0.0},
        {"context_id": "quality_upgrade", "alternative_id": 1, "price": 1.8, "quality": 0.95, "sustain": 0.50, "novelty": 0.40, "brand": 0.0},
        {"context_id": "quality_upgrade", "alternative_id": 2, "price": 2.3, "quality": 0.55, "sustain": 0.50, "novelty": 0.40, "brand": 0.0},
    ]
    sustain_upgrade = [
        {"context_id": "sustain_upgrade", "alternative_id": 0, "price": 1.5, "quality": 0.65, "sustain": 0.15, "novelty": 0.40, "brand": 0.0},
        {"context_id": "sustain_upgrade", "alternative_id": 1, "price": 1.8, "quality": 0.65, "sustain": 0.95, "novelty": 0.40, "brand": 0.0},
        {"context_id": "sustain_upgrade", "alternative_id": 2, "price": 2.3, "quality": 0.65, "sustain": 0.55, "novelty": 0.40, "brand": 0.0},
    ]
    premium_too_large = [
        {"context_id": "premium_too_large", "alternative_id": 0, "price": 1.0, "quality": 0.65, "sustain": 0.45, "novelty": 0.40, "brand": 0.0},
        {"context_id": "premium_too_large", "alternative_id": 1, "price": 4.0, "quality": 0.72, "sustain": 0.55, "novelty": 0.40, "brand": 0.0},
        {"context_id": "premium_too_large", "alternative_id": 2, "price": 4.5, "quality": 0.74, "sustain": 0.58, "novelty": 0.40, "brand": 0.0},
    ]
    return [
        ("budget_price_clear", budget, _slate(common), 0),
        ("budget_extreme_premium", budget, _slate(premium_too_large), 0),
        ("quality_upgrade", quality, _slate(quality_upgrade), 1),
        ("sustain_upgrade", sustain, _slate(sustain_upgrade), 1),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/llm_prompt_refinement.yaml")
    parser.add_argument("--model", default=None, help="Override backend.model for matrix runs")
    parser.add_argument("--provider", default=None, help="Pin one OpenRouter provider slug with no fallbacks")
    parser.add_argument("--output", default="outputs/llm_transfer_gate.json")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    backend_cfg = dict(cfg["backend"])
    if args.model:
        backend_cfg["model"] = args.model
    # Metadata-only config fields are not part of the generic client schema.
    backend_cfg.pop("models", None)
    backend_cfg.pop("provider_routing", None)
    backend_cfg.pop("providers", None)
    if backend_cfg["model"] == "TO_BE_PINNED_BEFORE_RUN":
        raise SystemExit("Pin backend.model in the config before running the live transfer gate.")
    if args.provider and args.provider.lower() not in {"openai", "direct", "native", "local", "vllm", "lmstudio"}:
        backend_cfg["extra_body"] = {
            "provider": {"order": [args.provider], "allow_fallbacks": False}
        }
    # Repeat-consistency must measure fresh model calls, not exact-prompt cache hits.
    backend_cfg["cache_dir"] = None
    client = OpenAICompatibleChatClient(OpenAICompatibleConfig.from_config(backend_cfg))
    simulator = TextPersonaChoiceSimulator(client)

    repeats = int(cfg["transfer_gate"]["repeats_per_pair"])
    request_rows: list[dict] = []
    choice_rows: list[dict] = []
    expected_by_pair: dict[str, int] = {}

    for pair_id, persona, slate, expected in diagnostic_pairs():
        expected_by_pair[pair_id] = expected
        for repeat_id in range(repeats):
            try:
                result = simulator.choose(slate=slate, persona=persona)
                request_rows.append({"pair_id": pair_id, "repeat_id": repeat_id, "parse_ok": True})
                frame = slate.copy()
                frame["pair_id"] = pair_id
                frame["repeat_id"] = repeat_id
                frame["persona_id"] = persona.persona_id
                frame["chosen"] = (frame["alternative_id"].astype(int) == result.alternative_id).astype(int)
                choice_rows.extend(frame.to_dict("records"))
            except Exception as exc:  # diagnostic should record failures instead of stopping early
                request_rows.append(
                    {
                        "pair_id": pair_id,
                        "repeat_id": repeat_id,
                        "parse_ok": False,
                        "error": repr(exc),
                    }
                )

    requests = pd.DataFrame(request_rows)
    choices = pd.DataFrame(choice_rows)
    if choices.empty:
        payload = {
            "model": backend_cfg["model"],
            "provider": args.provider,
            "repeats_per_pair": repeats,
            "passed": False,
            "reason": "no_successful_choice_records",
            "request_failures": request_rows,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit("LLM simulator transfer gate failed: no successful choice records.")

    direction_checks = []
    if not choices.empty:
        chosen = choices.loc[choices["chosen"].astype(int) == 1]
        for pair_id, group in chosen.groupby("pair_id"):
            modal = int(group["alternative_id"].astype(int).value_counts().idxmax())
            direction_checks.append(modal == expected_by_pair[pair_id])

    gate_cfg = cfg["transfer_gate"]
    report = evaluate_transfer_gate(
        request_log=requests,
        choice_records=choices,
        direction_checks=direction_checks,
        thresholds=TransferGateThresholds(
            min_parse_success=float(gate_cfg["min_parse_success"]),
            min_repeat_consistency=float(gate_cfg["min_repeat_consistency"]),
            min_directional_accuracy=float(gate_cfg["min_directional_accuracy"]),
        ),
    )
    payload = report.to_dict()
    payload["model"] = backend_cfg["model"]
    payload["provider"] = args.provider
    payload["repeats_per_pair"] = repeats
    payload["direction_checks"] = direction_checks
    failures = requests.loc[~requests["parse_ok"].astype(bool)] if not requests.empty else pd.DataFrame()
    payload["request_failures"] = (
        failures[["pair_id", "repeat_id", "error"]].to_dict("records") if not failures.empty else []
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if gate_cfg.get("proceed_to_refinement_only_if_passed", True) and not report.passed:
        raise SystemExit("LLM simulator transfer gate failed; do not proceed to prompt refinement.")


if __name__ == "__main__":
    main()
