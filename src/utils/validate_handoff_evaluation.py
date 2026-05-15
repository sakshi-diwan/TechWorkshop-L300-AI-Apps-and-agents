import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv


def load_dataset(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc

            if "id" not in row or "query" not in row or "expected_domain" not in row:
                raise ValueError(
                    f"Missing required keys at line {line_number}. "
                    "Expected: id, query, expected_domain"
                )
            rows.append(row)

    if not rows:
        raise ValueError("Dataset is empty")

    return rows


def get_predicted_domain(openai_client: Any, agent_name: str, query: str) -> Tuple[str, str]:
    conversation = openai_client.conversations.create(
        items=[{"type": "message", "role": "user", "content": query}]
    )

    response = openai_client.responses.create(
        conversation=conversation.id,
        extra_body={
            "agent_reference": {"name": agent_name, "type": "agent_reference"}
        },
        input="",
    )

    output_text = response.output_text or ""
    payload = json.loads(output_text)
    return str(payload.get("domain", "")).strip(), output_text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate handoff evaluation dataset against live Foundry handoff-service output"
    )
    parser.add_argument(
        "--dataset",
        default="data/handoff_service_evaluation_grounded.jsonl",
        help="Path to grounded JSONL dataset",
    )
    parser.add_argument(
        "--agent",
        default="handoff-service",
        help="Foundry agent name",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Limit number of rows for quick checks (0 = all rows)",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.0,
        help="Fail if exact-match accuracy is lower than this value (0.0 to 1.0)",
    )
    args = parser.parse_args()

    load_dotenv()
    endpoint = os.environ.get("FOUNDRY_ENDPOINT")
    if not endpoint:
        print("ERROR: FOUNDRY_ENDPOINT is not set")
        return 2

    try:
        rows = load_dataset(args.dataset)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2

    if args.max_cases > 0:
        rows = rows[: args.max_cases]

    project_client = AIProjectClient(endpoint=endpoint, credential=DefaultAzureCredential())
    openai_client = project_client.get_openai_client()

    matches = 0
    runtime_errors = 0
    malformed_outputs = 0
    mismatches: List[Tuple[str, str, str]] = []

    for row in rows:
        case_id = str(row["id"])
        expected = str(row["expected_domain"]).strip()
        query = str(row["query"])

        try:
            predicted, raw_output = get_predicted_domain(openai_client, args.agent, query)
        except json.JSONDecodeError as exc:
            malformed_outputs += 1
            print(f"MALFORMED_OUTPUT id={case_id} error={exc}")
            continue
        except Exception as exc:
            runtime_errors += 1
            print(f"RUNTIME_ERROR id={case_id} error={type(exc).__name__}: {exc}")
            continue

        if predicted == expected:
            matches += 1
        else:
            mismatches.append((case_id, expected, predicted))
            print(f"MISMATCH id={case_id} expected={expected} predicted={predicted}")
            if not raw_output:
                print(f"EMPTY_OUTPUT id={case_id}")

    total = len(rows)
    accuracy = (matches / total) if total else 0.0

    print("\nSUMMARY")
    print(f"cases={total}")
    print(f"matches={matches}")
    print(f"accuracy={accuracy:.3f}")
    print(f"mismatches={len(mismatches)}")
    print(f"runtime_errors={runtime_errors}")
    print(f"malformed_outputs={malformed_outputs}")

    if runtime_errors > 0 or malformed_outputs > 0:
        return 1
    if accuracy < args.min_accuracy:
        print(f"FAILED: accuracy {accuracy:.3f} is below min_accuracy {args.min_accuracy:.3f}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())