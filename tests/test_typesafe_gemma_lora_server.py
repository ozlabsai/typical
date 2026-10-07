from __future__ import annotations

import math
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import httpx2
import pytest
import uvicorn
from fastapi.testclient import TestClient
from typesafe_sdk import RetryPolicy, TypeSafeClient

from scripts.gemma_decision_runtime import (
    GemmaDecisionRuntime,
    OptionScores,
    Overlength,
    artifact_identity,
    render_decision_prompt,
    render_typesafe_decision_prompt,
)
from scripts.preflight_r043_assets import tree_hash
from scripts.serve_typesafe_gemma_lora import (
    _require_safe_bind,
    build_startup_receipt,
    create_app,
    write_startup_receipt,
)

MODEL = "typical-r050-sha256-0123456789abcdef"


class FakeRuntime:
    """Deterministic option scorer that records the semantic order sent by the server."""

    def __init__(self, vectors: Sequence[Sequence[float]] | None = None) -> None:
        self.vectors = iter(vectors or ())
        self.calls: list[dict] = []
        self.failure: Exception | None = None

    def score_options(self, state, instructions, items, *, question_type):
        self.calls.append({
            "state": state,
            "instructions": instructions,
            "items": list(items),
            "question_type": question_type,
        })
        if self.failure is not None:
            raise self.failure
        try:
            values = list(next(self.vectors))
        except StopIteration:
            values = [index + 1 for index in range(len(items))]
        total = float(sum(values))
        probabilities = {
            label: value / total for (label, _), value in zip(items, values, strict=True)
        }
        return OptionScores(probabilities=probabilities, input_tokens=10 + len(items))


def client(runtime: FakeRuntime | None = None, **kwargs) -> tuple[TestClient, FakeRuntime]:
    runtime = runtime or FakeRuntime()
    return TestClient(create_app(runtime, model_id=MODEL, **kwargs)), runtime


def post(http: TestClient, *, state="state", questions=None, model=MODEL, headers=None):
    return http.post(
        "/v1/systemone",
        json={"state": state, "model": model, "questions": questions or {
            "ok": {"type": "noul", "instructions": "Is it true?"}
        }},
        headers=headers,
    )


def test_direct_script_help_bootstraps_repository_imports(tmp_path):
    script = Path("scripts/serve_typesafe_gemma_lora.py").resolve()
    result = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=tmp_path,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "TypeSafe-compatible" in result.stdout
    assert "base URL is the host root" in result.stdout


def test_all_primitives_preserve_order_and_shape_answers_exactly():
    runtime = FakeRuntime([
        [9, 1],
        [2, 7, 1],
        [1, 2, 3],
    ])
    http, _ = client(runtime)
    questions = {
        "truth": {
            "type": "noul",
            "instructions": "אמת?",
            "criteria": {"true": "כן", "false": "לא"},
        },
        "route": {
            "type": "choice",
            "instructions": {"he": "בחר"},
            "criteria": {"third": "ג", "first": "א", "second": "ב"},
        },
        "risk": {
            "type": "score",
            "instructions": ["דרג", {"domain": "סיכון"}],
            "criteria": ["נמוך", {"level": "בינוני"}, ["גבוה"]],
        },
    }

    response = post(http, state={"לקוח": "שרה", "אירועים": [1, 2]}, questions=questions)

    assert response.status_code == 200, response.text
    body = response.json()
    assert list(body) == ["model", "answers", "usage"]
    assert body["model"] == MODEL
    assert list(body["answers"]) == list(questions)
    assert body["answers"]["truth"] == {"type": "noul", "noul": pytest.approx(0.9)}
    assert body["answers"]["route"] == {
        "type": "choice",
        "choice": "first",
        "confidence": pytest.approx(0.55),
        "probabilities": {
            "third": pytest.approx(0.2),
            "first": pytest.approx(0.7),
            "second": pytest.approx(0.1),
        },
    }
    assert body["answers"]["risk"] == {
        "type": "score",
        "score": pytest.approx(8 / 6),
        "confidence": pytest.approx(0.25),
        "legend": {"0": "נמוך", "1": {"level": "בינוני"}, "2": ["גבוה"]},
        "probabilities": {
            "0": pytest.approx(1 / 6), "1": pytest.approx(2 / 6), "2": pytest.approx(3 / 6)
        },
    }
    assert body["usage"] == {"input_tokens": 38, "output_tokens": 0}
    assert runtime.calls[0]["items"] == [("true", "כן"), ("false", "לא")]
    assert [label for label, _ in runtime.calls[1]["items"]] == ["third", "first", "second"]
    assert [label for label, _ in runtime.calls[2]["items"]] == ["0", "1", "2"]
    assert runtime.calls[0]["state"] == {"לקוח": "שרה", "אירועים": [1, 2]}


def test_noul_is_semantic_true_probability_under_criteria_order_permutations():
    left, left_runtime = client(FakeRuntime([[7, 3]]))
    right, right_runtime = client(FakeRuntime([[3, 7]]))
    ordinary = {"q": {"type": "noul", "criteria": {"true": "T", "false": "F"}}}
    reversed_criteria = {"q": {"type": "noul", "criteria": {"false": "F", "true": "T"}}}

    assert post(left, questions=ordinary).json()["answers"]["q"] == {
        "type": "noul", "noul": 0.7
    }
    assert post(right, questions=reversed_criteria).json()["answers"]["q"] == {
        "type": "noul", "noul": 0.7
    }
    assert left_runtime.calls[0]["items"] == [("true", "T"), ("false", "F")]
    assert right_runtime.calls[0]["items"] == [("false", "F"), ("true", "T")]


def test_joint_request_is_atomic_for_120_questions_and_usage_is_summed():
    questions = {
        f"q{index:03d}": {
            "type": "choice",
            "criteria": {"z": None, "a": {"index": index}},
        }
        for index in range(120)
    }
    http, runtime = client()

    response = post(http, state=["שלום", {"structured": True}], questions=questions)

    assert response.status_code == 200
    body = response.json()
    assert list(body["answers"]) == list(questions)
    assert all(answer["choice"] == "a" for answer in body["answers"].values())
    assert body["usage"] == {"input_tokens": 1440, "output_tokens": 0}
    assert len(runtime.calls) == 120


def test_twenty_six_choice_items_are_supported_in_incoming_order():
    criteria = {chr(ord("a") + index): f"option {index}" for index in reversed(range(26))}
    http, runtime = client()
    response = post(http, questions={"wide": {"type": "choice", "criteria": criteria}})
    assert response.status_code == 200
    answer = response.json()["answers"]["wide"]
    assert list(answer["probabilities"]) == list(criteria)
    assert runtime.calls[0]["items"] == list(criteria.items())
    assert math.isclose(sum(answer["probabilities"].values()), 1.0, abs_tol=1e-12)


@pytest.mark.parametrize(
    ("payload", "status"),
    [
        ({"state": "s", "model": MODEL, "questions": {}}, 422),
        ({"state": "s", "model": "", "questions": {"q": {"type": "noul"}}}, 422),
        ({"state": 3, "model": MODEL, "questions": {"q": {"type": "noul"}}}, 422),
        ({"state": "s", "model": MODEL, "questions": {"q": {"type": "unknown"}}}, 422),
        ({"state": "s", "model": MODEL, "questions": {
            "q": {"type": "noul", "criteria": {"yes": "wrong vocabulary"}}
        }}, 422),
        ({"state": "s", "model": MODEL, "questions": {
            "q": {"type": "choice", "criteria": {}}
        }}, 422),
        ({"state": "s", "model": MODEL, "questions": {
            "q": {"type": "score", "criteria": []}
        }}, 422),
        ({"state": "s", "model": MODEL, "questions": {
            "q": {"type": "choice", "criteria": {str(i): None for i in range(27)}}
        }}, 422),
    ],
)
def test_malformed_requests_fail_deterministically(payload, status):
    http, runtime = client()
    response = http.post("/v1/systemone", json=payload)
    assert response.status_code == status
    assert runtime.calls == []


def test_too_many_questions_and_wrong_model_fail_before_runtime():
    http, runtime = client(max_questions=120)
    too_many = {f"q{i}": {"type": "noul"} for i in range(121)}
    assert post(http, questions=too_many).status_code == 422
    mismatch = post(http, model="mutable-alias")
    assert mismatch.status_code == 404
    assert mismatch.json() == {"detail": "model not found"}
    assert runtime.calls == []


def test_overlength_is_422_and_runtime_failure_is_sanitized_503():
    runtime = FakeRuntime()
    http, _ = client(runtime)
    runtime.failure = Overlength(17000, 16384)
    overlength = post(http)
    assert overlength.status_code == 422
    assert overlength.json() == {"detail": "prompt has 17000 tokens; configured maximum is 16384"}

    runtime.failure = RuntimeError("CUDA secret implementation detail")
    failed = post(http)
    assert failed.status_code == 503
    assert failed.json() == {"detail": "decision runtime unavailable"}


@pytest.mark.parametrize("probabilities", [
    {"true": 1.0},
    {"true": math.nan, "false": math.nan},
    {"false": 0.4, "true": 0.4},
])
def test_invalid_runtime_probabilities_are_503(probabilities):
    class InvalidRuntime:
        def score_options(self, state, instructions, items, *, question_type):
            return OptionScores(probabilities, 4)

    http = TestClient(create_app(InvalidRuntime(), model_id=MODEL))
    response = post(http)
    assert response.status_code == 503
    assert response.json() == {"detail": "decision runtime unavailable"}


def test_ties_use_incoming_choice_order_and_normalized_zero_confidence():
    http, _ = client(FakeRuntime([[1, 1]]))
    answer = post(http, questions={
        "tie": {"type": "choice", "criteria": {"second": None, "first": None}}
    }).json()["answers"]["tie"]
    assert answer == {
        "type": "choice",
        "choice": "second",
        "confidence": 0.0,
        "probabilities": {"second": 0.5, "first": 0.5},
    }


def test_optional_and_one_sided_noul_criteria_remain_semantic():
    absent, absent_runtime = client(FakeRuntime([[8, 2]]))
    one_sided, one_sided_runtime = client(FakeRuntime([[3, 7]]))
    assert post(absent, questions={"q": {"type": "noul"}}).json()["answers"]["q"]["noul"] == .8
    assert post(one_sided, questions={
        "q": {"type": "noul", "criteria": {"false": "No evidence"}}
    }).json()["answers"]["q"]["noul"] == .7
    assert absent_runtime.calls[0]["items"] == [("true", None), ("false", None)]
    assert one_sided_runtime.calls[0]["items"] == [("false", "No evidence"), ("true", None)]


def test_mid_request_failure_returns_no_partial_answers():
    class FailsSecond:
        def __init__(self):
            self.calls = 0

        def score_options(self, state, instructions, items, *, question_type):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("second question failed")
            return OptionScores({label: 1 / len(items) for label, _ in items}, 5)

    runtime = FailsSecond()
    http = TestClient(create_app(runtime, model_id=MODEL))
    response = post(http, questions={
        "first": {"type": "noul"},
        "second": {"type": "noul"},
        "never": {"type": "noul"},
    })
    assert response.status_code == 503
    assert response.json() == {"detail": "decision runtime unavailable"}
    assert runtime.calls == 2


def test_concurrent_requests_do_not_interleave_question_forwards():
    class BlockingRuntime:
        def __init__(self):
            self.calls = []
            self.first_started = threading.Event()
            self.release_first = threading.Event()

        def score_options(self, state, instructions, items, *, question_type):
            self.calls.append(state)
            if state == "first" and len(self.calls) == 1:
                self.first_started.set()
                assert self.release_first.wait(timeout=5)
            return OptionScores({label: 1 / len(items) for label, _ in items}, 3)

    runtime = BlockingRuntime()
    app = create_app(runtime, model_id=MODEL)
    first_client, second_client = TestClient(app), TestClient(app)
    questions = {"a": {"type": "noul"}, "b": {"type": "noul"}}
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(post, first_client, state="first", questions=questions)
        assert runtime.first_started.wait(timeout=5)
        second = pool.submit(post, second_client, state="second", questions=questions)
        time.sleep(0.1)
        assert runtime.calls == ["first"]
        runtime.release_first.set()
        assert first.result(timeout=5).status_code == 200
        assert second.result(timeout=5).status_code == 200
    assert runtime.calls == ["first", "first", "second", "second"]


def test_bearer_auth_and_models_endpoint():
    http, runtime = client(api_key="local-secret", release_date="2026-10-07")
    assert http.get("/v1/models").status_code == 401
    assert post(http).status_code == 401
    assert post(http, headers={"Authorization": "Basic local-secret"}).status_code == 401

    headers = {"Authorization": "Bearer local-secret"}
    models = http.get("/v1/models", headers=headers)
    assert models.status_code == 200
    assert models.json() == {"models": [{
        "name": MODEL,
        "description": "Typical Gemma decision model",
        "release_date": "2026-10-07",
    }]}
    assert post(http, headers=headers).status_code == 200
    assert len(runtime.calls) == 1


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "127.8.9.10", "::1"])
def test_loopback_may_bind_without_auth(host):
    _require_safe_bind(host, api_key=None, allow_unauthenticated_remote=False)


def test_remote_bind_requires_auth_or_explicit_override():
    with pytest.raises(ValueError, match="refusing unauthenticated non-loopback"):
        _require_safe_bind("0.0.0.0", api_key=None, allow_unauthenticated_remote=False)
    _require_safe_bind("0.0.0.0", api_key="secret", allow_unauthenticated_remote=False)
    _require_safe_bind("0.0.0.0", api_key=None, allow_unauthenticated_remote=True)


def test_startup_receipt_binds_artifacts_code_and_runtime(tmp_path):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    (base / "config.json").write_text('{"model":"gemma"}\n')
    (adapter / "adapter_config.json").write_text('{"rank":32}\n')
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    adapter_hash = artifact_identity(str(adapter), None)["canonical_sha256"]
    model_id = f"typical-r050-sha256-{adapter_hash[:16]}"

    receipt = build_startup_receipt(
        model_id=model_id,
        base_model=str(base),
        base_revision="base-rev",
        adapter=str(adapter),
        adapter_revision=None,
        max_context=32768,
        dtype="bfloat16",
        device="cuda",
        attn_implementation="sdpa",
    )

    assert receipt["model_id"] == model_id
    assert receipt["base_model"]["revision"] == "base-rev"
    canonical_hash, canonical_files = tree_hash(adapter)
    assert receipt["adapter"]["artifact"]["tree_sha256"] == canonical_hash == adapter_hash
    assert receipt["adapter"]["artifact"]["files"] == canonical_files
    assert receipt["runtime"] == {
        "max_context": 32768,
        "dtype": "bfloat16",
        "device": "cuda",
        "attn_implementation": "sdpa",
    }
    assert all(len(value) == 64 for value in receipt["code"].values())
    assert len(receipt["receipt_sha256"]) == 64

    path = tmp_path / "startup-receipt.json"
    write_startup_receipt(path, receipt)
    write_startup_receipt(path, receipt)
    assert path.read_text().endswith("\n")
    path.write_text('{}\n')
    with pytest.raises(ValueError, match="existing startup receipt differs"):
        write_startup_receipt(path, receipt)


def test_startup_receipt_rejects_model_id_not_bound_to_adapter(tmp_path):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    (base / "config.json").write_text("base")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    with pytest.raises(ValueError, match="adapter sha256 prefix"):
        build_startup_receipt(
            model_id="typical-r050-sha256-0000000000000000",
            base_model=str(base),
            base_revision="rev",
            adapter=str(adapter),
            adapter_revision=None,
            max_context=16384,
            dtype="bfloat16",
            device="cuda",
            attn_implementation="sdpa",
        )


@pytest.mark.parametrize("state", ["שלום עולם", {"טקסט": "בדיקה"}, ["א", {"ב": 2}]])
def test_rendering_supports_text_object_array_and_hebrew(state):
    prompt = render_decision_prompt(
        state, {"שאלה": "מה נכון?"}, [("לא", {"תיאור": "שגוי"}), ("כן", ["נכון"])]
    )
    assert "שלום" in prompt or "בדיקה" in prompt or '"א"' in prompt
    assert "שאלה" in prompt and "תיאור" in prompt and "נכון" in prompt
    assert "A." in prompt and "B." in prompt


def test_shared_renderer_preserves_the_training_wire_format():
    prompt = render_decision_prompt(
        {"state": [1, "שלום"]},
        {"question": "בחר"},
        [("implicit", None), ("structured", {"key": "ערך"})],
    )
    assert prompt == "\n".join([
        "{", ' \"state\": [', "  1,", '  \"שלום\"', " ]", "}",
        "", '{\"question\": \"בחר\"}', "", "Options:",
        "A. None", "B. {'key': 'ערך'}", "",
        "Answer with the letter of exactly one option, and nothing else:",
    ])


@pytest.mark.parametrize("instructions", [None, "", {}, []])
@pytest.mark.parametrize("description", [None, "", {}, []])
def test_legacy_renderer_is_byte_exact_for_empty_and_null_values(instructions, description):
    state = {"state": [1, "שלום"]}
    expected_state = '{\n "state": [\n  1,\n  "שלום"\n ]\n}'
    old_instructions = instructions or ""
    if not isinstance(old_instructions, str):
        import json
        old_instructions = json.dumps(old_instructions, ensure_ascii=False)
    expected = "\n".join([
        expected_state, "", old_instructions, "", "Options:",
        f"A. {description}", "",
        "Answer with the letter of exactly one option, and nothing else:",
    ])
    assert render_decision_prompt(state, instructions, [("label", description)]) == expected


@pytest.mark.parametrize(
    ("question_type", "items", "expected"),
    [
        (
            "choice",
            [("route", {"ז": 2, "a": [1, True]}), ("blank", ""), ("null", None)],
            ['A. route: {"a":[1,true],"ז":2}', "B. blank", "C. null"],
        ),
        (
            "score",
            [("0", {"ז": 2, "a": [1, True]}), ("1", ""), ("2", [])],
            ['A. {"a":[1,true],"ז":2}', "B. Level 1", "C. Level 2"],
        ),
        (
            "noul",
            [("true", None), ("false", {"reason": "לא"})],
            ["A. Yes", 'B. No: {"reason":"לא"}'],
        ),
    ],
)
def test_typesafe_renderer_is_question_aware(question_type, items, expected):
    prompt = render_typesafe_decision_prompt(
        {"state": "שלום"}, {"question": "בחר"}, items, question_type=question_type
    )
    assert [line for line in prompt.splitlines() if line.startswith(("A. ", "B. ", "C. "))] == expected


def test_runtime_type_safe_path_selects_question_aware_rendering(monkeypatch):
    runtime = GemmaDecisionRuntime("base", "rev", None, None, 16384, "bfloat16")
    observed = {}

    def capture(prompt, labels):
        observed.update(prompt=prompt, labels=list(labels))
        return OptionScores({"object": 0.25, "array": 0.75}, 12)

    monkeypatch.setattr(runtime, "score_prompt", capture)
    result = runtime.score_options(
        {"state": "שלום"},
        {"question": "בחר"},
        [("object", {"ז": 2, "a": [1, True]}), ("array", ["שלום", {"b": 2}])],
        question_type="choice",
    )
    assert observed["labels"] == ["object", "array"]
    assert 'A. object: {"a":[1,true],"ז":2}' in observed["prompt"]
    assert 'B. array: ["שלום",{"b":2}]' in observed["prompt"]
    assert result.input_tokens == 12


def test_response_decodes_with_pinned_typesafe_sdk_0_7_2():
    assert version("typesafe-sdk") == "0.7.2"
    http, _ = client(FakeRuntime([[4, 6], [1, 2, 7], [2, 3]]))
    response = post(http, questions={
        "n": {"type": "noul", "criteria": {"true": "yes", "false": "no"}},
        "c": {"type": "choice", "criteria": {"a": None, "b": None, "c": None}},
        "s": {"type": "score", "criteria": ["low", "high"]},
    })
    wire = httpx2.Response(
        response.status_code,
        content=response.content,
        headers={"content-type": "application/json"},
        request=httpx2.Request("POST", "http://127.0.0.1:8090/v1/systemone"),
    )

    from typesafe_sdk import SystemOneResponse

    decoded = SystemOneResponse.from_http_response(wire)

    assert decoded.model == MODEL
    assert decoded.nouls["n"].noul == pytest.approx(0.4)
    assert decoded.choices["c"].choice == "c"
    assert decoded.choices["c"].confidence == pytest.approx(0.55)
    assert decoded.scores["s"].score == pytest.approx(0.6)
    assert decoded.scores["s"].legend == {0: "low", 1: "high"}
    assert decoded.usage.input_tokens == 37
    assert decoded.usage.output_tokens == 0


@contextmanager
def _serve_locally(app):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="error", access_log=False,
    ))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()
        raise RuntimeError("test server did not start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()
        assert not thread.is_alive()


def test_pinned_typesafe_client_round_trips_over_real_http():
    assert version("typesafe-sdk") == "0.7.2"
    app = create_app(
        FakeRuntime([[4, 6], [1, 2, 7], [2, 3]]),
        model_id=MODEL,
        api_key="sdk-secret",
    )
    with _serve_locally(app) as base_url, TypeSafeClient(
        api_key="sdk-secret",
        model=MODEL,
        base_url=base_url,
        timeout=5,
        retry=RetryPolicy(max_retries=0),
    ) as sdk:
        response = sdk.system_one(state={"לקוח": "שרה"}, questions={
            "n": {"type": "noul", "criteria": {"true": "כן", "false": "לא"}},
            "c": {"type": "choice", "criteria": {"א": None, "ב": None, "ג": None}},
            "s": {"type": "score", "criteria": ["נמוך", "גבוה"]},
        })
        models = sdk.models.list()

    assert response.model == MODEL
    assert list(response.answers) == ["n", "c", "s"]
    assert response.nouls["n"].noul == pytest.approx(.4)
    assert response.choices["c"].choice == "ג"
    assert response.scores["s"].score == pytest.approx(.6)
    assert response.usage.input_tokens == 37
    assert response.usage.output_tokens == 0
    assert [model.name for model in models.models] == [MODEL]
