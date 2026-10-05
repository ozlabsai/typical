from scripts.eval_gemma_mm_he import MM_FILES, NULL_OPTION, _read


def test_locked_hebrew_and_image_panels_adapt_without_dropping_nulls():
    he = _read("data_he/eval/massive_he_test.jsonl", multimodal=False)
    assert len(he) == 128 and not any(case["null"] for case in he)
    real = _read(f"data_v/eval/{MM_FILES['real']}", multimodal=True)
    mismatch = _read(f"data_v/eval/{MM_FILES['mismatch']}", multimodal=True)
    assert len(real) == len(mismatch) == 128
    assert all(case["candidates"][-1] == NULL_OPTION for case in real + mismatch)
    assert all(case["label"] == len(case["candidates"]) - 1 for case in mismatch)
