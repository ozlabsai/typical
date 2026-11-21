import sys
import types
from argparse import Namespace

import train


def test_wandb_failure_inside_a_job_does_not_kill_training(monkeypatch):
    fake = types.SimpleNamespace(init=lambda **kw: (_ for _ in ()).throw(RuntimeError("401 bad key")))
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setenv("WANDB_RUN_ID", "acme-ab12")
    assert train.init_wandb(Namespace(name="co_acme")) is None


def test_wandb_failure_outside_a_job_still_raises(monkeypatch):
    fake = types.SimpleNamespace(init=lambda **kw: (_ for _ in ()).throw(RuntimeError("401 bad key")))
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    try:
        train.init_wandb(Namespace(name="r1"))
    except RuntimeError:
        return
    raise AssertionError("expected the research run to fail loudly")
