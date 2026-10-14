import numpy as np
from src.detector.rules import apply_rules


def test_no_table_raises_low_elf():
    lvl, why, raised = apply_rules("low", "ELF", np.zeros(10, int), None)
    assert lvl == "mid" and raised and why


def test_low_agreement_raises_low_elf():
    pred = np.zeros(10, int)
    truth = np.array([0] * 4 + [1] * 6)
    lvl, why, raised = apply_rules("low", "ELF", pred, truth)
    assert lvl == "mid" and raised


def test_high_stays_high_but_reason_kept():
    lvl, why, raised = apply_rules("high", "ELF", np.zeros(10, int), None)
    assert lvl == "high" and not raised and why


def test_normal_elf_unchanged():
    p = np.zeros(10, int)
    assert apply_rules("low", "ELF", p, p.copy()) == ("low", [], False)


def test_pe_never_changed():
    assert apply_rules("low", "PE", np.zeros(10, int), None) == ("low", [], False)