AGREE_MIN = 0.6


def apply_rules(lvl, kind, pred, truth):
    """Структурні правила підвищення рівня для ELF.

    R1: таблицю секцій прочитати не вдалося.
    R2: розмітка моделі збігається з таблицею секцій менш ніж на AGREE_MIN.
    Повертає (рівень, причини, чи_підвищено).
    """
    if kind != "ELF" or pred is None:
        return lvl, [], False

    reasons = []
    if truth is None:
        reasons.append("ELF без читаної таблиці секцій")
    else:
        agree = float((pred == truth).mean())
        if agree < AGREE_MIN:
            reasons.append(
                f"розмітка за вмістом збігається з таблицею секцій лише на {agree * 100:.1f} %"
            )

    raised = bool(reasons) and lvl == "low"
    return ("mid" if raised else lvl), reasons, raised