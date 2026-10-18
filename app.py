import os
import sys
import pathlib
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import lief
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.structure.predict import predict_blocks, table_labels
from src.detector.verdict import score_file, level
from src.detector.rules import apply_rules, elf_machine

try:
    lief.logging.disable()
except Exception:
    pass

COLORS = ["#7f7f7f", "#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
UA = ["заголовок", "код", "дані тільки для читання", "дані", "ресурси", "інше"]
LABELS = {"high": "Шкідливий", "mid": "Підозрілий", "low": "Ймовірно безпечний"}

MODES = ["Стандартний", "Підвищений", "Максимальний"]
MODE_T = {"Підвищений": (0.5, 0.9), "Максимальний": (0.2, 0.5)}
MODE_STATS = {
    ("win64", "Підвищений"): "на власному наборі: знайдено 88,9 % шкідливих, хибних тривог 0,16 %",
    ("win64", "Максимальний"): "на власному наборі: знайдено 92,6 % шкідливих, хибних тривог 0,16 %",
    ("elf", "Підвищений"): "на власному наборі ELF (64 шкідливі, 1 890 чистих, разом із структурними правилами): знайдено 95,3 % шкідливих, хибних тривог 0,3 %",
    ("elf", "Максимальний"): "на власному наборі ELF (64 шкідливі, 1 890 чистих, разом із структурними правилами): знайдено 95,3 % шкідливих, хибних тривог 1,0 %",
}


def strip(labels, width=128):
    n = len(labels)
    rows = -(-n // width)
    arr = np.full(rows * width, len(COLORS))
    arr[:n] = labels
    arr = arr.reshape(rows, width)
    cmap = ListedColormap(COLORS + ["#ffffff"])
    fig, ax = plt.subplots(figsize=(10, max(0.8, rows * 0.16)))
    ax.imshow(arr, cmap=cmap, vmin=0, vmax=len(COLORS),
              aspect="auto", interpolation="nearest")
    ax.axis("off")
    fig.tight_layout()
    return fig


def sections_table(path):
    b = lief.parse(str(path))
    if b is None:
        return None
    rows = []
    for s in b.sections:
        try:
            ent = round(float(s.entropy), 2)
        except Exception:
            ent = None
        rows.append({"секція": s.name, "зсув": int(s.offset),
                     "розмір": int(s.size), "ентропія": ent})
    return pd.DataFrame(rows)


st.set_page_config(page_title="Аналіз бінарних файлів", layout="wide")
st.title("Інтелектуальний аналіз бінарних файлів")
st.caption("Вердикт про шкідливість і розмітка структури файла на рівні блоків")

mode = st.sidebar.radio("Режим чутливості", MODES)
st.sidebar.caption(
    "Стандартний: пороги з навчання на EMBER2024. Підвищений і максимальний: "
    "нижчі пороги, більше знайдених шкідливих, але й вищий ризик хибних тривог. "
    "Режим діє для файлів, які оцінює власна модель (PE x64 та ELF).")
st.sidebar.caption(
    "Для ELF додатково діють структурні правила: якщо таблицю секцій прочитати "
    "не вдалося (для всіх архітектур) або розмітка за вмістом майже не збігається "
    "з нею (для x86 і x86-64), рівень «Ймовірно безпечний» підвищується до «Підозрілий».")

up = st.file_uploader("Завантажте файл PE (.exe, .dll) або ELF")

if up is not None:
    with tempfile.NamedTemporaryFile(delete=False) as t:
        t.write(up.getbuffer())
        path = t.name
    try:
        with st.spinner("Розмітка структури файла"):
            out = predict_blocks(path)
        pred = truth = fmt = ent = None
        n = 0
        if out is not None:
            pred, ent, n = out
            fmt, truth = table_labels(path, n)

        st.header("Вердикт")
        with st.spinner("Обчислення ознак і оцінка моделлю"):
            res = score_file(path)
        if res is None:
            st.warning("Формат не підтримується: потрібні файли PE або ELF")
        else:
            own = res["key"] in ("win64", "elf")
            mode_note = ""
            if mode != "Стандартний":
                if own:
                    res["t_mid"], res["t_high"] = MODE_T[mode]
                    res["test_fpr_mid"] = None
                    mode_note = f" Режим «{mode}»: {MODE_STATS[(res['key'], mode)]}."
                else:
                    mode_note = (" Режим чутливості на цей файл не впливає: його оцінює готова "
                                 "модель EMBER2024 з фіксованими порогами.")
            lvl = level(res)
            machine = elf_machine(path) if res["kind"] == "ELF" else None
            lvl, reasons, raised = apply_rules(lvl, res["kind"], pred, truth, machine)

            c1, c2, c3 = st.columns(3)
            c1.metric("Оцінка моделі", f"{res['score'] * 100:.1f} %")
            if "prob" in res:
                c2.metric("Відкалібрована ймовірність", f"{res['prob'] * 100:.1f} %")
            else:
                c2.metric("Відкалібрована ймовірність", "немає")
            c3.markdown(f"**Модель**  \n{res['source']}")

            if lvl == "high":
                st.error(LABELS[lvl])
            elif lvl == "mid":
                st.warning(LABELS[lvl])
            else:
                st.success(LABELS[lvl])
            st.progress(min(max(res["score"], 0.0), 1.0))

            if reasons:
                head = ("Рівень підвищено до «Підозрілий» за структурними ознаками: "
                        if raised else
                        "Структурні ознаки також нетипові для звичайних файлів: ")
                st.info(head + "; ".join(reasons) + ".")

            note = (f"Пороги оцінки моделі: підозрілий від {res['t_mid'] * 100:.1f} %, "
                    f"шкідливий від {res['t_high'] * 100:.1f} %.")
            if res.get("test_fpr_mid") is not None:
                note += (f" На тестових даних ці пороги дали частку хибних тривог "
                         f"{res['test_fpr_mid'] * 100:.2f} % та "
                         f"{res['test_fpr_high'] * 100:.2f} % відповідно.")
            note += mode_note
            note += (" Ймовірність відкалібрована на вибірці, де шкідливих і безпечних порівну, "
                     "тому в реальному потоці файлів справжня ймовірність шкідливості нижча.")
            st.caption(note)

        st.header("Структура файла")
        if out is None:
            st.error("Файл надто малий для аналізу структури")
        else:
            legend = " &nbsp; ".join(
                f"<span style='color:{c}'>&#9632;</span> {nm}"
                for c, nm in zip(COLORS, UA))
            st.markdown(legend, unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Розмір, байт", f"{len(up.getbuffer()):,}".replace(",", " "))
            c2.metric("Блоків по 512 байтів", n)
            c3.metric("Формат", fmt or "не розпізнано")

            st.subheader("Розмітка за моделлю")
            st.pyplot(strip(pred))

            if truth is not None:
                st.subheader("Розмітка за таблицею секцій")
                st.pyplot(strip(truth))
                st.metric("Збіг моделі з таблицею секцій",
                          f"{(pred == truth).mean() * 100:.1f} %")
            else:
                st.info("Таблицю секцій прочитати не вдалося. "
                        "Модель усе одно розмітила файл за вмістом блоків.")

            st.subheader("Склад файла за версією моделі")
            share = pd.Series(pred).value_counts().sort_index()
            comp = pd.DataFrame({
                "тип": [UA[i] for i in share.index],
                "блоків": share.values,
                "частка, %": (share.values / n * 100).round(1)})
            st.dataframe(comp, hide_index=True)

            st.subheader("Ентропія вздовж файла")
            st.line_chart(pd.DataFrame({"ентропія": ent}))

            tbl = sections_table(path)
            if tbl is not None and len(tbl):
                st.subheader("Секції за заголовком")
                st.dataframe(tbl, hide_index=True)
    finally:
        os.unlink(path)