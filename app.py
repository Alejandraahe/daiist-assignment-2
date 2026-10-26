"""
Assignment 2: Gradio dashboard (counterfeit peso screening).

    uv run python main.py app

Loads only what train.py produced: weights/best.pt and artifacts/. Nothing is
trained here. The YOLO model is only used to predict on images you upload.
"""

import json

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO

from banknotes import ARTIFACTS, BASE_RATE, NOTES_PER_DAY, WEIGHTS, cost_per_1000, is_positive, staffing

model = YOLO(str(WEIGHTS / "best.pt"))
metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
stats = json.loads((ARTIFACTS / "dataset_stats.json").read_text())
test = pd.read_csv(ARTIFACTS / "test_scores.csv")
sweep_val = pd.read_csv(ARTIFACTS / "cost_sweep_val.csv")
sweep_test = pd.read_csv(ARTIFACTS / "cost_sweep_test.csv")
CH = metrics["chosen"]
LOW0, HIGH0, T0 = CH["review_low"], CH["review_high"], CH["single_threshold"]


def cost(low, high, base_rate):
    return cost_per_1000(test.fake_score, test.is_fake, test.value_php, low, high, base_rate=base_rate)


# ------------------------------------------------------------- check a note
def decide(score, low, high):
    if score >= high:
        return "FAIL: retain as counterfeit"
    if score >= low:
        return "REVIEW: send to a person"
    return "PASS: accept as genuine"


def check_note(image, low, high):
    if image is None:
        return None, "Upload a banknote photo."
    shown = max(min(low, high), 0.05)  # below 0.05 the picture is just noise
    r = model.predict(image, conf=shown, verbose=False)[0]
    names = r.names
    cls = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy()
    fake = [c for k, c in zip(cls, conf) if is_positive(names[int(k)])]
    score = float(max(fake)) if fake else 0.0
    order = np.argsort(-conf)[:10]
    boxes = "\n".join(f"- {names[int(cls[i])]}: {conf[i]:.2f}" for i in order) or "- no boxes above the low threshold"
    md = (f"### Decision: {decide(score, low, high)}\n"
          f"Highest **counterfeit** confidence: **{score:.2f}** "
          f"(pass < {low:.2f} ≤ review < {high:.2f} ≤ fail)\n\n"
          f"Top boxes shown (confidence ≥ {shown:.2f}):\n{boxes}")
    return r.plot()[:, :, ::-1], md


# ------------------------------------------------------------- decision & cost
def policy_view(low, high, base_rate):
    if high < low:
        low, high = high, low
    r = cost(low, high, base_rate)
    s = staffing(r["review_share"])
    cells = np.array([[r["fake_pass"], r["fake_review"], r["fake_fail"]],
                      [r["real_pass"], r["real_review"], r["real_fail"]]])
    cm = px.imshow(cells, text_auto=True, color_continuous_scale="Blues", aspect="auto",
                   x=["PASS", "REVIEW (person)", "FAIL"], y=["actually counterfeit", "actually genuine"],
                   title=f"Test notes by outcome (low {low:.2f}, high {high:.2f})")
    cm.update_layout(coloraxis_showscale=False, height=330)

    alts = {
        "Do nothing (pass all)": cost(1.01, 1.01, base_rate),
        "Person checks everything": cost(0, 1.01, base_rate),
        f"Model alone (t = {T0:.2f})": cost(T0, T0, base_rate),
        "Your band (sliders)": r,
    }
    bar = go.Figure(go.Bar(x=list(alts), y=[v["cost_per_1000"] for v in alts.values()],
                           text=[f"€{v['cost_per_1000']:,.0f}" for v in alts.values()], textposition="outside"))
    bar.update_layout(title=f"Expected cost per 1,000 notes (base rate {base_rate:.1%})",
                      yaxis_title="€ per 1,000 notes", height=330)

    md = (f"### Expected cost: **€{r['cost_per_1000']:,.1f} per 1,000 notes**\n"
          f"- Sent to a person: **{r['review_share']:.1%}** of notes → **{s['reviews_per_day']:,.0f}/day** "
          f"at {NOTES_PER_DAY:,} notes/day → {s['review_hours']:.1f} h → **{s['tellers']:.2f} tellers**\n"
          f"- Counterfeits stopped (fail + review): **{r['fake_caught']:.1%}** · genuine notes wrongly "
          f"retained: **{r['genuine_failed']:.2%}**\n"
          f"- Chosen on validation: band **{LOW0:.2f}–{HIGH0:.2f}**; model-alone threshold **{T0:.2f}**")
    return cm, bar, md


# ------------------------------------------------------------- cost curve
def cost_curve():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep_val.threshold, y=sweep_val.cost_per_1000, name="validation (used to choose)"))
    fig.add_trace(go.Scatter(x=sweep_test.threshold, y=sweep_test.cost_per_1000, name="test",
                             line=dict(dash="dot")))
    best = sweep_val.loc[sweep_val.cost_per_1000.idxmin()]
    fig.add_vline(x=T0, line_dash="dash", line_color="red",
                  annotation_text=f"chosen t = {T0:.2f} (€{best.cost_per_1000:,.1f}/1,000)")
    fig.add_vrect(x0=LOW0, x1=HIGH0, fillcolor="orange", opacity=0.15, line_width=0,
                  annotation_text="review band", annotation_position="top left")
    fig.update_layout(title="Model alone: expected cost per 1,000 notes vs confidence threshold",
                      xaxis_title="threshold on counterfeit confidence", yaxis_title="€ per 1,000 notes",
                      height=450)
    return fig


# ------------------------------------------------------------- per class
def per_class_table():
    t = pd.DataFrame(metrics["test_per_class"])
    t["positive"] = t["positive"].map({True: "yes (counterfeit)", False: "no"})
    return t.round(3)


def baseline_table():
    o = metrics["test_overall"]
    return pd.DataFrame([
        {"model": "Finetuned YOLO", "mAP50": o["mAP50"], "mAP50-95": o["mAP50_95"]},
        {"model": "COCO yolo11n zero-shot", "mAP50": 0.0, "mAP50-95": 0.0},
        {"model": "Predict nothing", "mAP50": 0.0, "mAP50-95": 0.0},
    ]).round(3)


def dataset_table():
    rows = []
    for split, s in stats["splits"].items():
        for cls, n in s["instances_per_class"].items():
            rows.append({"split": split, "class": cls, "instances": n})
    t = pd.DataFrame(rows).pivot_table(index="class", columns="split", values="instances", fill_value=0)
    return t.reindex(columns=[c for c in ["train", "valid", "test"] if c in t.columns]).reset_index()


def policies_table():
    p = pd.DataFrame(metrics["policies_on_test"]).T
    p = p[["low", "high", "cost_per_1000", "review_share", "fake_caught", "genuine_failed",
           "reviews_per_day", "tellers"]]
    return p.reset_index(names="policy").round(3)


with gr.Blocks(title="Counterfeit peso screening") as demo:
    gr.Markdown("# Counterfeit peso screening: pass, review or fail\n"
                "Finetuned YOLO (positive class = **counterfeit**). All numbers are on the held-out **test** "
                "split; thresholds were chosen on validation. Nothing is trained in this app.")
    with gr.Row():
        low = gr.Slider(0, 1, value=LOW0, step=0.01, label="Low threshold: below → PASS")
        high = gr.Slider(0, 1, value=HIGH0, step=0.01, label="High threshold: at or above → FAIL")
        br = gr.Slider(0.005, 0.5, value=BASE_RATE, step=0.005, label="Assumed share of fakes (base rate)")

    with gr.Tab("Check a note"):
        with gr.Row():
            img_in = gr.Image(type="numpy", label="Upload a banknote photo")
            img_out = gr.Image(label="Predictions")
        verdict = gr.Markdown()
        for comp in [img_in, low, high]:
            comp.change(check_note, [img_in, low, high], [img_out, verdict])

    with gr.Tab("Decision & cost"):
        summary = gr.Markdown()
        with gr.Row():
            cm_plot = gr.Plot()
            bar_plot = gr.Plot()
        gr.Dataframe(policies_table(), label="Policies on test (base rate and costs from banknotes.py)")
        for comp in [low, high, br]:
            comp.change(policy_view, [low, high, br], [cm_plot, bar_plot, summary])
        demo.load(policy_view, [low, high, br], [cm_plot, bar_plot, summary])

    with gr.Tab("Cost vs threshold"):
        gr.Plot(cost_curve())

    with gr.Tab("Per-class results"):
        gr.Dataframe(per_class_table(), label="Test split, per class")
        gr.Dataframe(baseline_table(), label="Finetuned vs baselines (test split)")
        cm_png = ARTIFACTS / "confusion_matrix.png"
        if cm_png.exists():
            gr.Image(str(cm_png), label="Detection confusion matrix (test, boxes with confidence ≥ 0.25)")
        gr.Dataframe(dataset_table(), label="Instances per class and split")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
