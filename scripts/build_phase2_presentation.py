"""Build the Phase 2 project presentation from committed result artifacts.

Run from the repository root with:
    python -m scripts.build_phase2_presentation
"""

from pathlib import Path

from PIL import Image as PILImage
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "outputs" / "figures" / "phase2"
OUTPUT = ROOT / "docs" / "phase2_presentation.pptx"

NAVY = RGBColor(20, 38, 63)
TEAL = RGBColor(22, 145, 145)
CORAL = RGBColor(238, 119, 89)
INK = RGBColor(36, 48, 62)
MUTED = RGBColor(102, 117, 132)
PALE = RGBColor(239, 244, 248)
WHITE = RGBColor(255, 255, 255)


def add_text(slide, text, x, y, w, h, size=20, color=INK, bold=False):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.02)
    frame.margin_right = Inches(0.02)
    frame.margin_top = Inches(0.02)
    frame.margin_bottom = Inches(0.02)
    paragraph = frame.paragraphs[0]
    paragraph.text = f"•  {text}"
    paragraph.font.name = "Aptos"
    paragraph.font.size = Pt(size)
    paragraph.font.bold = bold
    paragraph.font.color.rgb = color
    return box


def add_base(prs, title, section, number):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.background.fill
    background.solid()
    background.fore_color.rgb = WHITE
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(0.18)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = TEAL
    bar.line.fill.background()
    add_text(slide, section.upper(), 0.65, 0.38, 11.8, 0.3, 10, TEAL, True)
    add_text(slide, title, 0.65, 0.76, 12.0, 0.65, 28, NAVY, True)
    add_text(slide, "PROJECT 7  ·  FD001", 0.65, 7.12, 4.0, 0.2, 9, MUTED)
    add_text(slide, f"{number:02d}", 12.2, 7.08, 0.45, 0.25, 10, MUTED)
    return slide


def add_bullets(slide, items, x=0.85, y=1.75, w=11.6, h=4.9, size=21):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.03)
    frame.margin_right = Inches(0.03)
    frame.vertical_anchor = MSO_ANCHOR.TOP
    for index, text in enumerate(items):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = text
        paragraph.level = 0
        paragraph.space_after = Pt(16)
        paragraph.font.name = "Aptos"
        paragraph.font.size = Pt(size)
        paragraph.font.color.rgb = INK
    return box


def add_picture(slide, filename, x, y, w, h):
    path = FIGURES / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required presentation figure is missing: {path}")
    with PILImage.open(path) as image:
        image_ratio = image.width / image.height
    box_ratio = w / h
    if image_ratio > box_ratio:
        picture_w = w
        picture_h = w / image_ratio
    else:
        picture_h = h
        picture_w = h * image_ratio
    picture_x = x + (w - picture_w) / 2
    picture_y = y + (h - picture_h) / 2
    slide.shapes.add_picture(
        str(path),
        Inches(picture_x),
        Inches(picture_y),
        width=Inches(picture_w),
        height=Inches(picture_h),
    )


def add_metric_table(slide):
    rows = [
        ("Model", "RMSE", "MAE", "NASA score"),
        ("Phase 1 XGBoost", "10.754", "8.085", "161.109"),
        ("Phase 2 LSTM", "14.857", "10.559", "309.965"),
    ]
    table = slide.shapes.add_table(
        len(rows), 4, Inches(0.85), Inches(2.0), Inches(11.6), Inches(1.75)
    ).table
    for column, width in enumerate((4.7, 2.0, 2.0, 2.9)):
        table.columns[column].width = Inches(width)
    for row_index, values in enumerate(rows):
        for column_index, value in enumerate(values):
            cell = table.cell(row_index, column_index)
            cell.text = value
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if row_index == 0 else (
                PALE if row_index == 1 else WHITE
            )
            cell.margin_left = Inches(0.12)
            cell.margin_right = Inches(0.12)
            cell.margin_top = Inches(0.08)
            cell.margin_bottom = Inches(0.06)
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.name = "Aptos"
                paragraph.font.size = Pt(18)
                paragraph.font.bold = row_index == 0 or column_index == 0
                paragraph.font.color.rgb = WHITE if row_index == 0 else INK
                paragraph.alignment = (
                    PP_ALIGN.LEFT if column_index == 0 else PP_ALIGN.CENTER
                )


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = NAVY
    accent = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.0), Inches(0.18), Inches(4.9)
    )
    accent.fill.solid()
    accent.fill.fore_color.rgb = TEAL
    accent.line.fill.background()
    add_text(slide, "PREDICTIVE MAINTENANCE", 1.35, 1.15, 10.8, 0.4, 15, TEAL, True)
    add_text(
        slide,
        "Remaining Useful Life\nPrediction for Turbofan Engines",
        1.3,
        1.9,
        10.9,
        1.8,
        34,
        WHITE,
        True,
    )
    add_text(
        slide,
        "Engineered features vs. sequence learning · NASA C-MAPSS FD001",
        1.35,
        4.15,
        10.7,
        0.55,
        20,
        WHITE,
    )
    add_text(
        slide,
        "Phase 2 evaluation: the validation-selected LSTM did not outperform XGBoost.",
        1.35,
        5.85,
        10.8,
        0.6,
        16,
        RGBColor(206, 220, 232),
    )

    slide = add_base(prs, "Problem and benchmark", "01 · Task", 2)
    add_bullets(
        slide,
        [
            "Predict remaining operating cycles before simulated turbofan engine failure.",
            "FD001: one operating condition, one fault mode, 100 run-to-failure training engines and 100 official test engines.",
            "Training target: min(remaining cycles, 125); final test label is the supplied uncapped RUL.",
            "Compare RMSE, MAE, and asymmetric NASA score; lower is better.",
        ],
        size=20,
    )

    slide = add_base(prs, "Phase 1 establishes a strong benchmark", "02 · Classical ML", 3)
    add_bullets(
        slide,
        [
            "Causal feature pipeline: rolling statistics, sensor trends, engine-baseline deviations, cycle count, and health index.",
            "Transformations are fitted on training engines; cross-validation groups by engine ID.",
            "Tuned XGBoost on the official test set: RMSE 10.754, MAE 8.085, NASA score 161.109.",
            "The 14-sensor engineered representation substantially improves on raw sensor baselines.",
        ],
        size=19,
    )

    slide = add_base(prs, "Why evaluate an LSTM?", "03 · Motivation", 4)
    add_bullets(
        slide,
        [
            "Handcrafted windows may miss longer degradation trajectories and differences in timing.",
            "An LSTM can learn temporal representations directly from chronological sensor histories.",
            "The hypothesis is testable, not guaranteed: the sequence model must beat the established baseline on unseen engines.",
        ],
        size=21,
    )

    slide = add_base(prs, "Leakage-safe Phase 2 protocol", "04 · Data", 5)
    add_bullets(
        slide,
        [
            "Split 100 training engines into 80 fit / 20 validation engines (seed 42) before creating overlapping windows.",
            "Choose deterministic pre-failure validation cutoffs (seed 43); validation is not restricted to failure rows.",
            "Fit StandardScaler only on fit-engine rows; reuse it unchanged for validation and test.",
            "Keep windows chronological and inside each engine; repeat the first normalized vector for left padding.",
            "Align each target with the window endpoint; cap training RUL at 125.",
        ],
        size=18,
    )

    slide = add_base(prs, "Selected LSTM and training", "05 · Architecture", 6)
    add_bullets(
        slide,
        [
            "Input: 50 cycles × 14 standardized sensors; one LSTM layer, hidden size 32, dropout 0.2.",
            "Fully connected regression head predicts one continuous RUL value.",
            "AdamW, Huber loss, learning rate 0.001, weight decay 0.0001, batch size 256, seed 42.",
            "Window/regularization selection uses capped-RUL RMSE on validation engines only.",
            "Refit selected model on all 100 training engines for the validation-selected 32 epochs.",
        ],
        size=18,
    )

    slide = add_base(prs, "Validation ablations", "06 · Model selection", 7)
    add_bullets(
        slide,
        [
            "Window 20: RMSE 18.385; window 30: 12.895; window 50: 8.359.",
            "At window 30, dropout 0 / 0.4 gave RMSE 12.886 / 10.082.",
            "At window 30, no weight decay gave RMSE 12.090.",
            "The 50-cycle configuration was selected solely on validation RMSE; these are not test metrics.",
        ],
        x=0.8,
        y=1.6,
        w=5.9,
        h=4.8,
        size=17,
    )
    add_picture(slide, "validation_ablation.png", 6.7, 1.7, 5.9, 4.4)

    slide = add_base(prs, "Official FD001 test comparison", "07 · Final evaluation", 8)
    add_metric_table(slide)
    add_text(
        slide,
        "Same 100 test engines · one prediction per engine at its last observed cycle · uncapped true RUL",
        0.9,
        4.05,
        11.6,
        0.6,
        17,
        MUTED,
    )
    add_text(
        slide,
        "The selected LSTM did not improve the benchmark. No post-test tuning was performed.",
        0.9,
        5.0,
        11.7,
        0.8,
        23,
        CORAL,
        True,
    )

    slide = add_base(prs, "The validation-to-test gap matters", "08 · Interpretation", 9)
    add_bullets(
        slide,
        [
            "LSTM validation RMSE 8.359 did not generalize: official-test RMSE is 14.857.",
            "The single 20-engine holdout was not sufficiently representative; one split is a study limitation.",
            "11 test engines have uncapped final RUL above 125; capped-truth LSTM RMSE is 13.412 versus 14.857 against uncapped truth.",
            "The early-stage cap mismatch contributes to error but does not explain the full generalization gap.",
        ],
        size=18,
    )

    slide = add_base(prs, "Stage-wise error: 100 final engine observations", "09 · Error analysis", 10)
    add_bullets(
        slide,
        [
            "Early (RUL > 125, n=11): XGBoost 17.59 RMSE; LSTM 26.31.",
            "Middle (50 < RUL ≤ 125, n=56): XGBoost 11.50; LSTM 15.69.",
            "Late (RUL ≤ 50, n=33): XGBoost 4.79; LSTM 4.52.",
            "Unlike Phase 1's all-cycle stage analysis, Phase 2 stage values use one last-cycle prediction per engine.",
        ],
        x=0.8,
        y=1.55,
        w=5.6,
        h=4.9,
        size=17,
    )
    add_picture(slide, "test_stage_errors.png", 6.55, 1.7, 6.0, 4.2)

    slide = add_base(prs, "Limitations and defensible next steps", "10 · Discussion", 11)
    add_bullets(
        slide,
        [
            "FD001 is simulated, with one operating condition and fault mode; transfer to FD002–FD004 and real engines is unverified.",
            "Only one validation split, one final seed, and one LSTM architecture were evaluated.",
            "No Temporal CNN or hybrid model was tested; selecting one against the already-used test set would bias the comparison.",
            "Next: repeated engine-level validation, pre-registered selection, then one untouched evaluation population.",
        ],
        size=18,
    )

    slide = add_base(prs, "Conclusion", "11 · Takeaway", 12)
    add_text(
        slide,
        "The leakage-controlled Phase 1 XGBoost remains the best measured FD001 model.",
        0.9,
        1.8,
        11.5,
        0.9,
        25,
        NAVY,
        True,
    )
    add_text(
        slide,
        "The Phase 2 LSTM pipeline is reproducible and engine-aware, but its validation gain did not translate to the official test set.",
        0.9,
        3.0,
        11.4,
        1.2,
        22,
        INK,
    )
    add_text(
        slide,
        "Report the negative result honestly; test new architectures only with a fresh, untouched final evaluation.",
        0.9,
        4.8,
        11.4,
        1.0,
        20,
        TEAL,
        True,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUTPUT)
    print(f"Saved {OUTPUT.relative_to(ROOT)} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build()
