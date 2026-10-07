"""Build the 2-page project summary PDF (abstract, features, novelty, flow diagram).

Requires reportlab (not a project dependency): pip install reportlab
Usage: python puea_detection/docs/make_summary_pdf.py puea_detection/docs/PUEA_Adaptive_Detection_Summary.pdf
"""

import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

OUT = sys.argv[1]

INK = colors.HexColor("#1f1f1d")
MUTED = colors.HexColor("#5b5a56")
LINE = colors.HexColor("#d9d8d3")
BLUE = colors.HexColor("#2a78d6")
BLUE_BG = colors.HexColor("#eaf2fc")
ORANGE = colors.HexColor("#d9531e")
ORANGE_BG = colors.HexColor("#fdeee6")
GREEN = colors.HexColor("#138a60")
GREEN_BG = colors.HexColor("#e5f5ee")
GREY_BG = colors.HexColor("#f4f3f0")

ss = getSampleStyleSheet()
# All running text (title, headings, body, table, footnote) is plain black;
# colour is kept only for table highlights and the flow diagram.
TEXT = colors.black
TITLE = ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=19, leading=23,
                       textColor=TEXT, spaceAfter=2, alignment=0)
SUB = ParagraphStyle("s", fontName="Helvetica", fontSize=9.5, leading=13, textColor=TEXT, spaceAfter=10)
H = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=TEXT,
                   spaceBefore=9, spaceAfter=4)
BODY = ParagraphStyle("b", fontName="Helvetica", fontSize=9.6, leading=13.6, textColor=TEXT, alignment=TA_JUSTIFY)
BUL = ParagraphStyle("bl", parent=BODY, alignment=0, leftIndent=11, bulletIndent=1, spaceAfter=2.2)
SMALL = ParagraphStyle("sm", fontName="Helvetica", fontSize=8, leading=10.5, textColor=TEXT)
CELL = ParagraphStyle("c", fontName="Helvetica", fontSize=8.6, leading=11, textColor=TEXT)
CELLB = ParagraphStyle("cb", parent=CELL, fontName="Helvetica-Bold")


def bullets(items):
    return [Paragraph(t, BUL, bulletText="•") for t in items]


# ---------------------------------------------------------------------------
# Flow diagram
# ---------------------------------------------------------------------------

class FlowDiagram(Flowable):
    W, H = 170 * mm, 205 * mm

    def wrap(self, *_):
        return self.W, self.H

    # helpers --------------------------------------------------------------
    def box(self, cx, cy, w, h, lines, fill, stroke, bold_first=True, dashed=False, size=8.6):
        c = self.canv
        c.setFillColor(fill)
        c.setStrokeColor(stroke)
        c.setLineWidth(1.2)
        if dashed:
            c.setDash(3, 2)
        c.roundRect(cx - w / 2, cy - h / 2, w, h, 5, stroke=1, fill=1)
        c.setDash()
        c.setFillColor(INK)
        lead = size + 2.6
        y = cy + (len(lines) - 1) * lead / 2 - size * 0.35
        for i, t in enumerate(lines):
            c.setFont("Helvetica-Bold" if (i == 0 and bold_first) else "Helvetica", size if i == 0 else size - 0.6)
            if i > 0:
                c.setFillColor(MUTED)
            c.drawCentredString(cx, y, t)
            y -= lead
        c.setFillColor(INK)

    def diamond(self, cx, cy, w, h, text, stroke=BLUE):
        c = self.canv
        p = c.beginPath()
        p.moveTo(cx, cy + h / 2)
        p.lineTo(cx + w / 2, cy)
        p.lineTo(cx, cy - h / 2)
        p.lineTo(cx - w / 2, cy)
        p.close()
        c.setFillColor(colors.white)
        c.setStrokeColor(stroke)
        c.setLineWidth(1.4)
        c.drawPath(p, stroke=1, fill=1)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 9)
        c.drawCentredString(cx, cy - 3, text)

    def arrow(self, pts, color=MUTED, label=None, label_at=None, dashed=False):
        c = self.canv
        c.setStrokeColor(color)
        c.setLineWidth(1.3)
        if dashed:
            c.setDash(3, 2)
        p = c.beginPath()
        p.moveTo(*pts[0])
        for q in pts[1:]:
            p.lineTo(*q)
        c.drawPath(p, stroke=1, fill=0)
        c.setDash()
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        import math
        a = math.atan2(y2 - y1, x2 - x1)
        s = 6
        c.setFillColor(color)
        ah = c.beginPath()
        ah.moveTo(x2, y2)
        ah.lineTo(x2 - s * math.cos(a - 0.4), y2 - s * math.sin(a - 0.4))
        ah.lineTo(x2 - s * math.cos(a + 0.4), y2 - s * math.sin(a + 0.4))
        ah.close()
        c.drawPath(ah, stroke=0, fill=1)
        if label:
            lx, ly = label_at
            c.setFont("Helvetica-Bold", 8.5)
            c.setFillColor(color)
            c.drawString(lx, ly, label)
            c.setFillColor(INK)

    def draw(self):
        W, H = self.W, self.H
        cx = W * 0.47
        bw = 70 * mm

        y1 = H - 12 * mm
        self.box(cx, y1, bw, 13 * mm, ["1  New sensing slot", "RSS reported by all 40 secondary users"], BLUE_BG, BLUE)
        y2 = y1 - 25 * mm
        self.box(cx, y2, bw, 13 * mm, ["2  Feature extraction", "5 RSS statistics + mean SNR"], BLUE_BG, BLUE)
        self.arrow([(cx, y1 - 6.5 * mm), (cx, y2 + 6.5 * mm)])

        y3 = y2 - 25 * mm
        sw = 44 * mm
        lx, rx = cx - 24 * mm, cx + 24 * mm
        self.box(lx, y3, sw, 13 * mm, ["3a  XGBoost", "static classifier"], BLUE_BG, BLUE)
        self.box(rx, y3, sw, 13 * mm, ["3b  Adaptive PU profile", "Mahalanobis distance"], BLUE_BG, BLUE)
        self.arrow([(cx, y2 - 6.5 * mm), (cx, y2 - 11 * mm), (lx, y2 - 11 * mm), (lx, y3 + 6.5 * mm)])
        self.arrow([(cx, y2 - 11 * mm), (rx, y2 - 11 * mm), (rx, y3 + 6.5 * mm)])

        y4 = y3 - 25 * mm
        self.diamond(cx, y4, 36 * mm, 15 * mm, "PU-like?")
        self.arrow([(lx, y3 - 6.5 * mm), (lx, y3 - 11 * mm), (cx, y3 - 11 * mm), (cx, y4 + 7.5 * mm)])
        self.arrow([(rx, y3 - 6.5 * mm), (rx, y3 - 11 * mm), (cx, y3 - 11 * mm)])
        c = self.canv
        c.setFont("Helvetica-Oblique", 7.6)
        c.setFillColor(MUTED)
        c.drawString(cx + 21 * mm, y4 + 1 * mm, "PU-like = not flagged by both")
        c.drawString(cx + 21 * mm, y4 - 3 * mm, "XGBoost and the profile")

        # NO branch -> PUEA
        nx = 17 * mm
        self.box(nx, y4, 30 * mm, 13 * mm, ["PUEA", "no update"], ORANGE_BG, ORANGE)
        self.arrow([(cx - 18 * mm, y4), (nx + 15 * mm, y4)], ORANGE, "No", (cx - 29 * mm, y4 + 2.5 * mm))

        y5 = y4 - 30 * mm
        self.box(cx, y5, 78 * mm, 16 * mm,
                 ["4  Physical consistency check (the novelty)",
                  "Does the RSS pattern across SUs fit a",
                  "transmitter at the known PU location?"], GREY_BG, INK)
        self.arrow([(cx, y4 - 7.5 * mm), (cx, y5 + 8 * mm)], BLUE, "Yes", (cx + 2 * mm, y4 - 13.5 * mm))

        y6 = y5 - 26 * mm
        self.diamond(cx, y6, 36 * mm, 15 * mm, "Consistent?")
        self.arrow([(cx, y5 - 8 * mm), (cx, y6 + 7.5 * mm)])

        # NO branch -> blocked
        bx = cx + 47 * mm
        self.box(bx, y6, 34 * mm, 16 * mm, ["PUEA / poisoning", "flag it and BLOCK", "the profile update"], ORANGE_BG, ORANGE)
        self.arrow([(cx + 18 * mm, y6), (bx - 17 * mm, y6)], ORANGE, "No", (cx + 19.5 * mm, y6 + 2.5 * mm))

        y7 = y6 - 25 * mm
        self.box(cx, y7, bw, 13 * mm, ["5  Legitimate PU", "signal drift is accepted"], GREEN_BG, GREEN)
        self.arrow([(cx, y6 - 7.5 * mm), (cx, y7 + 6.5 * mm)], GREEN, "Yes", (cx + 2 * mm, y6 - 13.5 * mm))

        y8 = y7 - 22 * mm
        self.box(cx, y8, bw, 13 * mm, ["6  Update PU profile", "small EWMA step (λ = 0.01)"], GREEN_BG, GREEN)
        self.arrow([(cx, y7 - 6.5 * mm), (cx, y8 + 6.5 * mm)], GREEN)

        # feedback loop to the profile (right side)
        fx = W - 5 * mm
        self.arrow([(cx + bw / 2, y8), (fx, y8), (fx, y3), (rx + sw / 2, y3)], GREEN, dashed=True)
        c.setFont("Helvetica-Oblique", 7.6)
        c.setFillColor(GREEN)
        c.saveState()
        c.translate(fx + 3.5 * mm, (y8 + y3) / 2 - 20 * mm)
        c.rotate(90)
        c.drawString(0, 0, "profile adapts to legitimate drift")
        c.restoreState()

        # naive contrast note
        c.setFillColor(MUTED)
        c.setFont("Helvetica-Oblique", 7.6)
        tx = 3 * mm
        for k, line in enumerate(["Naive adaptation skips", "step 4: every PU-like slot",
                                  "updates the profile, so", "an attacker can drag it."]):
            c.drawString(tx, y5 + 5 * mm - k * 3.9 * mm, line)

        # calibration note
        self.box(17 * mm, y8, 32 * mm, 16 * mm, ["Calibration", "PU-only data sets all", "thresholds (1% FA)"],
                 colors.white, MUTED, dashed=True, size=8)


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

def build():
    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=17 * mm, bottomMargin=15 * mm,
                            title="Physically Anchored Adaptive PUEA Detection",
                            author="PUEA Detection Project")
    s = []
    s.append(Paragraph("Physically Anchored Adaptive PUEA Detection", TITLE))
    s.append(Paragraph("Project summary, cognitive radio network security", SUB))

    s.append(Paragraph("Abstract", H))
    s.append(Paragraph(
        "Primary user emulation attacks (PUEA) allow a malicious node to imitate the licensed primary user "
        "(PU) so that secondary users (SUs) give up the channel. Most learning-based detectors, including the "
        "one-class approach of Chhetry and Marchang (2021), are trained once on a fixed description of PU "
        "behaviour. In practice the PU's received signal changes over time with transmit power, shadowing and "
        "noise, which leads to false alarms, and retraining the model on new data makes it possible for an "
        "attacker to poison it with gradually more PU-like signals. In this project we maintain an adaptive "
        "profile of the PU and only allow it to be updated when the signal strengths measured across the SUs "
        "are consistent with a transmitter at the known PU location. We evaluated the approach on a simulated "
        "network of 40 SUs. The adaptive detector kept the false alarm rate at about 2% under legitimate drift "
        "and prevented poisoning when the attacker was roughly 10 m or more from the PU, but it gave little "
        "protection against attackers closer than this.", BODY))

    s.append(Paragraph("Implementation", H))
    s += bullets([
        "Baseline classifiers (KNN, SVM, ANN, random forest, XGBoost and a stacking ensemble) on the original "
        "dataset, with SHAP analysis. XGBoost performed best with 92.2% accuracy and an F1 score of 0.918.",
        "Eleven test scenarios: a stable PU, four types of legitimate drift, four attack types (basic, power "
        "matching, gradual and stealthy) and two poisoning attacks in which the attacker reacts to the "
        "detector's decisions.",
        "A physical consistency check that compares how well the received signals fit a transmitter at the PU "
        "location with the best fit anywhere in the area.",
        "An adaptive PU profile that can only be updated through this check.",
        "An ablation study with eight detector variants, two network settings and five random seeds. All "
        "thresholds were calibrated on PU-only data.",
    ])

    s.append(Paragraph("Novelty", H))
    s += bullets([
        "We treat adaptive PUEA detection as the problem of separating legitimate drift from poisoning. "
        "Earlier PUEA detectors, including our baseline paper, rely on a fixed PU profile.",
        "Profile updates depend on physical consistency with the PU location. An observation that "
        "statistically resembles the PU but does not fit its location is rejected and treated as an attack.",
        "We measured how close an attacker has to be before the physical check stops working, so the limits "
        "of the method are reported together with the results.",
    ])

    s.append(Paragraph("Key results", H))
    CELLC = ParagraphStyle("cc", parent=CELL, alignment=1)
    CELLBC = ParagraphStyle("cbc", parent=CELLB, alignment=1)

    def row(test, measure, static, naive, ours):
        return [Paragraph(test, CELL), Paragraph(measure, CELL),
                Paragraph(static, CELLC), Paragraph(naive, CELLC), Paragraph(ours, CELLBC)]

    rows = [
        [Paragraph("Test", CELLB), Paragraph("What is measured", CELLB), Paragraph("Static XGBoost", CELLBC),
         Paragraph("Naive adaptive", CELLBC), Paragraph("Our method", CELLBC)],
        row("The PU's transmit power slowly increases (no attack)",
            "False alarms (lower is better)", "83%", "1%", "2%"),
        row("An attacker copies the PU's signal strength from a different location",
            "Attacks detected (higher is better)", "61%", "1%", "100%"),
        row("An attacker slowly poisons the PU profile, then attacks",
            "Attacks detected after poisoning", "100%", "0.1%", "100%"),
        row("Same poisoning attack, but the attacker is very close to the PU and the network is sparser",
            "Attacks detected after poisoning", "99%", "1%", "13%"),
    ]
    t = Table(rows, colWidths=[62 * mm, 42 * mm, 22 * mm, 22 * mm, 22 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), GREY_BG),
        ("BACKGROUND", (4, 1), (4, 3), BLUE_BG),
        ("BACKGROUND", (0, 4), (-1, 4), ORANGE_BG),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    s.append(t)
    s.append(Spacer(1, 2.5 * mm))
    s.append(Paragraph(
        "Averages over five simulation runs. The last row is the main limitation: when the attacker is too close "
        "to the PU, the physical check cannot tell the attacker apart from the PU. In most tests the physical check on its own "
        "performed about as well as the full method.", SMALL))

    s.append(PageBreak())
    s.append(Paragraph("How the detector works", H))
    s.append(Paragraph(
        "Each sensing slot is processed as shown below. Step 4 is the part we added: an observation has to "
        "pass it before it can be used to update the PU profile.",
        BODY))
    s.append(Spacer(1, 5 * mm))
    s.append(KeepTogether([FlowDiagram()]))
    s.append(Spacer(1, 3 * mm))
    s.append(Paragraph(
        "PU = primary user (licensed transmitter) · SU = secondary user · RSS = received signal strength · "
        "EWMA = exponentially weighted moving average · FA = false-alarm rate.", SMALL))
    doc.build(s)


build()
print("wrote", OUT)
