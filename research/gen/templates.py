"""Render fictional invoices and marksheets with Pillow (via research.gen.fonts).

Each render returns a `Page`: the RGB pixel array plus a list of every text field drawn
(text, font, size, anchor, colour, ink box, and whether it is a value that a forger might
edit). The tamper step uses that field list to pick what to overwrite.

Layouts vary by seed: header style, logo, title position, table style (grid / rows / zebra),
colours, margins, font families and sizes, number of rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from research.gen import fake
from research.gen.fake import pick
from research.gen.fonts import FontRegistry, composite, get_font, render_text_alpha

INK_COLORS = [(20, 20, 20), (35, 35, 45), (15, 25, 60), (45, 40, 40)]
ACCENTS = [(31, 78, 121), (120, 30, 30), (20, 90, 60), (60, 60, 60), (90, 50, 120), (10, 10, 10)]
RESULTS = ["PASS", "FIRST DIVISION", "SECOND DIVISION", "DISTINCTION", "FAIL"]


@dataclass
class TextField:
    """One piece of text drawn on the page."""
    name: str
    kind: str                # amount | int | date | name | id | grade | result | percent | text
    text: str
    family: str
    bold: bool
    size: int
    xy: tuple[float, float]  # anchor point
    anchor: str              # Pillow anchor: "la" (left), "ra" (right), "ma" (centre)
    color: tuple[int, int, int]
    bbox: tuple[int, int, int, int]  # ink box xyxy (exclusive), page coordinates
    editable: bool
    meta: dict = field(default_factory=dict)  # e.g. {"fmt": "%d/%m/%Y"} or {"lo": 1, "hi": 50}


@dataclass
class Page:
    doc_type: str
    image: np.ndarray        # HxWx3 uint8 RGB
    fields: list[TextField]
    body_family: str
    header_family: str
    layout: dict             # layout choices, recorded in metadata

    def editable_fields(self) -> list[TextField]:
        return [f for f in self.fields if f.editable]


class Canvas:
    """A paper-coloured numpy page with helpers that record every text field drawn."""

    def __init__(self, width, height, paper, noise_std, rng, fonts: FontRegistry):
        img = np.empty((height, width, 3), np.float32)
        img[:] = paper
        if noise_std > 0:
            img += rng.normal(0.0, noise_std, size=(height, width, 1))
        self.img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
        self.fonts = fonts
        self.fields: list[TextField] = []

    def text(self, x, y, text, family, size, color, bold=False, anchor="la",
             name="", kind="text", editable=False, **meta) -> TextField:
        ta = render_text_alpha(text, self.fonts.path(family, bold), int(size), (x, y), anchor)
        composite(self.img, ta, color)
        f = TextField(name or text[:20], kind, text, family, bold, int(size), (float(x), float(y)),
                      anchor, tuple(int(c) for c in color), ta.bbox, editable, dict(meta))
        self.fields.append(f)
        return f

    def line(self, x0, y0, x1, y1, color, thickness=1):
        cv2.line(self.img, (int(x0), int(y0)), (int(x1), int(y1)), tuple(int(c) for c in color),
                 int(thickness), lineType=cv2.LINE_AA)

    def rect(self, x0, y0, x1, y1, color, thickness=-1):
        cv2.rectangle(self.img, (int(x0), int(y0)), (int(x1), int(y1)),
                      tuple(int(c) for c in color), int(thickness))

    def circle(self, cx, cy, r, color, thickness=-1):
        cv2.circle(self.img, (int(cx), int(cy)), int(r), tuple(int(c) for c in color),
                   int(thickness), lineType=cv2.LINE_AA)


def _tint(color, amount: float, paper):
    """Mix `color` towards the paper colour (amount=0 -> color, 1 -> paper)."""
    return tuple(int(round(c * (1 - amount) + p * amount)) for c, p in zip(color, paper))


def _setup(rng, page_cfg, render_cfg, families):
    """Pick the page-wide style: paper, ink, accent, fonts, sizes, margin."""
    paper = tuple(int(v) for v in (rng.integers(244, 256), rng.integers(242, 256), rng.integers(236, 254)))
    noise = float(rng.uniform(*render_cfg["paper_noise_std"]))
    body = pick(rng, families)
    header = pick(rng, families) if rng.random() < render_cfg["secondary_header_font_prob"] else body
    style = dict(paper=paper, noise_std=round(noise, 3), ink=pick(rng, INK_COLORS),
                 accent=pick(rng, ACCENTS), body_family=body, header_family=header,
                 s_body=int(rng.integers(18, 23)), s_title=int(rng.integers(40, 56)),
                 s_org=int(rng.integers(30, 40)), margin=int(rng.integers(70, 110)))
    style["s_small"] = style["s_body"] - 2
    return style


def render_invoice(rng, page_cfg, render_cfg, fonts: FontRegistry, families) -> Page:
    """Render a fictional invoice. All names, numbers and IDs come from research.gen.fake."""
    W, H = page_cfg["width"], page_cfg["height"]
    st = _setup(rng, page_cfg, render_cfg, families)
    c = Canvas(W, H, st["paper"], st["noise_std"], rng, fonts)
    ink, acc, paper, m = st["ink"], st["accent"], st["paper"], st["margin"]
    body, head, sb, ss = st["body_family"], st["header_family"], st["s_body"], st["s_small"]
    layout = dict(header_style=pick(rng, ["band", "plain", "line"]),
                  table_style=pick(rng, ["grid", "rows", "zebra"]),
                  title_pos=pick(rng, ["right", "center"]), logo=bool(rng.random() < 0.5))
    date_fmt = pick(rng, fake.DATE_FORMATS)
    org = fake.org_name(rng)

    # --- Header --------------------------------------------------------------------------
    y = m
    head_color = ink
    if layout["header_style"] == "band":
        band_h = m + 150
        c.rect(0, 0, W, band_h, acc)
        head_color = paper
        y = m - 30
    x_org = m
    if layout["logo"]:
        r = int(rng.integers(32, 44))
        logo_color = paper if layout["header_style"] == "band" else acc
        c.circle(m + r, y + r, r, logo_color)
        initials = "".join(w[0] for w in org.split()[:2])
        c.text(m + r, y + r - 12, initials, head, 24, acc if logo_color == paper else paper,
               bold=True, anchor="ma", name="logo")
        x_org = m + 2 * r + 20
    c.text(x_org, y, org, head, st["s_org"], head_color, bold=True, name="org_name")
    yy = y + st["s_org"] + 14
    for i, line in enumerate(fake.address_lines(rng)):
        c.text(x_org, yy, line, body, ss, head_color, name=f"org_addr{i}")
        yy += ss + 8
    c.text(x_org, yy, f"Phone: {fake.phone(rng)}", body, ss, head_color, name="org_phone")
    yy += ss + 8
    c.text(x_org, yy, "Tax ID: ", body, ss, head_color, name="org_taxid_label")
    c.text(x_org + 80, yy, fake.tax_id(rng), body, ss, head_color, name="org_taxid", kind="id",
           editable=layout["header_style"] != "band")
    title = pick(rng, ["TAX INVOICE", "INVOICE", "BILL OF SUPPLY"])
    if layout["title_pos"] == "right":
        c.text(W - m, y, title, head, st["s_title"], head_color if layout["header_style"] == "band" else acc,
               bold=True, anchor="ra", name="title")
        y = max(yy + ss + 40, m + 190)
    else:
        y = max(yy + ss + 40, m + 190)
        c.text(W // 2, y, title, head, st["s_title"], acc, bold=True, anchor="ma", name="title")
        y += st["s_title"] + 30
    if layout["header_style"] == "line":
        c.line(m, y - 15, W - m, y - 15, acc, 3)

    # --- Bill-to (left) and invoice meta (right) --------------------------------------------
    y0 = y + 10
    c.text(m, y0, "Bill To:", head, sb, acc, bold=True, name="billto_label")
    yl = y0 + sb + 12
    c.text(m, yl, fake.person_name(rng), body, sb, ink, name="customer_name", kind="name", editable=True)
    yl += sb + 10
    for i, line in enumerate(fake.address_lines(rng)):
        c.text(m, yl, line, body, ss, ink, name=f"cust_addr{i}")
        yl += ss + 8
    c.text(m, yl, f"Tax ID: {fake.tax_id(rng)}", body, ss, ink, name="cust_taxid")
    yl += ss + 8
    xl, xr = W // 2 + 110, W - m
    yr = y0
    meta_rows = [("Invoice No:", fake.invoice_number(rng), "invoice_no", "id", {}),
                 ("Invoice Date:", fake.random_date(rng, date_fmt), "invoice_date", "date", {"fmt": date_fmt}),
                 ("Due Date:", fake.random_date(rng, date_fmt), "due_date", "date", {"fmt": date_fmt}),
                 ("PO Number:", f"PO-{fake.digits(rng, 6)}", "po_number", "id", {})]
    for label, value, name, kind, meta in meta_rows:
        c.text(xl, yr, label, body, sb, ink, bold=True, name=name + "_label")
        c.text(xr, yr, value, body, sb, ink, anchor="ra", name=name, kind=kind, editable=True, **meta)
        yr += sb + 14
    y = max(yl, yr) + 40

    # --- Items table ---------------------------------------------------------------------
    amt_r = W - m - 12
    unit_r = amt_r - 220
    qty_r = unit_r - 190
    desc_x, num_x = m + 62, m + 12
    rh = sb + int(rng.integers(16, 26))
    hdr = [("#", num_x, "la"), ("Description", desc_x, "la"), ("Qty", qty_r, "ra"),
           ("Unit Price", unit_r, "ra"), ("Amount", amt_r, "ra")]
    hdr_filled = layout["table_style"] != "rows" or rng.random() < 0.5
    if hdr_filled:
        c.rect(m, y, W - m, y + rh, acc)
    else:
        c.line(m, y + rh, W - m, y + rh, ink, 2)
    for label, x, anc in hdr:
        c.text(x, y + (rh - sb) // 2, label, head, sb, paper if hdr_filled else ink, bold=True,
               anchor=anc, name=f"th_{label}")
    y_tab0 = y
    y += rh
    n_items = int(rng.integers(3, 9))
    subtotal = 0.0
    zebra = _tint(acc, 0.9, paper)
    for i in range(n_items):
        if layout["table_style"] == "zebra" and i % 2 == 1:
            c.rect(m, y, W - m, y + rh, zebra)
        qty = int(rng.integers(1, 51))
        unit = round(float(rng.uniform(50, 20000)), 2)
        amt = round(qty * unit, 2)
        subtotal += amt
        ty = y + (rh - sb) // 2
        c.text(num_x, ty, str(i + 1), body, sb, ink, name=f"item{i}_num")
        c.text(desc_x, ty, pick(rng, fake.PRODUCTS), body, sb, ink, name=f"item{i}_desc")
        c.text(qty_r, ty, str(qty), body, sb, ink, anchor="ra", name=f"item{i}_qty", kind="int",
               editable=True, lo=1, hi=99)
        c.text(unit_r, ty, fake.amount(unit), body, sb, ink, anchor="ra", name=f"item{i}_unit",
               kind="amount", editable=True)
        c.text(amt_r, ty, fake.amount(amt), body, sb, ink, anchor="ra", name=f"item{i}_amount",
               kind="amount", editable=True)
        y += rh
        if layout["table_style"] in ("rows", "grid"):
            c.line(m, y, W - m, y, _tint(ink, 0.5, paper), 1)
    if layout["table_style"] == "grid":
        for x in (m, desc_x - 12, qty_r - 110, qty_r + 14, unit_r + 14, W - m):
            c.line(x, y_tab0, x, y, _tint(ink, 0.5, paper), 1)
        c.line(m, y_tab0, W - m, y_tab0, _tint(ink, 0.5, paper), 1)

    # --- Totals --------------------------------------------------------------------------
    y += 24
    rate = pick(rng, [5, 12, 18])
    tax = round(subtotal * rate / 100, 2)
    rows = [("Subtotal", subtotal, "subtotal"), (f"GST @ {rate}%", tax, "tax")]
    total = subtotal + tax
    if rng.random() < 0.3:
        disc = round(subtotal * float(rng.uniform(0.02, 0.1)), 2)
        rows.append(("Discount", disc, "discount"))
        total -= disc
    # Right edge of the labels: leave room for the widest (bold, larger) total amount.
    widest = get_font(fonts.path(body, True), sb + 4).getlength(fake.amount(max(total, subtotal)))
    lab_r = int(min(unit_r - 10, amt_r - widest - 30))
    for label, val, name in rows:
        c.text(lab_r, y, label, body, sb, ink, anchor="ra", name=name + "_label")
        c.text(amt_r, y, fake.amount(val), body, sb, ink, anchor="ra", name=name, kind="amount", editable=True)
        y += sb + 14
    c.line(lab_r - 160, y, W - m, y, ink, 2)
    y += 12
    c.text(lab_r, y, "Total", head, sb + 4, ink, bold=True, anchor="ra", name="total_label")
    c.text(amt_r, y, fake.amount(total), body, sb + 4, ink, bold=True, anchor="ra", name="total",
           kind="amount", editable=True)
    y += sb + 60

    # --- Footer --------------------------------------------------------------------------
    c.text(m, y, f"Payment Terms: Net {pick(rng, [7, 15, 30, 45])} days", body, ss, ink, name="terms")
    y += ss + 10
    c.text(m, y, "Bank A/c No:", body, ss, ink, name="bank_label")
    c.text(m + 150, y, fake.digits(rng, 12), body, ss, ink, name="bank_account", kind="id", editable=True)
    y += ss + 10
    c.text(m, y, f"IFSC: {fake.letters(rng, 4)}0{fake.digits(rng, 6)}", body, ss, ink, name="ifsc")
    y += ss + 40
    c.text(m, y, pick(rng, ["Thank you for your business!", "Goods once sold will not be taken back.",
                            "Please quote the invoice number with your payment."]), body, ss, ink, name="note")
    sig_y = min(max(y + 40, H - m - 120), H - m - 60)
    c.line(W - m - 300, sig_y, W - m, sig_y, ink, 1)
    c.text(W - m - 150, sig_y + 10, "Authorised Signatory", body, ss, ink, anchor="ma", name="sig")
    c.text(W - m - 150, sig_y + 10 + ss + 8, f"for {org}", body, ss - 2, ink, anchor="ma", name="sig_org")
    return Page("invoice", c.img, c.fields, body, head, layout)


def _grade(pct: float) -> str:
    for g, lo in zip(fake.GRADES, [90, 80, 70, 60, 50, 0]):
        if pct >= lo:
            return g
    return "D"


def render_marksheet(rng, page_cfg, render_cfg, fonts: FontRegistry, families) -> Page:
    """Render a fictional statement of marks."""
    W, H = page_cfg["width"], page_cfg["height"]
    st = _setup(rng, page_cfg, render_cfg, families)
    c = Canvas(W, H, st["paper"], st["noise_std"], rng, fonts)
    ink, acc, paper, m = st["ink"], st["accent"], st["paper"], st["margin"]
    body, head, sb, ss = st["body_family"], st["header_family"], st["s_body"], st["s_small"]
    layout = dict(border=pick(rng, ["double", "single", "none"]),
                  table_style=pick(rng, ["grid", "rows", "zebra"]),
                  emblem=bool(rng.random() < 0.6), details_cols=int(rng.integers(1, 3)))
    date_fmt = pick(rng, fake.DATE_FORMATS)

    if layout["border"] != "none":
        c.rect(m // 2, m // 2, W - m // 2, H - m // 2, acc, 3)
        if layout["border"] == "double":
            c.rect(m // 2 + 8, m // 2 + 8, W - m // 2 - 8, H - m // 2 - 8, acc, 1)

    # --- Header (centred) ------------------------------------------------------------------
    y = m + 10
    if layout["emblem"]:
        r = int(rng.integers(36, 50))
        c.circle(W // 2, y + r, r, acc, 4)
        c.circle(W // 2, y + r, r - 12, _tint(acc, 0.6, paper))
        y += 2 * r + 20
    inst = fake.institution(rng)
    c.text(W // 2, y, inst, head, st["s_org"], acc, bold=True, anchor="ma", name="institution")
    y += st["s_org"] + 16
    c.text(W // 2, y, f"{fake.place(rng)} - {fake.digits(rng, 6)}", body, ss, ink, anchor="ma", name="inst_addr")
    y += ss + 22
    title = pick(rng, ["STATEMENT OF MARKS", "GRADE CARD", "MARKS MEMORANDUM", "PROVISIONAL MARKSHEET"])
    c.text(W // 2, y, title, head, sb + 10, ink, bold=True, anchor="ma", name="title")
    y += sb + 24
    year = int(rng.integers(2019, 2027))
    exam = pick(rng, [f"Semester {pick(rng, ['I', 'II', 'III', 'IV', 'V', 'VI'])} Examination, "
                      f"{pick(rng, ['May', 'December', 'April', 'November'])} {year}",
                      f"Annual Examination {year}"])
    c.text(W // 2, y, exam, body, sb, ink, anchor="ma", name="exam")
    y += sb + 30
    c.line(m, y, W - m, y, acc, 2)
    y += 30

    # --- Student details -----------------------------------------------------------------
    details = [("Name of Candidate", fake.person_name(rng), "student_name", "name", {}),
               ("Roll No", fake.digits(rng, int(rng.integers(7, 11))), "roll_no", "id", {}),
               ("Registration No", f"{year}{fake.letters(rng, 2)}{fake.digits(rng, 5)}", "reg_no", "id", {}),
               ("Guardian's Name", fake.person_name(rng), "guardian_name", "name", {}),
               ("Date of Birth", fake.random_date(rng, date_fmt, 1995, 2010), "dob", "date", {"fmt": date_fmt}),
               ("Programme", pick(rng, fake.PROGRAMS), "programme", "text", {})]
    label_w = 230
    if layout["details_cols"] == 1:
        for label, value, name, kind, meta in details:
            c.text(m + 20, y, label, body, sb, ink, bold=True, name=name + "_label")
            c.text(m + 20 + label_w, y, ":", body, sb, ink, name=name + "_colon")
            c.text(m + 20 + label_w + 20, y, value, body, sb, ink, name=name, kind=kind,
                   editable=kind != "text", **meta)
            y += sb + 16
    else:
        col_x = [m + 20, W // 2 + 20]
        for i, (label, value, name, kind, meta) in enumerate(details):
            x = col_x[i % 2]
            c.text(x, y, label + ":", body, ss, ink, bold=True, name=name + "_label")
            c.text(x, y + ss + 6, value, body, sb, ink, name=name, kind=kind, editable=kind != "text", **meta)
            if i % 2 == 1:
                y += ss + sb + 26
        if len(details) % 2:
            y += ss + sb + 26
    y += 24

    # --- Marks table ---------------------------------------------------------------------
    grade_r = W - m - 20
    obt_r = grade_r - 150
    max_r = obt_r - 170
    code_x, subj_x, num_x = m + 70, m + 210, m + 14
    rh = sb + int(rng.integers(16, 26))
    filled = layout["table_style"] != "rows"
    if filled:
        c.rect(m, y, W - m, y + rh, acc)
    else:
        c.line(m, y + rh, W - m, y + rh, ink, 2)
    for label, x, anc in [("S.No", num_x, "la"), ("Code", code_x, "la"), ("Subject", subj_x, "la"),
                          ("Max", max_r, "ra"), ("Obtained", obt_r, "ra"), ("Grade", grade_r, "ra")]:
        c.text(x, y + (rh - ss) // 2, label, head, ss, paper if filled else ink, bold=True,
               anchor=anc, name=f"th_{label}")
    y_tab0 = y
    y += rh
    subjects = list(rng.choice(fake.SUBJECTS, size=int(rng.integers(5, 9)), replace=False))
    total_max = total_obt = 0
    zebra = _tint(acc, 0.9, paper)
    for i, subj in enumerate(subjects):
        if layout["table_style"] == "zebra" and i % 2 == 1:
            c.rect(m, y, W - m, y + rh, zebra)
        mx = int(pick(rng, [100, 100, 100, 50]))
        ob = int(rng.integers(int(0.3 * mx), mx + 1))
        total_max += mx
        total_obt += ob
        ty = y + (rh - sb) // 2
        c.text(num_x, ty, str(i + 1), body, sb, ink, name=f"subj{i}_num")
        c.text(code_x, ty, f"{fake.letters(rng, 3)}{fake.digits(rng, 3)}", body, sb, ink, name=f"subj{i}_code")
        c.text(subj_x, ty, str(subj), body, sb, ink, name=f"subj{i}_name")
        c.text(max_r, ty, str(mx), body, sb, ink, anchor="ra", name=f"subj{i}_max")
        c.text(obt_r, ty, str(ob), body, sb, ink, anchor="ra", name=f"subj{i}_obt", kind="int",
               editable=True, lo=int(0.3 * mx), hi=mx)
        c.text(grade_r, ty, _grade(100 * ob / mx), body, sb, ink, anchor="ra", name=f"subj{i}_grade",
               kind="grade", editable=True)
        y += rh
        if layout["table_style"] in ("rows", "grid"):
            c.line(m, y, W - m, y, _tint(ink, 0.5, paper), 1)
    if layout["table_style"] == "grid":
        for x in (m, code_x - 12, subj_x - 12, max_r - 90, max_r + 14, obt_r + 14, W - m):
            c.line(x, y_tab0, x, y, _tint(ink, 0.5, paper), 1)
        c.line(m, y_tab0, W - m, y_tab0, _tint(ink, 0.5, paper), 1)
    ty = y + (rh - sb) // 2
    c.text(subj_x, ty, "Total", head, sb, ink, bold=True, name="total_label")
    c.text(max_r, ty, str(total_max), body, sb, ink, bold=True, anchor="ra", name="total_max")
    c.text(obt_r, ty, str(total_obt), body, sb, ink, bold=True, anchor="ra", name="total_obt",
           kind="int", editable=True, lo=0, hi=total_max)
    y += rh + 30

    # --- Summary -------------------------------------------------------------------------
    pct = 100 * total_obt / total_max
    result = "FAIL" if pct < 40 else ("DISTINCTION" if pct >= 75 else
                                      "FIRST DIVISION" if pct >= 60 else "SECOND DIVISION")
    for label, value, name, kind in [("Percentage", f"{pct:.2f}%", "percentage", "percent"),
                                     ("Result", result, "result", "result")]:
        c.text(m + 20, y, f"{label}:", body, sb, ink, bold=True, name=name + "_label")
        c.text(m + 20 + label_w, y, value, body, sb, ink, bold=True, name=name, kind=kind, editable=True)
        y += sb + 18
    y += 20
    c.text(m + 20, y, "Date of Issue:", body, ss, ink, name="issue_label")
    c.text(m + 20 + label_w, y, fake.random_date(rng, date_fmt), body, ss, ink, name="issue_date",
           kind="date", editable=True, fmt=date_fmt)
    y += ss + 12
    c.text(m + 20, y, f"Place: {fake.place(rng)}", body, ss, ink, name="place")

    sig_y = min(max(y + 120, H - m - 160), H - m - 80)
    for x, label in [(m + 170, "Principal"), (W - m - 170, "Controller of Examinations")]:
        c.line(x - 150, sig_y, x + 150, sig_y, ink, 1)
        c.text(x, sig_y + 10, label, body, ss, ink, anchor="ma", name="sig_" + label[:5])
    c.text(W // 2, H - m - 30, "This is a computer-generated statement for demonstration only.",
           body, ss - 4, _tint(ink, 0.4, paper), anchor="ma", name="disclaimer")
    return Page("marksheet", c.img, c.fields, body, head, layout)


RENDERERS = {"invoice": render_invoice, "marksheet": render_marksheet}
