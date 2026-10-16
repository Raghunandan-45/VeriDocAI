"""Fictional content for synthetic documents: names, organisations, places, amounts, dates, IDs.

Names are random first/last combinations, places and organisations are built from invented
word parts, and all numbers are random. Nothing refers to a real person, firm or record.
Every function takes an explicit numpy Generator.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

FIRST = ["Aarav", "Diya", "Kabir", "Meera", "Rohan", "Ananya", "Vikram", "Ishita", "Arjun",
         "Nisha", "Farhan", "Leela", "Sameer", "Tara", "Kiran", "Zoya", "Devansh", "Priya",
         "Omar", "Sana", "Rahul", "Aditi", "Neel", "Kavya", "Yash", "Ritu", "Manav", "Pooja",
         "Elena", "Daniel", "Grace", "Lucas", "Maya", "Noah", "Sofia", "Ethan"]
LAST = ["Sharma", "Iyer", "Khan", "Patel", "Nair", "Reddy", "Menon", "Gupta", "Das", "Rao",
        "Joshi", "Bose", "Kulkarni", "Pillai", "Singh", "Mehta", "Chopra", "Varma", "Shetty",
        "Fernandes", "Mathew", "Hegde", "Saxena", "Bhat", "Thomas", "Walker", "Moreau"]
ORG_A = ["Blue", "Vertex", "Nimbus", "Saffron", "Lotus", "Apex", "Coral", "Orbit", "Maple",
         "Indigo", "Zenith", "Harbor", "Cedar", "Quartz", "Silverline", "Monsoon", "Kestrel"]
ORG_B = ["Traders", "Systems", "Enterprises", "Supplies", "Logistics", "Textiles", "Foods",
         "Hardware", "Printworks", "Electricals", "Solutions", "Agencies", "Industries"]
ORG_SUFFIX = ["Pvt. Ltd.", "LLP", "& Co.", "Ltd.", ""]
SYL = ["Ra", "Ven", "Kor", "Sha", "Mal", "Dev", "Tir", "Pal", "Nar", "Sur", "Ban", "Kal",
       "Vel", "Am", "Sit", "Har", "Mor", "Lin"]
PLACE_END = ["pur", "nagar", "abad", "garh", "wadi", "kota", "gram", "ville"]
STREET = ["Station Road", "Lake View", "MG Lane", "Market Street", "Temple Road",
          "Park Avenue", "Hill Road", "Canal Street", "Mill Lane", "Ring Road"]
PRODUCTS = ["A4 Copier Paper (500 sheets)", "LED Panel Light 18W", "Steel Cabinet 4-Drawer",
            "Cotton Fabric Roll", "Ball Bearing 6204", "Printer Toner Cartridge",
            "Office Chair Ergonomic", "Network Cable Cat6 (305 m)", "Packing Tape 48 mm",
            "Laptop Stand Aluminium", "Hand Sanitizer 5 L", "Copper Wire 2.5 sq mm",
            "Whiteboard Marker Set", "PVC Pipe 1 inch", "Safety Helmet", "Annual Maintenance",
            "Installation Charges", "Consulting Services", "Basmati Rice 25 kg",
            "Stainless Steel Bolt M8", "USB Keyboard", "Wall Clock", "Desk Lamp", "Courier Charges"]
SUBJECTS = ["Mathematics", "Physics", "Chemistry", "English", "Computer Science", "Biology",
            "Economics", "History", "Geography", "Statistics", "Data Structures",
            "Digital Electronics", "Engineering Mechanics", "Environmental Studies",
            "Accountancy", "Business Studies", "Hindi", "Political Science"]
PROGRAMS = ["Bachelor of Science", "Bachelor of Technology", "Bachelor of Commerce",
            "Senior Secondary Certificate", "Higher Secondary Certificate", "Diploma in Engineering"]
DATE_FORMATS = ["%d/%m/%Y", "%d-%b-%Y", "%d.%m.%Y", "%B %d, %Y"]
GRADES = ["A+", "A", "B+", "B", "C", "D"]


def pick(rng: np.random.Generator, items):
    """Choose one element of a list with a numpy Generator."""
    return items[int(rng.integers(len(items)))]


def person_name(rng) -> str:
    return f"{pick(rng, FIRST)} {pick(rng, LAST)}"


def place(rng) -> str:
    return pick(rng, SYL) + pick(rng, SYL).lower() + pick(rng, PLACE_END)


def org_name(rng) -> str:
    return " ".join(x for x in [pick(rng, ORG_A), pick(rng, ORG_B), pick(rng, ORG_SUFFIX)] if x)


def institution(rng) -> str:
    kind = pick(rng, ["Institute of Technology", "University", "College of Arts and Science",
                      "Board of Secondary Education", "Public School"])
    return f"{place(rng)} {kind}"


def address_lines(rng) -> list[str]:
    return [f"{int(rng.integers(1, 999))}, {pick(rng, STREET)}",
            f"{place(rng)} - {int(rng.integers(100000, 999999))}"]


def amount(value: float) -> str:
    """Currency formatting used everywhere: 'Rs. 12,345.00' (decision D2, no rupee glyph)."""
    return f"Rs. {value:,.2f}"


def parse_amount(text: str) -> float:
    return float(text.replace("Rs.", "").replace(",", "").strip())


def random_date(rng, fmt: str, start_year: int = 2019, end_year: int = 2026) -> str:
    start = dt.date(start_year, 1, 1).toordinal()
    end = dt.date(end_year, 12, 31).toordinal()
    return dt.date.fromordinal(int(rng.integers(start, end))).strftime(fmt)


def digits(rng, n: int) -> str:
    return "".join(str(int(d)) for d in rng.integers(0, 10, n))


def letters(rng, n: int) -> str:
    return "".join(chr(65 + int(c)) for c in rng.integers(0, 26, n))


def invoice_number(rng) -> str:
    return pick(rng, [f"INV-{int(rng.integers(2019, 2027))}-{digits(rng, 5)}",
                      f"{letters(rng, 3)}/{digits(rng, 4)}/{int(rng.integers(19, 27))}",
                      f"#{digits(rng, 6)}"])


def tax_id(rng) -> str:
    """Fictional tax registration number (random letters/digits)."""
    return f"{digits(rng, 2)}{letters(rng, 5)}{digits(rng, 4)}{letters(rng, 1)}{digits(rng, 1)}Z{digits(rng, 1)}"


def phone(rng) -> str:
    return f"+91 {digits(rng, 5)} {digits(rng, 5)}"


def perturb_digits(text: str, rng, n_changes: int | None = None) -> str:
    """Change 1-3 digits of `text` to different digits, keeping every other character."""
    pos = [i for i, c in enumerate(text) if c.isdigit()]
    if not pos:
        return text + str(int(rng.integers(0, 10)))
    n = n_changes or int(rng.integers(1, min(3, len(pos)) + 1))
    chars = list(text)
    for i in rng.choice(pos, size=min(n, len(pos)), replace=False):
        chars[i] = str((int(chars[i]) + int(rng.integers(1, 10))) % 10)
    return "".join(chars)
