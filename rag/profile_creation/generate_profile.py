"""
generate_profile.py
────────────────────────────────────────────────────────────────────────────────
Generates a personalised Cai voice assistant profile from a questionnaire row.

HOW TO USE:
    1. Edit the three values in the CONFIG section below.
    2. Run:  python generate_profile.py
────────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — edit these three values before running
# ══════════════════════════════════════════════════════════════════════════════

# Path to your questionnaire export file (.xlsx or .csv)
INPUT_FILE = "responses.csv"

# Row number of the participant (1 = first data row, not the header)
PARTICIPANT_ROW = 2

# Where to save the profile.
# Leave as "" to auto-name it profile_P01.txt, profile_P02.txt etc.
OUTPUT_FILE = "/rag/profile/p2_sample_profile.txt"

# ══════════════════════════════════════════════════════════════════════════════


# ── COLUMN NAME MAP ───────────────────────────────────────────────────────────
# Maps short keys to the exact column headers in the questionnaire export.
# Non-breaking spaces and trailing spaces are handled automatically.

COL = {
    "name": "How would you prefer to be called by?",
    "age": "How old are you?",
    "gender": "Please indicate your gender.",
    "city": "Which city in UK do you live in?",
    "marital": "Please indicate your marital status.",
    "family": "Would you like to tell us more about your immediate family?",
    "had_pets": "Have you ever had pets?",
    "pets_detail": "If you are comfortable sharing, please tell us more about your pets.",
    "important_people": "Is there anything you would like the voice assistant to know about important people in your life (such as close friends or family members) to make conversations more personal?",
    "education": "Please indicate your highest level of education.",
    "has_work": "Do you have any work experience?",
    "work_detail": "If you have worked, would you like to describe about your job or area of expertise?",
    "life_20s40s": "If you have not worked, would you like to share how was you life like during your 20s to 40s?",
    "typical_day": "Can you tell us what a typical day is like for you?",
    "activities_social": "Please mention a few activities you would like to do with your family and friends.",
    "activities_solo": "Please mention a few other activities that you would like to do on your own time.",
    "activities_dislike": "Please mention a few activities that you dislike or prefer not to do in your daily life.",
    "topics_new": "What topics do you like to talk about when you meet a new person for the first time.",
    "topics_known": "What topics do you like to talk about with people you know?",
}


# ── COLUMN NORMALISATION ──────────────────────────────────────────────────────

def normalize(s):
    """Strip whitespace and non-breaking spaces for fuzzy column matching."""
    return s.replace("\xa0", " ").replace("\u00a0", " ").strip().lower()


_col_lookup = {}


def build_col_lookup(df):
    """Build normalized column name to actual column name mapping."""
    global _col_lookup
    _col_lookup = {normalize(c): c for c in df.columns}


def get(row, key):
    """Get a value from the row by key. Handles whitespace and encoding variations."""
    col_target = COL.get(key)
    if col_target is None:
        return ""
    # Try exact match first
    if col_target in row.index:
        val = row[col_target]
        if pd.isna(val):
            return ""
        return str(val).strip()
    # Try normalized match
    norm_target = normalize(col_target)
    actual_col = _col_lookup.get(norm_target)
    if actual_col and actual_col in row.index:
        val = row[actual_col]
        if pd.isna(val):
            return ""
        return str(val).strip()
    return ""


# ── TEXT HELPERS ──────────────────────────────────────────────────────────────

def age_to_words(age_str):
    """Convert a numeric age string to words (e.g. 78 -> seventy-eight)."""
    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    try:
        n = int(float(age_str))
        if n < 20:
            return ones[n]
        elif n < 100:
            t, o = divmod(n, 10)
            return tens[t] + ("-" + ones[o] if o else "")
        else:
            return str(n)
    except (ValueError, TypeError):
        return age_str


def gender_noun(gender_str):
    """Return a man / a woman / a person based on gender string."""
    g = gender_str.lower()
    if "female" in g or "woman" in g:
        return "a woman"
    elif "male" in g or "man" in g:
        return "a man"
    return "a person"


def clean_list(text):
    """Split semicolon or comma separated text into a clean list."""
    if not text:
        return []
    for sep in [";", ","]:
        if sep in text:
            items = [i.strip().rstrip(";,.") for i in text.split(sep) if i.strip()]
            return [i for i in items if i and len(i) > 1]
    return [text.strip()]


def sentences(text):
    """Split free-text into individual sentences, filtering out empties."""
    result = []
    for sentence in text.split("."):
        s = sentence.strip()
        if s and len(s) > 3:
            result.append(s + ".")
    return result


# ── PROFILE BUILDER ───────────────────────────────────────────────────────────

def build_profile(row):
    """Build the full profile text from a questionnaire row."""

    lines = []

    def section(title):
        lines.append("")
        lines.append(f"{title}:")

    def add(text):
        lines.append(f"- {text}")

    # ── Personal Information ──────────────────────────────────────────────────
    section("Personal Information about the user")

    name = get(row, "name")
    age = get(row, "age")
    gender = get(row, "gender")
    city = get(row, "city").rstrip("., ")

    if name:
        add(f"My name is {name}.")
        add(f"I prefer to be called {name}.")

    if age:
        age_clean = age.replace(".0", "").strip()
        add(f"I am {age_clean} years old.")
        age_words = age_to_words(age_clean)
        if age_words and age_words != age_clean:
            add(f"I am {age_words}.")

    if gender:
        add(f"I am {gender.lower()}.")
        add(f"I am {gender_noun(gender)}.")

    if city:
        add(f"I live in {city}, United Kingdom.")

    living = get(row, "living")
    if living:
        add(f"My living situation: {living.lower().rstrip('., ')}.")

    # ── Relationships ─────────────────────────────────────────────────────────
    section("Relationships of the user")

    marital = get(row, "marital")
    if marital:
        add(f"I am {marital.lower()}.")

    family = get(row, "family")
    if family:
        for s in sentences(family):
            add(s)

    important = get(row, "important_people")
    if important:
        for s in sentences(important):
            add(s)

    had_pets = get(row, "had_pets").lower()
    pets_detail = get(row, "pets_detail")
    if "yes" in had_pets:
        if pets_detail:
            for s in sentences(pets_detail):
                add(s)
        else:
            add("I have had pets.")
    elif "no" in had_pets:
        add("I do not have any pets.")

    # ── Education ─────────────────────────────────────────────────────────────
    section("Education Qualifications of the user")

    education = get(row, "education")
    if education:
        add(f"My highest level of education is: {education.lower().rstrip('., ')}.")

    # ── Professional Background ───────────────────────────────────────────────
    section("Professional Background of the user")

    has_work = get(row, "has_work").lower()
    work_detail = get(row, "work_detail")
    life_20s = get(row, "life_20s40s")

    if "yes" in has_work:
        add("I have work experience.")
        if work_detail:
            for s in sentences(work_detail):
                add(s)
    elif "no" in has_work:
        add("I do not have formal work experience.")
        if life_20s:
            for s in sentences(life_20s):
                add(s)

    # ── Preferences ───────────────────────────────────────────────────────────
    section("Preferences of the user")

    typical_day = get(row, "typical_day")
    if typical_day:
        add(f"A typical day for me: {typical_day.strip()}")

    social = get(row, "activities_social")
    if social:
        items = clean_list(social)
        if len(items) > 1:
            add("Activities I enjoy with family and friends include: "
                + ", ".join(i.lower().rstrip(".") for i in items) + ".")
        else:
            add(f"With family and friends: {social.strip()}")

    solo = get(row, "activities_solo")
    if solo:
        items = clean_list(solo)
        if len(items) > 1:
            add("In my own time I enjoy: "
                + ", ".join(i.lower().rstrip(".") for i in items) + ".")
        else:
            add(f"In my own time: {solo.strip()}")

    dislike = get(row, "activities_dislike")
    if dislike:
        items = clean_list(dislike)
        if len(items) > 1:
            add("I dislike or prefer not to: "
                + ", ".join(i.lower().rstrip(".") for i in items) + ".")
        else:
            add(f"I dislike: {dislike.strip()}")

    topics_new = get(row, "topics_new")
    if topics_new:
        items = clean_list(topics_new)
        if len(items) > 1:
            add("When meeting someone new, I like to talk about: "
                + ", ".join(i.lower().rstrip(".") for i in items) + ".")
        else:
            add(f"When meeting someone new, I like to talk about: {topics_new.strip()}.")

    topics_known = get(row, "topics_known")
    if topics_known:
        items = clean_list(topics_known)
        if len(items) > 1:
            add("With people I know, I like to talk about: "
                + ", ".join(i.lower().rstrip(".") for i in items) + ".")
        else:
            add(f"With people I know, I like to talk about: {topics_known.strip()}.")

    # ── Interaction Style ─────────────────────────────────────────────────────
    section("Interaction Style of the user")

    style = get(row, "interaction_style")
    if style:
        add(f"{style.strip().rstrip('.')}.")

    freq = get(row, "interaction_freq")
    if freq:
        add(f"I interact with other people: {freq.lower().strip()}.")

    # ── Environment & Lifestyle ───────────────────────────────────────────────
    section("Environment & Lifestyle of the user")

    tech_freq = get(row, "tech_freq")
    if tech_freq:
        add(f"I use technology {tech_freq.lower().strip()}.")

    devices = get(row, "tech_devices")
    if devices:
        items = clean_list(devices)
        if items:
            cleaned = [i.lower().rstrip(".").strip() for i in items if i.strip()]
            add("The technology devices I use include: " + ", ".join(cleaned) + ".")

    return "\n".join(lines).strip()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def load_file(filepath):
    """Load CSV or Excel file into a DataFrame."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                return pd.read_csv(filepath, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not read the CSV file. Try saving it as UTF-8 CSV from Excel.")
    elif ext in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Please provide a .csv or .xlsx file.")


def main():
    # Load file
    print(f"\nLoading: {INPUT_FILE}")
    try:
        df = load_file(INPUT_FILE)
    except FileNotFoundError:
        print(f"ERROR: File not found: {INPUT_FILE}")
        print("Please check the INPUT_FILE path in the CONFIG section at the top of the script.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading file: {e}")
        sys.exit(1)

    # Build column lookup (handles non-breaking spaces etc.)
    build_col_lookup(df)

    # Validate row number
    if PARTICIPANT_ROW < 1 or PARTICIPANT_ROW > len(df):
        print(f"ERROR: PARTICIPANT_ROW {PARTICIPANT_ROW} is out of range.")
        print(f"The file has {len(df)} data rows (1 to {len(df)}).")
        sys.exit(1)

    # Get the row (0-indexed internally)
    row = df.iloc[PARTICIPANT_ROW - 1]

    print(f"Generating profile for row {PARTICIPANT_ROW}...")

    # Build profile text
    profile_text = build_profile(row)

    # Determine output path
    output_path = OUTPUT_FILE if OUTPUT_FILE else f"profile_P{str(PARTICIPANT_ROW).zfill(2)}.txt"

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write profile to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(profile_text)

    print(f"Profile saved to: {output_path}")
    print()
    print("-" * 60)
    print(profile_text)
    print("-" * 60)


if __name__ == "__main__":
    main()
