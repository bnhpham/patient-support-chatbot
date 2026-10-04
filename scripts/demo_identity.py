"""
Demo patient identity assignment for MedSynth records.

MedSynth has no patient-name column. Names appear inside the Note prose
and are heavily reused - across the full dataset there are only ~1,700 
distinct names for 10,240 rows, with "John Smith" alone on 992 of them.
Some descriptions contradict each other, such as ("John Smith, a 70-year-old 
African American male" vs. "a 52-year-old Caucasian male named John Smith")
They therefore cannot serve as the login identifier, which the API requires 
to map to exactly one patient_id.

So each demo patient is given a synthesized unique name instead. Two things
make that name consistent with the record it is attached to:

1. The name is drawn from a pool keyed by the gender and ethnicity stated in
   the patient's own note, so a note describing a woman never gets a male
   name.
2. Whatever name the note originally used is rewritten to the assigned name
   throughout both the note and the dialogue, so the record never contradicts
   the identity the patient logged in with.

A side benefit for the guardrails experiment: every patient now has a known
name embedded in retrievable text, which is a convenient PII target when
testing output-side leakage controls later.
"""

from __future__ import annotations

import re

# Name pools keyed by (gender, ethnicity). 20 first x 20 last names give 400
# collision-free combinations per bucket.
#
# Largest bucket actually measured over the real dataset, by prepare_medsynth
# --limit:  31 at 200 rows (the current default), 146 at 1000, and 1563 across
# all 10,240 rows. The pools therefore cover the default with room to spare.
# Beyond ~400 in a single bucket, assign_name() falls back to a numeric suffix
# ("Margaret Whitfield 2"); names stay unique - verified by simulating the
# 1563-row worst case - but read less naturally, so widen the pools if you ever
# prepare the whole dataset.
#
# Ethnicity buckets mirror the four groups MedSynth actually states, plus a
# fallback for notes that state none.
#
# For simplicity, non-binary identities and gender-neutral pronouns 
# (e.g. they/them) are not modeled as they are also not included in MedSynth.
_MALE_FIRST = {
    "caucasian": [
        "James", "Robert", "William", "Charles", "Thomas", "Daniel", "Matthew", "Andrew",
        "Christopher", "Nicholas", "Benjamin", "Samuel", "Nathan", "Patrick", "Gregory",
        "Timothy", "Douglas", "Frederick", "Lawrence", "Vincent",
    ],
    "hispanic": [
        "Carlos", "Miguel", "Javier", "Alejandro", "Diego", "Rafael", "Eduardo", "Fernando",
        "Ricardo", "Antonio", "Manuel", "Sergio", "Pablo", "Andres", "Emilio",
        "Hector", "Ramon", "Salvador", "Ignacio", "Mateo",
    ],
    "african_american": [
        "Marcus", "Andre", "Terrence", "Darnell", "Jamal", "Tyrone", "Malik", "Xavier",
        "Reginald", "Maurice", "Cedric", "Deandre", "Rashad", "Lamont", "Jerome",
        "Curtis", "Darius", "Isaiah", "Elijah", "Devon",
    ],
    "asian": [
        "Kenji", "Hiroshi", "Wei", "Jian", "Takeshi", "Minho", "Jun", "Haruto",
        "Ravi", "Arjun", "Sanjay", "Vikram", "Chen", "Yong", "Daisuke",
        "Satoshi", "Kiran", "Anil", "Rohan", "Tao",
    ],
}
_MALE_LAST = {
    "caucasian": [
        "Whitfield", "Ashford", "Prescott", "Kingsley", "Hartley", "Merritt", "Caldwell",
        "Ellsworth", "Fairbanks", "Thornton", "Grayson", "Winslow", "Bradshaw", "Kensington",
        "Alderman", "Pemberton", "Radcliffe", "Sinclair", "Weatherby", "Yardley",
    ],
    "hispanic": [
        "Ramirez", "Castillo", "Delgado", "Fuentes", "Herrera", "Ibarra", "Jimenez",
        "Montoya", "Navarro", "Ocampo", "Pacheco", "Quintero", "Rosales", "Salazar",
        "Trevino", "Urbina", "Valdez", "Zamora", "Aguilar", "Bermudez",
    ],
    "african_american": [
        "Abernathy", "Bankston", "Coleridge", "Dubois", "Ellington", "Freeman", "Gantt",
        "Hollis", "Ivory", "Jefferson", "Kincaid", "Lattimore", "Mosley", "Northcutt",
        "Overton", "Prentice", "Rutherford", "Stanfield", "Thurgood", "Waverly",
    ],
    "asian": [
        "Tanaka", "Nakamura", "Fujimoto", "Yamashita", "Okabe", "Hayashi", "Morita",
        "Sakamoto", "Ishikawa", "Kobayashi", "Chandra", "Deshpande", "Iyer", "Kapoor",
        "Malhotra", "Rajan", "Venkatesh", "Zhao", "Xiong", "Bai",
    ],
}
_FEMALE_FIRST = {
    "caucasian": [
        "Margaret", "Eleanor", "Charlotte", "Vivian", "Rosalind", "Beatrice", "Harriet",
        "Genevieve", "Clarissa", "Adelaide", "Winifred", "Cordelia", "Josephine", "Matilda",
        "Prudence", "Evelyn", "Gwendolyn", "Theodora", "Meredith", "Felicity",
    ],
    "hispanic": [
        "Lucia", "Valentina", "Esperanza", "Guadalupe", "Mariana", "Consuelo", "Dolores",
        "Ximena", "Alejandra", "Beatriz", "Catalina", "Rosario", "Marisol", "Pilar",
        "Isabela", "Renata", "Soledad", "Teresa", "Ynez", "Camila",
    ],
    "african_american": [
        "Latoya", "Shanice", "Tamika", "Imani", "Aaliyah", "Ebony", "Jasmine", "Keisha",
        "Nia", "Octavia", "Precious", "Raven", "Shantel", "Tanisha", "Zora",
        "Danielle", "Monique", "Yolanda", "Chantelle", "Brianna",
    ],
    "asian": [
        "Yuki", "Sakura", "Mei", "Aiko", "Hana", "Jia", "Xiu", "Priya",
        "Anjali", "Lakshmi", "Nadia", "Reiko", "Sunhee", "Thuy", "Wen",
        "Ayaka", "Divya", "Kaori", "Meera", "Ling",
    ],
}
_FEMALE_LAST = _MALE_LAST  # surnames are not gendered in these groups

_DEFAULT_ETHNICITY = "default"
_ETHNICITY_KEYS = ("caucasian", "hispanic", "african_american", "asian")


# Build name pools keyed by (gender, ethnicity)
def _build_pools() -> dict[tuple[str, str], tuple[list[str], list[str]]]:

    pools: dict[tuple[str, str], tuple[list[str], list[str]]] = {}

    for ethnicity in _ETHNICITY_KEYS:
        pools[("M", ethnicity)] = (_MALE_FIRST[ethnicity], _MALE_LAST[ethnicity])
        pools[("F", ethnicity)] = (_FEMALE_FIRST[ethnicity], _FEMALE_LAST[ethnicity])

    # Notes that if no ethnicity is stated, fall back to the caucasian pool (most common group in the dataset)
    pools[("M", _DEFAULT_ETHNICITY)] = pools[("M", "caucasian")]
    pools[("F", _DEFAULT_ETHNICITY)] = pools[("F", "caucasian")]

    return pools


GENDER_ETHNICITY_POOLS = _build_pools()

# Regex to identify a patient's gender (non-binary identities are outside the scope of this prototype)
_MALE_TERMS = re.compile(r"\b(male|man|gentleman|boy|Mr\.)\b", re.IGNORECASE)
_FEMALE_TERMS = re.compile(r"\b(female|woman|lady|girl|Mrs\.|Ms\.)\b", re.IGNORECASE)
_MALE_PRONOUNS = re.compile(r"\b(he|his|him)\b", re.IGNORECASE)
_FEMALE_PRONOUNS = re.compile(r"\b(she|her|hers)\b", re.IGNORECASE)

# Regex to identify a patient's ethnicity
_ETHNICITY_PATTERNS = (
    ("hispanic", re.compile(r"\b(hispanic|latino|latina)\b", re.IGNORECASE)),
    ("african_american", re.compile(r"\b(african[- ]american|black)\b", re.IGNORECASE)),
    ("asian", re.compile(r"\b(asian|chinese|japanese|korean|indian|vietnamese|thai)\b", re.IGNORECASE)),
    ("caucasian", re.compile(r"\b(caucasian|white)\b", re.IGNORECASE)),
)

# The phrasings MedSynth actually uses to introduce the patient by name
_NAME_PATTERNS = (
    re.compile(r"named\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)"),
    re.compile(r"\b(?:Mr|Mrs|Ms|Miss)\.?\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)"),
    re.compile(r"Patient(?:\s+Name)?[:,]\s*([A-Z][a-z]+)\s+([A-Z][a-z]+)"),
    re.compile(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\s+is\s+a\s+\d{1,3}-year-old"),
)


def detect_gender(note: str, fallback_index: int = 0) -> str:
    """
    Return "M" or "F" for the patient described in `note`.

    Explicit terms are counted first. If those tie, personal pronouns break it. 
    If there is no signal at all (~2% of rows), the row alternates deterministically by index,
    so repeated runs stay reproducible.
    """
    male_hits = len(_MALE_TERMS.findall(note))
    female_hits = len(_FEMALE_TERMS.findall(note))
    if male_hits != female_hits:
        return "M" if male_hits > female_hits else "F"

    male_hits = len(_MALE_PRONOUNS.findall(note))
    female_hits = len(_FEMALE_PRONOUNS.findall(note))
    if male_hits != female_hits:
        return "M" if male_hits > female_hits else "F"

    return "M" if fallback_index % 2 == 0 else "F"


def detect_ethnicity(note: str) -> str:
    """
    Return one of the pool ethnicity keys, or "default" if none is stated.
    """
    for key, pattern in _ETHNICITY_PATTERNS:
        if pattern.search(note):
            return key
    return _DEFAULT_ETHNICITY


def extract_original_name(note: str) -> tuple[str, str] | None:
    """
    Return the (first, last) name the note uses, if it states one.
    """
    for pattern in _NAME_PATTERNS:
        match = pattern.search(note)
        if match:
            return match.group(1), match.group(2)
        
    return None


def assign_name(gender: str, ethnicity: str, bucket_counts: dict[tuple[str, str], int], used_names: set[str]) -> str:
    """
    Pick the next unused name from the (gender, ethnicity) pool.

    Names are walked sequentially rather than sampled randomly, so a bucket yields 400
    distinct names before any repeat is even possible. `used_names` with a
    numeric suffix remains as a hard guarantee beyond that point.
    """

    # Get name pool for given gender and ethnicity
    key = (gender, ethnicity)
    if key not in GENDER_ETHNICITY_POOLS:
        key = (gender, _DEFAULT_ETHNICITY)
    first_names, last_names = GENDER_ETHNICITY_POOLS[key]

    index = bucket_counts.get(key, 0)
    bucket_counts[key] = index + 1

    # Build new patient name
    first = first_names[index % len(first_names)]
    last = last_names[(index // len(first_names)) % len(last_names)]
    base_name = f"{first} {last}"

    # Add numeric suffix if name has already been assigned (e.g. "James Whitfield" vs "James Whitfield 2")
    name = base_name
    suffix = 2
    while name in used_names:
        name = f"{base_name} {suffix}"
        suffix += 1
    used_names.add(name)

    return name


def rewrite_name(text: str, original: tuple[str, str], new_full_name: str) -> str:
    """
    Replace every reference to `original` in `text` with `new_full_name`.

    Notes and dialogues refer to the patient by full name, by first name
    alone, and by last name alone (the dialogue especially - "Good to see
    you, Mr. Smith"). The full name must be substituted first, otherwise
    replacing the first name would corrupt the full-name occurrences before
    they are matched.

    Matching is case-sensitive and word-bounded so lowercase clinical
    vocabulary is untouched - a patient surnamed "Brown" does not turn
    "brown discharge" into a name.
    """
    if not text:
        return text

    old_first, old_last = original
    new_first, new_last = new_full_name.split(" ", 1)

    text = re.sub(rf"\b{re.escape(old_first)}\s+{re.escape(old_last)}\b", new_full_name, text)
    text = re.sub(rf"\b{re.escape(old_first)}\b", new_first, text)
    text = re.sub(rf"\b{re.escape(old_last)}\b", new_last, text)
    
    return text
