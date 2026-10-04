"""
Transform data/raw/medsynth.jsonl into data/processed/{patients,records}.jsonl.

Demo identities are assigned by scripts/demo_identity.py: MedSynth has no
name column and the names embedded in its note prose are heavily reused, so
each patient gets a synthesized unique name drawn from a pool matching the
gender and ethnicity their own note states, and that name is then written back
over the note's original name. See that module's docstring for the full
rationale.

patient_id is assigned once, here, and is treated as stable and opaque
everywhere else in the app - it is never re-derived from full_name.

Usage:
    python scripts/prepare_medsynth.py [--limit N]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_app_config  # noqa: E402
from scripts.demo_identity import assign_name, detect_ethnicity, detect_gender, extract_original_name, rewrite_name # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:

    # Arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=200, help="Max number of demo patients to prepare"
    )
    args = parser.parse_args()

    # Config
    app_config = get_app_config()
    raw_path = app_config.data_dir / "raw" / "medsynth.jsonl"
    if not raw_path.exists():
        raise SystemExit(f"{raw_path} not found - run scripts/download_medsynth.py first.")

    processed_dir = app_config.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set()
    bucket_counts: dict[tuple[str, str], int] = {}
    stats = collections.Counter()
    patients = []
    records = []

    with open(raw_path, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):

            if index >= args.limit:
                break
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)

            # Build patient ID based on row's position in the file
            source_row_id = str(index)
            patient_id = f"patient_{uuid.uuid5(uuid.NAMESPACE_DNS, f'medsynth:{source_row_id}').hex[:12]}"

            # Extract data
            dialogue = str(row["Dialogue"]) if row.get("Dialogue") else ""
            note = str(row[" Note"]) if row.get(" Note") else ""
            icd10 = str(row["ICD10"]) if row.get("ICD10") else ""
            diagnosis = str(row["ICD10_desc"]) if row.get("ICD10_desc") else ""

            # Assign an identity that agrees with what this record says about the patient.
            detection_text = note or dialogue
            gender = detect_gender(detection_text, fallback_index=index)
            ethnicity = detect_ethnicity(detection_text)

            full_name = assign_name(gender, ethnicity, bucket_counts, used_names)

            # Make the record agree with the identity by replacing original name with new generated name
            original_name = extract_original_name(note) or extract_original_name(dialogue)
            if original_name:
                note = rewrite_name(note, original_name, full_name)
                dialogue = rewrite_name(dialogue, original_name, full_name)
                stats["rewritten"] += 1
            else:
                stats["no_original_name"] += 1

            stats[f"gender_{gender}"] += 1
            stats[f"ethnicity_{ethnicity}"] += 1

            patients.append(
                {
                    "patient_id": patient_id,
                    "full_name": full_name,
                    "note": note,
                    "doctor_patient_dialogue": dialogue,
                    "icd10": icd10,
                    "diagnosis": diagnosis,
                    "source_record_id": source_row_id,
                }
            )

            # For each patient, create three separate records.
            # Records are then chunked, embedded, indexed and retrieved by the RAG pipeline.
            for section, text in (
                ("dialogue", dialogue),
                ("note", note),
                ("diagnosis", diagnosis),
            ):
                if text:
                    records.append(
                        {
                            "patient_id": patient_id,
                            "source_record_id": source_row_id,
                            "section": section,
                            "text": text,
                        }
                    )

    _write_jsonl(processed_dir / "patients.jsonl", patients)
    _write_jsonl(processed_dir / "records.jsonl", records)

    print(f"Wrote {len(patients)} patients and {len(records)} records to {processed_dir}")
    print(f"  unique full names   : {len({p['full_name'] for p in patients})}")
    print(f"  gender              : M={stats['gender_M']} F={stats['gender_F']}")
    print("  ethnicity           : " + " ".join(f"{key.removeprefix('ethnicity_')}={count}" 
                                                for key, count in sorted(stats.items()) if key.startswith("ethnicity_")))
    print(f"  names rewritten     : {stats['rewritten']}")
    print(f"  no name in source   : {stats['no_original_name']}")


if __name__ == "__main__":
    main()
