#!/usr/bin/env python3
"""Build the full-candidate Hebrew translation overlay for R052 development.

Translations are keyed only by group and source chunk reference.  The builder
never reads expected outcomes, option labels, or gold fields.  Every candidate,
including decisive candidates, uses the same exact-fragment catalog path.

The result remains AI-authored, unreviewed development material.  It is not a
frozen panel, B1 evidence, or benchmark evidence.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from scripts import run_r052_he_mm_dev_diag as diagnostic
    from scripts import score_r052_evidence_retrieval as retrieval
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    import run_r052_he_mm_dev_diag as diagnostic
    import score_r052_evidence_retrieval as retrieval

SCHEMA_VERSION = "r052-hebrew-complete-candidate-translation-overlay-v2"
STATUS = "ai_unreviewed_development_only"

# These translations were authored uniformly from the English candidate text,
# before retrieval scoring, without consulting option correctness or outcomes.
TRANSLATIONS: dict[tuple[str, str], str] = {
    (
        "he_en_01_aar_probable_cause",
        "aar_contributing",
    ): "חשש מנגיעה מוקדמת במסלול כדי להבטיח מרחק עצירה מספיק",
    (
        "he_en_01_aar_probable_cause",
        "aar_egress",
    ): "כל חמשת הנוסעים נפצעו באורח קל בהתרסקות ובמהלך היציאה לאחר מכן דרך חלון תא הטייס.",
    (
        "he_en_01_aar_probable_cause",
        "aar_goaround",
    ): "הנגיעה במסלול והקפיצה היו מונעות את התאונה.",
    ("he_en_01_aar_probable_cause", "aar_issues"): "טכניקות נחיתה, נחיתה עם קפיצה",
    ("he_en_01_aar_probable_cause", "aar_routine"): "שגרתית וללא אירועים חריגים",
    (
        "he_en_01_aar_probable_cause",
        "aar_structure",
    ): "לא היה נזק או בלאי מוקדם במבנה המטוס",
    ("he_en_01_aar_probable_cause", "aar_weather"): "מזג האוויר לא היה גורם בתאונה.",
    ("he_en_02_har_phone_role", "har_barrier"): "מחסום הפרדה יעיל באתר התאונה.",
    ("he_en_02_har_phone_role", "har_call"): "שיחה רגעים לפני התאונה.",
    ("he_en_02_har_phone_role", "har_fatalities"): "חמישה מבוגרים נהרגו",
    ("he_en_02_har_phone_role", "har_novice"): "למעשה, היא הייתה נהגת מתחילה.",
    (
        "he_en_02_har_phone_role",
        "har_speed",
    ): "נסיעה במהירות של 15 עד 20 מייל לשעה מעל מגבלת 55 המייל לשעה",
    ("he_en_02_har_phone_role", "har_unfamiliar"): "נהגה במכונית פחות מ-50 מייל",
    (
        "he_en_03_mar_death_band",
        "mar_astern",
    ): "מנע מאנשי הצוות להיכנס לאסדות ההצלה מן הצד האחורי",
    (
        "he_en_03_mar_death_band",
        "mar_integrity",
    ): "היעדר אטימות פנימית למים אפשר הצפה מתקדמת",
    (
        "he_en_03_mar_death_band",
        "mar_probable",
    ): "הצפה בלתי נשלטת ומתקדמת עקב היעדר אטימות פנימית",
    (
        "he_en_03_mar_death_band",
        "mar_rafts",
    ): "22 אנשי הצוות שהגיעו לאסדות שרדו את הטביעה.",
    (
        "he_en_03_mar_death_band",
        "mar_rescue",
    ): "החילוץ היה מהיר ויעיל וצמצם את אובדן החיים בתאונה.",
    ("he_en_03_mar_death_band", "mar_rudder"): "אובדן פיזי של הגה כיוון",
    (
        "he_en_04_par_fireball_visibility",
        "par_casualties",
    ): "שני אנשים נהרגו ושבעה אנשים נפצעו באורח קל.",
    (
        "he_en_04_par_fireball_visibility",
        "par_comms",
    ): "תקשורת חירום במחוז קלארק, מיסיסיפי.",
    ("he_en_04_par_fireball_visibility", "par_damage"): "3,377,247 דולר",
    ("he_en_04_par_fireball_visibility", "par_houses"): "ארבעה בתים נהרסו",
    (
        "he_en_04_par_fireball_visibility",
        "par_lferw",
    ): "מנגנוני כשל ובטיחות של צינור מרותך בהתנגדות חשמלית בתדר נמוך",
    (
        "he_en_04_par_fireball_visibility",
        "par_release",
    ): "כ-10,253 חביות (430,626 גלונים) של",
    ("he_en_04_par_fireball_visibility", "par_weld"): "כשל של ריתוך שגרם לצינור להיסדק",
    ("he_en_05_rar_training_completion", "rar_damage"): "הנזק עלה על 8.7 מיליון דולר.",
    (
        "he_en_05_rar_training_completion",
        "rar_fatalities",
    ): "שני אנשי הצוות ברכבת הפוגעת נהרגו",
    (
        "he_en_05_rar_training_completion",
        "rar_probable",
    ): "אי-ציות של צוות הרכבת הפוגעת לאות",
    ("he_en_05_rar_training_completion", "rar_ptc"): "לא היה מונע תאונה זו",
    (
        "he_en_05_rar_training_completion",
        "rar_response",
    ): "תגובת החירום לתאונה הייתה מהירה ומתאימה.",
    (
        "he_en_05_rar_training_completion",
        "rar_schedules",
    ): "לוחות עבודה לא סדירים תרמו לעייפותם",
    ("he_en_05_rar_training_completion", "rar_survival"): "הצוות לא היה יכול לשרוד.",
    (
        "he_en_06_gao_consultation_delay_rate",
        "c01-consultation-tracker-quality",
    ): "לא מלאים ולא מדויקים עבור 769 (75 אחוזים) מתוך 1,026",
    (
        "he_en_06_gao_consultation_delay_rate",
        "c02-hcp-no-local-tracker",
    ): "לא הייתה לו מערכת מעקב משלו",
    (
        "he_en_06_gao_consultation_delay_rate",
        "c03-hcp-missing-contact-date",
    ): "לא ציין את התאריך שבו פונה יצר קשר לראשונה עם משרד קרלסבד",
    (
        "he_en_06_gao_consultation_delay_rate",
        "c04-remediation-not-assessed",
    ): "לא העריכו את היעילות",
    (
        "he_en_06_gao_consultation_delay_rate",
        "c06-delay-cause-undetermined",
    ): "מן התיעוד לא יכולנו לקבוע זאת.",
    (
        "he_en_06_gao_consultation_delay_rate",
        "c07-complaint-process-gap",
    ): "אין נהלים כתובים",
    (
        "he_en_06_gao_consultation_delay_rate",
        "c08-staffing-recommendation",
    ): "לזהות ולהעריך אפשרויות לשיפור יכולתו של משרד קרלסבד",
    (
        "he_en_07_gao_lifecycle_tracking",
        "c01-progress-overall",
    ): "הפעולות הפחיתו את שיעורי הפערים במלאי ושיפרו",
    (
        "he_en_07_gao_lifecycle_tracking",
        "c02-cdc-partner-agreements",
    ): "CDC טרם השלים הסכמים עם שותפיו",
    (
        "he_en_07_gao_lifecycle_tracking",
        "c03-temperature-exposure",
    ): "הצג רשם תשע מעלות",
    (
        "he_en_07_gao_lifecycle_tracking",
        "c04-oep-requirements-list",
    ): "להחזיק רשימת דרישות מלאי מעודכנת המשקפת שינויים אחרונים",
    (
        "he_en_07_gao_lifecycle_tracking",
        "c06-cbirf-discrepancy-rate",
    ): "שיעור הפערים ירד מ-26 אחוזים לכ-10 אחוזים.",
    (
        "he_en_07_gao_lifecycle_tracking",
        "c07-expired-cbirf-items",
    ): "מצאנו 161 פריטים שפג תוקפם",
    (
        "he_en_07_gao_lifecycle_tracking",
        "c08-written-agreements-recommendation",
    ): "לחתום על הסכמים כתובים בהקדם האפשרי",
    (
        "he_en_07_gao_lifecycle_tracking",
        "c09-temperature-remedy",
    ): "להחליף את הפריטים שנקבע כי אינם שמישים עוד",
    (
        "he_en_08_gao_omb_action_count",
        "c01-three-failure-themes",
    ): "שלושה נושאים חוזרים",
    (
        "he_en_08_gao_omb_action_count",
        "c02-centers-of-excellence",
    ): "הקים מרכזי מצוינות",
    (
        "he_en_08_gao_omb_action_count",
        "c03-over-forty-reports",
    ): "סקירה של יותר מ-40 דוחות קודמים של GAO ושל מפקחים כלליים",
    (
        "he_en_08_gao_omb_action_count",
        "c04-army-manual-workarounds",
    ): "פתרונות עקיפה ידניים כדי לצמצם שיבושים",
    (
        "he_en_08_gao_omb_action_count",
        "c05-omb-initiatives-early",
    ): "שלבים מוקדמים של היישום",
    (
        "he_en_08_gao_omb_action_count",
        "c06-four-underdeveloped-concepts",
    ): "מושגים שטרם פותחו במלואם",
    (
        "he_en_08_gao_omb_action_count",
        "c07-estimated-consolidation-savings",
    ): "צפוי חיסכון של יותר מ-5 מיליארד דולר לאורך תקופה של 10 שנים",
    (
        "he_en_08_gao_omb_action_count",
        "c09-migration-path",
    ): "קביעת נתיב מעבר או לוח זמנים",
    (
        "he_en_09_gao_positive_assurance",
        "c01-information-quality-gap",
    ): "להפיק באופן שגרתי מידע פיננסי אמין, שימושי ועדכני.",
    (
        "he_en_09_gao_positive_assurance",
        "c03-fy2005-noncompliance-count",
    ): "בשנת הכספים 2005, 18 מתוך 24 הסוכנויות הכפופות לחוק CFO",
    (
        "he_en_09_gao_positive_assurance",
        "c05-six-primary-problems",
    ): "אותן שש בעיות עיקריות",
    (
        "he_en_09_gao_positive_assurance",
        "c06-shared-service-documentation-gap",
    ): "הדרישות שעליהם לספק לא תועדו במידה מספקת או באופן יעיל",
    (
        "he_en_10_gao_search_method",
        "c01-voluntary-and-mandatory-tools",
    ): "הממשלה משתמשת הן בשותפויות וולונטריות",
    (
        "he_en_10_gao_search_method",
        "c03-total-requirements",
    ): "זוהו 34 חוקים, תקנות ותקנים מחייבים פדרליים",
    (
        "he_en_10_gao_search_method",
        "c04-requirement-type-counts",
    ): "אחד הוא חוק, 25 הם תקנות ו-8 הם תקנים מחייבים",
    (
        "he_en_10_gao_search_method",
        "c05-sector-coverage-count",
    ): "הדרישות נוגעות ל-10 מתוך 18 מגזרי התשתית החיונית",
    (
        "he_en_10_gao_search_method",
        "c06-eight-sectors-without-identified-requirements",
    ): "שמונה מגזרים לא זיהו דרישות",
    (
        "he_en_10_gao_search_method",
        "c07-enforcement-coverage",
    ): "לכל אחת מ-34 הדרישות יש לפחות מנגנון אכיפה אחד",
    (
        "he_en_10_gao_search_method",
        "c08-drinking-water-law",
    ): "חוק אחד החל על מגזר מי השתייה ומערכות הטיפול במים",
    (
        "he_en_10_gao_search_method",
        "c09-regulations-by-sector",
    ): "יש 25 תקנות, והן חלות ב-8 מגזרים",
    (
        "he_en_10_gao_search_method",
        "c10-mandatory-standard-sectors",
    ): "תקנים מחייבים חלים במגזרי הסכרים והאנרגיה",
    (
        "he_en_01_aar_probable_cause",
        "aar_primary_cause",
    ): "שליטת-היתר של הקברניט במטוס במהלך הנחיתה",
    (
        "he_en_02_har_phone_role",
        "har_phone_contributed",
    ): "הסחת הדעת עקב שיחת הטלפון האלחוטי עם חברתה תרמה לכך שהיא",
    (
        "he_en_02_har_phone_role",
        "har_uncertainty",
    ): "ההיקף המדויק של תפקיד הסחת הדעת עקב שימוש בטלפון אלחוטי.",
    (
        "he_en_03_mar_death_band",
        "mar_fatalities",
    ): "חמישה מתוך 47 האנשים שהיו על הסיפון מתו",
    ("he_en_03_mar_death_band", "mar_survival"): "20 שרדו ו-5 לא שרדו",
    (
        "he_en_04_par_fireball_visibility",
        "par_ignition",
    ): "כדור אש שנשמע ונראה ממרחק של מיילים.",
    (
        "he_en_05_rar_training_completion",
        "rar_training",
    ): "לא הכרטיסן ולא נהג הקטר של רכבת הפחם השתתפו בהדרכה זו.",
    (
        "he_en_06_gao_consultation_delay_rate",
        "c05-formal-consultation-delay-rate",
    ): "כ-128 (35 אחוזים) מתוך 361",
    (
        "he_en_07_gao_lifecycle_tracking",
        "c05-lifecycle-tracking-gap",
    ): "המערכות עדיין אינן מתעדות ועוקבות אחר כל הפעילות",
    (
        "he_en_08_gao_omb_action_count",
        "c08-eighteen-actions",
    ): "18 פעולות. הדבר יחייב מתן עדיפות גבוהה לשילוב מלא",
    (
        "he_en_09_gao_positive_assurance",
        "c09-positive-assurance-recommendation-conclusion",
    ): "אנו מאשרים מחדש את המלצתנו מדוחות קודמים",
    (
        "he_en_09_gao_positive_assurance",
        "he09_omb_pma_initiatives",
    ): "OMB ממשיך לקדם יוזמות התומכות באג׳נדת הניהול של הנשיא כדי לשפר את הניהול הפיננסי ולספק מידע מוכוון-תוצאות בממשלה הפדרלית.",
    (
        "he_en_10_gao_search_method",
        "c02-methodology-search-limit",
    ): "לא ערכנו חיפוש עצמאי",
}


def _load_inputs(
    he_base: Path,
    he_overlay: Path,
    ntsb_candidates: Path,
    gao_candidates: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, str]], dict[str, str]]:
    base, base_sha = diagnostic._load_json(he_base, "Hebrew v2 base")
    overlay, overlay_sha = diagnostic._load_json(he_overlay, "Hebrew v4 overlay")
    effective = diagnostic.materialize_hebrew_v4(base, base_sha, overlay)
    candidates, candidate_hashes = diagnostic._candidate_maps(
        ntsb_candidates, gao_candidates
    )
    declared = effective.get("input_bindings", {})
    if candidate_hashes["ntsb_candidate_sha256"] != declared.get(
        "ntsb_candidate_dossier_sha256"
    ) or candidate_hashes["gao_candidate_sha256"] != declared.get(
        "gao_candidate_dossier_sha256"
    ):
        raise ValueError("Hebrew draft does not bind supplied candidate dossiers")
    bindings = {
        "hebrew_v2_raw_sha256": base_sha,
        "hebrew_v4_raw_sha256": overlay_sha,
        "hebrew_v4_effective_canonical_sha256": retrieval.sha256_bytes(
            diagnostic._canonical_bytes(effective)
        ),
        "ntsb_candidate_sha256": candidate_hashes["ntsb_candidate_sha256"],
        "gao_candidate_sha256": candidate_hashes["gao_candidate_sha256"],
    }
    return effective, candidates, bindings


def build_overlay(
    *,
    he_base: Path,
    he_overlay: Path,
    ntsb_candidates: Path,
    gao_candidates: Path,
    translation_catalog: Mapping[tuple[str, str], str] = TRANSLATIONS,
) -> dict[str, Any]:
    effective, candidates, bindings = _load_inputs(
        he_base, he_overlay, ntsb_candidates, gao_candidates
    )
    groups: dict[str, list[dict[str, str]]] = {}
    consumed_catalog: set[tuple[str, str]] = set()
    catalog_count = 0
    for group in sorted(effective["groups"], key=lambda item: item["group_id"]):
        group_id = diagnostic._text(group.get("group_id"), "group id")
        source_id = diagnostic._text(group.get("source_id"), "source id")
        corpus = dict(candidates.get(source_id, {}))
        for added in group.get("added_genuine_distractor_chunks", []):
            corpus[added["chunk_ref"]] = diagnostic._text(
                added.get("en_complete_proposition"), "added English proposition"
            )
        refs = [
            ref
            for evidence in group.get("decisive_evidence", [])
            for ref in evidence.get("candidate_chunk_refs", [])
        ] + list(group.get("known_distractor_chunk_refs", []))
        if (
            not refs
            or len(refs) != len(set(refs))
            or set(refs) != set(corpus) & set(refs)
        ):
            raise ValueError(f"{group_id}: incomplete or duplicate candidate coverage")
        rows = []
        for ref in sorted(refs):
            english = diagnostic._text(corpus[ref], f"{group_id}/{ref} English")
            key = (group_id, ref)
            if key not in translation_catalog:
                raise ValueError(f"missing catalog translation for {group_id}/{ref}")
            hebrew = diagnostic._text(
                translation_catalog[key], f"{group_id}/{ref} Hebrew"
            )
            consumed_catalog.add(key)
            catalog_count += 1
            rows.append(
                {
                    "chunk_ref": ref,
                    "en_text_sha256": retrieval.sha256_bytes(english.encode("utf-8")),
                    "he_text": hebrew,
                    "he_text_sha256": retrieval.sha256_bytes(hebrew.encode("utf-8")),
                }
            )
        groups[group_id] = rows
    if set(translation_catalog) != consumed_catalog:
        unused = sorted(set(translation_catalog) - consumed_catalog)
        raise ValueError(f"translation catalog contains stale entries: {unused}")
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "human_reviewed": False,
        "eligible_for_b1_or_benchmark_claims": False,
        "translation_policy": (
            "uniform English-source translation keyed only by group_id/chunk_ref; "
            "no options, gold labels, expected outcomes, or retrieval scores consulted"
        ),
        "source_bindings": dict(sorted(bindings.items())),
        "counts": {
            "groups": len(groups),
            "rows": sum(map(len, groups.values())),
            "catalog_translations": catalog_count,
        },
        "groups": groups,
    }
    return {**body, "overlay_sha256": retrieval.sha256_json(body)}


def _safe_output_path(path: Path) -> Path:
    absolute = path.absolute()
    if path.is_symlink():
        raise FileExistsError(f"refusing symlink output: {path}")
    try:
        resolved_parent = absolute.parent.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"output directory does not exist: {absolute.parent}"
        ) from exc
    if resolved_parent != absolute.parent or not resolved_parent.is_dir():
        raise ValueError("output ancestors must be existing non-symlink directories")
    return absolute


def _exclusive_publish(path: Path, data: bytes) -> None:
    path = _safe_output_path(path)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing output: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def build_and_publish(*, output: Path, **kwargs: Any) -> dict[str, Any]:
    output = _safe_output_path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    overlay = build_overlay(**kwargs)
    _exclusive_publish(output, retrieval.canonical_json(overlay))
    return overlay


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--he-base", type=Path, required=True)
    parser.add_argument("--he-overlay", type=Path, required=True)
    parser.add_argument("--ntsb-candidates", type=Path, required=True)
    parser.add_argument("--gao-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    build_and_publish(
        he_base=args.he_base,
        he_overlay=args.he_overlay,
        ntsb_candidates=args.ntsb_candidates,
        gao_candidates=args.gao_candidates,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
