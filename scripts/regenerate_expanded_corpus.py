"""
Stage 2.1.5.1 Corpus Repair Script
Regenerates expanded manifests with correct provenance classification:
- DETERMINISTIC_CURRICULUM_TEMPLATE (not TEACHER_DISTILLATION_OFFLINE_QUALIFIED)
- Corrected partition: 400 train / 50 val / 50 test = 500 total
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from teacher.factory.curator import TeacherCorpusCurator

curator = TeacherCorpusCurator()
result = curator.expand_and_partition_corpus(
    preserve_existing=True,
    overwrite_production=False,
    teacher_installed=False
)

print("=== STAGE 2.1.5.1 CORPUS REPAIR RESULT ===")
print(f"Train records:  {result['train_records']}")
print(f"Val records:    {result['val_records']}")
print(f"Test records:   {result['test_records']}")
print(f"Total records:  {result['total_records']}")
print(f"Total tokens:   {result['total_tokens']:,}")
print(f"Readiness:      {result['readiness_status']}")
print(f"Blockers:       {len(result['blockers'])}")
for b in result["blockers"]:
    print(f"  - {b}")
print(f"Provenance:     {result.get('provenance_note', 'N/A')}")
print(f"Train SHA-256:  {result['train_sha256']}")
print(f"Val SHA-256:    {result['val_sha256']}")
print(f"Test SHA-256:   {result['test_sha256']}")

