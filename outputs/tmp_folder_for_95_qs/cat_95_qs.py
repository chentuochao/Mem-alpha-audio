import json
from pathlib import Path

# 按你的实际版本选择。
# anonymized 版本：
# qa_files = [
#     "QA_designs/bazinga/TBBT/S1/anony_speaker_attributed_QA.jsonl",      # 39
#     "QA_designs/bazinga/TBBT/S1/anony_long_context_QA.jsonl",            # 36
#     "QA_designs/bazinga/TBBT/s1_arc/speaker_atributted_qa_anonymized.jsonl", # 10
#     "QA_designs/bazinga/TBBT/s1_arc/long_context_QA_anonymized.jsonl",       # 10
# ]

qa_files = [
    "/gscratch/intelligentsystems/wencheng/Mem-alpha-audio/QA_designs/bazinga/TBBT/S1/non_anony_speaker_attributed_QA.jsonl",
    "/gscratch/intelligentsystems/wencheng/Mem-alpha-audio/QA_designs/bazinga/TBBT/S1/non_anony_long_context_QA.jsonl",
    "/gscratch/intelligentsystems/wencheng/Mem-alpha-audio/QA_designs/bazinga/TBBT/s1_arc/speaker_atributted_qa_deanonymized.jsonl",
    "/gscratch/intelligentsystems/wencheng/Mem-alpha-audio/QA_designs/bazinga/TBBT/s1_arc/long_context_QA_unmasked.jsonl",
]
out_path = Path("/gscratch/intelligentsystems/wencheng/Mem-alpha-audio/outputs/tmp_folder_for_95_qs/merged_95.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)

merged = []
seen = set()

for f in qa_files:
    with open(f, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            # Some long-context files are wrapped as {"qa": {...}}
            qa = obj["qa"] if "qa" in obj else obj

            required = ["question", "answer", "options", "gt_source"]
            missing = [k for k in required if k not in qa]
            if missing:
                raise ValueError(f"{f}:{line_no} missing fields: {missing}")

            # Basic option QA check
            if not isinstance(qa["options"], dict):
                raise ValueError(f"{f}:{line_no} options is not a dict")

            if "A" not in qa["options"] or "B" not in qa["options"]:
                raise ValueError(f"{f}:{line_no} options must contain A and B")

            answer = str(qa["answer"])
            if not (answer.startswith("A.") or answer.startswith("B.")):
                raise ValueError(f"{f}:{line_no} answer should start with A. or B.: {answer}")

            # De-duplicate by question text + answer
            key = (qa["question"], qa["answer"])
            if key in seen:
                print(f"[WARN] duplicate skipped: {f}:{line_no} {qa['question'][:80]}")
                continue
            seen.add(key)

            # Preserve useful provenance
            qa["_source_file"] = f
            qa["_source_line"] = line_no

            merged.append(qa)

with open(out_path, "w", encoding="utf-8") as fout:
    for qa in merged:
        fout.write(json.dumps(qa, ensure_ascii=False) + "\n")

print(f"Saved {len(merged)} QA items to {out_path}")
if len(merged) != 95:
    raise SystemExit(f"[ERROR] Expected 95 QA items, got {len(merged)}")