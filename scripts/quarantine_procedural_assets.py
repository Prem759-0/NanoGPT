"""
scripts/quarantine_procedural_assets.py
Stage 2.0.1: Quarantines all procedurally generated images and videos into synthetic development manifests.
Empties production manifests so that genuine counts are truthfully reported as 0.
Chief ML Systems Architect: Prem Gaikwad (India)
"""
import os
import json

def load_records(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def save_records(path, records):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    print("=== STAGE 2.0.1: QUARANTINE PROCEDURAL ASSETS ===")
    
    # 1. Image Quarantine
    prod_image_files = [
        "datasets/manifests/image_train.jsonl",
        "datasets/manifests/image_val.jsonl",
        "datasets/manifests/image_test.jsonl"
    ]
    all_prod_images = []
    for p in prod_image_files:
        records = load_records(p)
        print(f"Loaded {len(records)} records from {p}")
        all_prod_images.extend(records)
        save_records(p, []) # Empty production manifest
        print(f"Emptied {p} -> 0 records")

    synth_images = load_records("datasets/manifests/synthetic_image_dev.jsonl")
    print(f"Existing synthetic images: {len(synth_images)}")
    existing_img_ids = {r["id"] for r in synth_images}

    reclassified_images = 0
    for r in all_prod_images:
        r["is_synthetic"] = True
        r["source_type"] = "procedural_synthetic_fixture"
        r["provenance_type"] = "synthetic_dev_fixture"
        r["source_url"] = "internal://scripts/build_pilot_datasets.py"
        r["notes"] = "Procedurally generated image fixture for pipeline/unit tests only. Not genuine media."
        if r["id"] not in existing_img_ids:
            synth_images.append(r)
            existing_img_ids.add(r["id"])
            reclassified_images += 1

    save_records("datasets/manifests/synthetic_image_dev.jsonl", synth_images)
    print(f"Updated datasets/manifests/synthetic_image_dev.jsonl -> {len(synth_images)} total synthetic images (+{reclassified_images} reclassified)")

    # 2. Video Quarantine
    prod_video_files = [
        "datasets/manifests/video_train.jsonl",
        "datasets/manifests/video_val.jsonl",
        "datasets/manifests/video_test.jsonl"
    ]
    all_prod_videos = []
    for p in prod_video_files:
        records = load_records(p)
        print(f"Loaded {len(records)} records from {p}")
        all_prod_videos.extend(records)
        save_records(p, []) # Empty production manifest
        print(f"Emptied {p} -> 0 records")

    synth_videos = load_records("datasets/manifests/synthetic_video_dev.jsonl")
    print(f"Existing synthetic videos: {len(synth_videos)}")
    existing_vid_ids = {r["id"] for r in synth_videos}

    reclassified_videos = 0
    for r in all_prod_videos:
        r["is_synthetic"] = True
        r["source_type"] = "procedural_synthetic_fixture"
        r["provenance_type"] = "synthetic_dev_fixture"
        r["source_url"] = "internal://scripts/build_pilot_datasets.py"
        r["notes"] = "Procedurally generated video animation fixture for pipeline/unit tests only. Not genuine media."
        if r["id"] not in existing_vid_ids:
            synth_videos.append(r)
            existing_vid_ids.add(r["id"])
            reclassified_videos += 1

    save_records("datasets/manifests/synthetic_video_dev.jsonl", synth_videos)
    print(f"Updated datasets/manifests/synthetic_video_dev.jsonl -> {len(synth_videos)} total synthetic videos (+{reclassified_videos} reclassified)")

    # 3. Chat and Code Quarantine / Dev Labeling
    chat_train = load_records("datasets/manifests/chat_train.jsonl")
    chat_val = load_records("datasets/manifests/chat_val.jsonl")
    all_chat = []
    for r in chat_train + chat_val:
        r["is_dev_fixture"] = True
        r["source_type"] = "dev_example_fixture"
        r["notes"] = "Development example only. Not a production training corpus."
        all_chat.append(r)
    save_records("datasets/manifests/synthetic_chat_dev.jsonl", all_chat)
    save_records("datasets/manifests/chat_train.jsonl", [])
    save_records("datasets/manifests/chat_val.jsonl", [])
    print(f"Quarantined {len(all_chat)} chat records to synthetic_chat_dev.jsonl; emptied production chat manifests.")

    code_train = load_records("datasets/manifests/code_train.jsonl")
    code_val = load_records("datasets/manifests/code_val.jsonl")
    all_code = []
    for r in code_train + code_val:
        r["is_dev_fixture"] = True
        r["source_type"] = "dev_example_fixture"
        r["notes"] = "Development example only. Not a production training corpus."
        all_code.append(r)
    save_records("datasets/manifests/synthetic_code_dev.jsonl", all_code)
    save_records("datasets/manifests/code_train.jsonl", [])
    save_records("datasets/manifests/code_val.jsonl", [])
    print(f"Quarantined {len(all_code)} code records to synthetic_code_dev.jsonl; emptied production code manifests.")

    print("\nQuarantine complete. All production manifests now have 0 records.")
    print("Genuine image count: 0")
    print("Genuine video count: 0")
    print(f"Synthetic dev image count: {len(synth_images)}")
    print(f"Synthetic dev video count: {len(synth_videos)}")

if __name__ == "__main__":
    main()

