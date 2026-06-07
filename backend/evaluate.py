import os
import sys
from collections import defaultdict

import pandas as pd
import requests

API_URL = "http://localhost:8000/search"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(BASE_DIR, "data_test")
DATASET_DIR = os.path.join(BASE_DIR, "dataset_aug")
OUTPUT_CSV = os.path.join(BASE_DIR, "evaluation_results.csv")
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def collect_valid_labels():
    valid = set()
    for group in sorted(os.listdir(DATASET_DIR)):
        gp = os.path.join(DATASET_DIR, group)
        if not os.path.isdir(gp):
            continue
        for label in sorted(os.listdir(gp)):
            if os.path.isdir(os.path.join(gp, label)):
                valid.add((group, label))
    return valid


def collect_test_images(valid_labels):
    images, skipped = [], []
    for group in sorted(os.listdir(TEST_DIR)):
        gp = os.path.join(TEST_DIR, group)
        if not os.path.isdir(gp):
            continue
        for label in sorted(os.listdir(gp)):
            lp = os.path.join(gp, label)
            if not os.path.isdir(lp):
                continue
            if (group, label) not in valid_labels:
                skipped.append(f"{group}/{label}")
                continue
            for f in sorted(os.listdir(lp)):
                if f.lower().endswith(IMAGE_EXTS):
                    images.append((group, label, f, os.path.join(lp, f)))
    return images, skipped


def main():
    try:
        requests.get("http://localhost:8000/", timeout=5)
    except Exception:
        print("Server chưa chạy. Mở terminal khác: python backend/main.py")
        sys.exit(1)

    valid_labels = collect_valid_labels()
    test_images, skipped = collect_test_images(valid_labels)
    total = len(test_images)

    print(f"Biển hợp lệ: {len(valid_labels)}, Ảnh test: {total}")
    if skipped:
        print(f"Bỏ qua {len(skipped)} thư mục: {', '.join(skipped[:10])}")
    print()

    results = []
    group_stats = defaultdict(lambda: {"c1": 0, "c2": 0, "c3": 0, "total": 0, "errors": []})

    for i, (group, true_label, img_file, img_path) in enumerate(test_images, 1):
        try:
            with open(img_path, "rb") as f:
                resp = requests.post(API_URL, files={"file": (img_file, f, "image/png")}, params={"top_k": 3}, timeout=30)

            if resp.status_code != 200:
                results.append({"group": group, "true": true_label, "top1_label": f"HTTP_{resp.status_code}",
                                "top2_label": "", "top3_label": "", "top1": False, "top2": False, "top3": False, "score": 0})
                group_stats[group]["total"] += 1
                print(f"[{i}/{total}] FAIL {true_label}/{img_file}: HTTP {resp.status_code}")
                continue

            top = resp.json().get("results", [])
            labels = [r.get("label", "") for r in top[:3]]
            while len(labels) < 3:
                labels.append("")

            ok1 = true_label == labels[0]
            ok2 = true_label in labels[:2]
            ok3 = true_label in labels[:3]
            score = top[0].get("score", 0) if top else 0

            results.append({"group": group, "true": true_label, "top1_label": labels[0],
                            "top2_label": labels[1], "top3_label": labels[2],
                            "top1": ok1, "top2": ok2, "top3": ok3, "score": score})

            gs = group_stats[group]
            gs["total"] += 1
            if ok1: gs["c1"] += 1
            else: gs["errors"].append(f"{true_label} -> {labels[0]}")
            if ok2: gs["c2"] += 1
            if ok3: gs["c3"] += 1

            print(f"[{i}/{total}] {'OK' if ok1 else 'NO'} {true_label} -> {labels[0]} ({score:.0%})")

        except Exception as e:
            print(f"[{i}/{total}] ERROR {true_label}/{img_file}: {e}")

    if not results:
        print("Không có kết quả.")
        return

    n = len(results)
    c1 = sum(1 for r in results if r["top1"])
    c2 = sum(1 for r in results if r["top2"])
    c3 = sum(1 for r in results if r["top3"])

    print(f"\n{'='*50}")
    print(f"Top-1: {c1}/{n} = {c1/n:.1%}")
    print(f"Top-2: {c2}/{n} = {c2/n:.1%}")
    print(f"Top-3: {c3}/{n} = {c3/n:.1%}")

    print("\nTheo nhóm:")
    for g, s in sorted(group_stats.items()):
        if s["total"] == 0:
            continue
        t = s["total"]
        print(f"  {g}: Top1={s['c1']/t:.0%} Top2={s['c2']/t:.0%} Top3={s['c3']/t:.0%} ({s['c1']}/{t})")
        for e in s["errors"][:5]:
            print(f"    - {e}")

    df = pd.DataFrame(results)
    summary = pd.DataFrame([
        {"group": "TOTAL", "true": f"{n} ảnh",
         "top1_label": f"Top-1: {c1/n:.1%}", "top2_label": f"Top-2: {c2/n:.1%}",
         "top3_label": f"Top-3: {c3/n:.1%}", "top1": c1, "top2": c2, "top3": c3, "score": ""}
    ])
    for g, s in sorted(group_stats.items()):
        if s["total"] == 0:
            continue
        t = s["total"]
        summary = pd.concat([summary, pd.DataFrame([
            {"group": g, "true": f"{t} ảnh",
             "top1_label": f"Top-1: {s['c1']/t:.1%}", "top2_label": f"Top-2: {s['c2']/t:.1%}",
             "top3_label": f"Top-3: {s['c3']/t:.1%}", "top1": s["c1"], "top2": s["c2"], "top3": s["c3"], "score": ""}
        ])])

    df = pd.concat([df, pd.DataFrame([{}]), summary], ignore_index=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
