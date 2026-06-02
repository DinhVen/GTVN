"""
Đánh giá accuracy trên tập test data_test/.

Cấu trúc test:
    data_test/{Nhóm biển}/{Mã biển}/1.png

Label đúng được lấy từ tên thư mục mã biển.
Script chỉ đánh giá các mã biển có tồn tại trong dataset_aug để tránh tính nhầm
các thư mục dư hoặc nhãn không có trong bộ dữ liệu chính.

Chạy:
    1. Mở backend: python backend/main.py
    2. Chạy đánh giá: python backend/evaluate.py
"""

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
        group_path = os.path.join(DATASET_DIR, group)
        if not os.path.isdir(group_path):
            continue

        for label in sorted(os.listdir(group_path)):
            label_path = os.path.join(group_path, label)
            if os.path.isdir(label_path):
                valid.add((group, label))

    return valid


def collect_test_images(valid_labels):
    images = []
    skipped = []

    for group in sorted(os.listdir(TEST_DIR)):
        group_path = os.path.join(TEST_DIR, group)
        if not os.path.isdir(group_path):
            continue

        for label in sorted(os.listdir(group_path)):
            label_path = os.path.join(group_path, label)
            if not os.path.isdir(label_path):
                continue

            if (group, label) not in valid_labels:
                skipped.append(f"{group}/{label}")
                continue

            for img_file in sorted(os.listdir(label_path)):
                if img_file.lower().endswith(IMAGE_EXTS):
                    images.append((group, label, img_file, os.path.join(label_path, img_file)))

    return images, skipped


def check_server():
    try:
        requests.get("http://localhost:8000/", timeout=5)
    except Exception:
        print("Server chưa chạy. Mở terminal khác và chạy: python backend/main.py")
        sys.exit(1)


def main():
    check_server()

    valid_labels = collect_valid_labels()
    test_images, skipped = collect_test_images(valid_labels)

    print(f"Tổng mã biển hợp lệ trong dataset_aug: {len(valid_labels)}")
    print(f"Tổng ảnh test hợp lệ: {len(test_images)}")
    if skipped:
        print(f"Bỏ qua {len(skipped)} thư mục test không hợp lệ:")
        for item in skipped[:20]:
            print(f"  - {item}")
        if len(skipped) > 20:
            print("  ...")
    print()

    results = []
    group_stats = defaultdict(
        lambda: {"correct1": 0, "correct2": 0, "correct3": 0, "total": 0, "errors": []}
    )

    for index, (group, true_label, img_file, img_path) in enumerate(test_images, start=1):
        try:
            with open(img_path, "rb") as file:
                response = requests.post(
                    API_URL,
                    files={"file": (img_file, file, "image/png")},
                    params={"top_k": 3},
                    timeout=30,
                )

            if response.status_code != 200:
                top_labels = [f"HTTP_{response.status_code}", "NONE", "NONE"]
                results.append(
                    {
                        "group": group,
                        "true": true_label,
                        "top1_label": top_labels[0],
                        "top2_label": top_labels[1],
                        "top3_label": top_labels[2],
                        "top1": False,
                        "top2": False,
                        "top3": False,
                        "score": 0,
                        "image_path": img_path,
                    }
                )
                group_stats[group]["total"] += 1
                group_stats[group]["errors"].append(f"{true_label} -> HTTP {response.status_code}")
                print(f"[{index}/{len(test_images)}] FAIL {true_label}/{img_file}: HTTP {response.status_code}")
                continue

            top_results = response.json().get("results", [])
            top_labels = [item.get("label", "NONE") for item in top_results[:3]]
            while len(top_labels) < 3:
                top_labels.append("NONE")

            top1_label, top2_label, top3_label = top_labels[:3]
            ok1 = true_label == top1_label
            ok2 = true_label in top_labels[:2]
            ok3 = true_label in top_labels[:3]
            score = top_results[0].get("score", 0) if top_results else 0

            results.append(
                {
                    "group": group,
                    "true": true_label,
                    "top1_label": top1_label,
                    "top2_label": top2_label,
                    "top3_label": top3_label,
                    "top1": ok1,
                    "top2": ok2,
                    "top3": ok3,
                    "score": score,
                    "image_path": img_path,
                }
            )

            group_stats[group]["total"] += 1
            if ok1:
                group_stats[group]["correct1"] += 1
            else:
                group_stats[group]["errors"].append(f"{true_label} -> {top1_label}")
            if ok2:
                group_stats[group]["correct2"] += 1
            if ok3:
                group_stats[group]["correct3"] += 1

            mark = "OK" if ok1 else "NO"
            print(f"[{index}/{len(test_images)}] {mark} {true_label} -> {top1_label} ({score:.0%})")

        except Exception as exc:
            print(f"[{index}/{len(test_images)}] ERROR {true_label}/{img_file}: {exc}")

    if not results:
        print("Không có kết quả đánh giá.")
        return

    c1 = sum(1 for item in results if item["top1"])
    c2 = sum(1 for item in results if item["top2"])
    c3 = sum(1 for item in results if item["top3"])
    total = len(results)

    print("\n" + "=" * 50)
    print("KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 50)
    print(f"Tổng ảnh: {total}")
    print(f"Top-1 Accuracy: {c1}/{total} = {c1 / total:.1%}")
    print(f"Top-2 Accuracy: {c2}/{total} = {c2 / total:.1%}")
    print(f"Top-3 Accuracy: {c3}/{total} = {c3 / total:.1%}")

    print("\nTheo nhóm:")
    for group, stats in sorted(group_stats.items()):
        total_group = stats["total"]
        if total_group == 0:
            continue
        a1 = stats["correct1"] / total_group
        a2 = stats["correct2"] / total_group
        a3 = stats["correct3"] / total_group
        print(
            f"  {group}: "
            f"Top1={a1:.1%}, Top2={a2:.1%}, Top3={a3:.1%} "
            f"({stats['correct1']}/{total_group})"
        )
        for error in stats["errors"][:5]:
            print(f"    - {error}")

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nChi tiết đã lưu: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
