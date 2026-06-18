import json
import os
import warnings

warnings.filterwarnings("ignore")

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SRC_DIR, "..", "data"))

EXPECTED = {
    "bq_corpus": ["train.jsonl", "validation.jsonl", "test.jsonl"],
    "lcqmc":     ["train.jsonl", "validation.jsonl", "test.jsonl"],
}


def check_data():
    print(f"数据目录: {DATA_DIR}")
    all_ok = True
    for dataset, files in EXPECTED.items():
        d = os.path.join(DATA_DIR, dataset)
        if not os.path.isdir(d):
            print(f"  [缺失] {dataset}/")
            all_ok = False
            continue
        for f in files:
            p = os.path.join(d, f)
            if os.path.isfile(p):
                size = os.path.getsize(p)
                print(f"  [OK]   {dataset}/{f}  ({size:,} bytes)")
            else:
                print(f"  [缺失] {dataset}/{f}")
                all_ok = False
    if all_ok:
        print("\n所有数据已就绪，无需下载。")
    else:
        print("\n数据不完整，请检查。")
    return all_ok


if __name__ == "__main__":
    check_data()
