"""investor_dashboard.html 内の DATA 変数を output/dashboard_data.json で更新する。"""
import json
import re
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SOURCE_HTML = "investor_dashboard.html"
JSON_PATH = "output/dashboard_data.json"
OUTPUT_HTML = "output/investor_dashboard_latest.html"


def update_dashboard():
    if not os.path.exists(SOURCE_HTML):
        print(f"{SOURCE_HTML} が見つかりません。")
        return
    if not os.path.exists(JSON_PATH):
        print(f"{JSON_PATH} が見つかりません。先に main.py を実行してください。")
        return

    with open(SOURCE_HTML, "r", encoding="utf-8") as f:
        html = f.read()
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    data_js = json.dumps(data, ensure_ascii=False, indent=2)
    pattern = r"const DATA = \{.*?\n\};"
    replacement = f"const DATA = {data_js};"
    new_html, count = re.subn(pattern, replacement, html, flags=re.DOTALL)

    if count == 0:
        print("DATA変数が見つかりませんでした。")
        return

    os.makedirs("output", exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(new_html)

    print(f"更新完了: {OUTPUT_HTML} ({data.get('updated', '')})")


if __name__ == "__main__":
    update_dashboard()
