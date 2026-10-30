import os
import sys
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

def find_project():
    home = Path.home()
    candidates = [
        home / "OneDrive" / "Desktop" / "ML_Algorithm_Battle_Arena",
        home / "Desktop" / "ML_Algorithm_Battle_Arena",
        Path.cwd() / "ML_Algorithm_Battle_Arena",
        Path.cwd(),
    ]
    for p in candidates:
        if (p / "app.py").exists() and (p / "static").exists():
            return p
    return None

def request(url, timeout=8):
    req = urllib.request.Request(url, headers={"User-Agent": "ML-Arena-Diagnostic/1.0", "Cache-Control": "no-cache"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            return r.status, r.headers.get("Content-Type", ""), body[:150000].decode("utf-8", "replace"), None
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", "replace")[:5000]
        except Exception:
            body = ""
        return e.code, e.headers.get("Content-Type", ""), body, "HTTP error"
    except Exception as e:
        return None, "", "", repr(e)

def main():
    lines = []
    lines.append("ML ARENA DIAGNOSTIC REPORT")
    lines.append("Created: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("Python: " + sys.version.replace("\n", " "))
    project = find_project()
    lines.append("\n[PROJECT]")
    if project:
        lines.append("Project folder: " + str(project))
        for rel in ["app.py", "static/js/app.js", "static/css/style.css", "templates/index.html"]:
            f = project / rel
            lines.append(f"{rel}: " + (f"FOUND ({f.stat().st_size} bytes; modified {time.ctime(f.stat().st_mtime)})" if f.exists() else "MISSING"))
        js_path = project / "static" / "js" / "app.js"
        if js_path.exists():
            js = js_path.read_text(encoding="utf-8", errors="replace")
            lines.append("Local JS contains runBattle: " + str("runBattle" in js))
            lines.append("Local JS contains battle API path: " + str("/api/battle" in js))
            lines.append("Local JS contains duplicate-click guard: " + str("battleInProgress" in js))
        html_path = project / "templates" / "index.html"
        if html_path.exists():
            html = html_path.read_text(encoding="utf-8", errors="replace")
            lines.append("HTML references static/js/app.js: " + str("static', filename='js/app.js'" in html or 'static/js/app.js' in html))
            lines.append("HTML contains battle form: " + str('id="battle-form"' in html))
            lines.append("HTML contains run button: " + str('id="configure-battle"' in html))
    else:
        lines.append("Could not automatically find project folder.")
        lines.append("Current folder: " + str(Path.cwd()))

    lines.append("\n[LOCAL SERVER CHECKS]")
    checks = [
        ("Home page", "http://127.0.0.1:5000/"),
        ("JavaScript served by Flask", "http://127.0.0.1:5000/static/js/app.js"),
        ("Datasets API", "http://127.0.0.1:5000/api/datasets"),
        ("Classification algorithms API", "http://127.0.0.1:5000/api/algorithms?task_type=classification"),
        ("Regression algorithms API", "http://127.0.0.1:5000/api/algorithms?task_type=regression"),
        ("History API", "http://127.0.0.1:5000/api/history"),
    ]
    for label, url in checks:
        status, ctype, body, err = request(url)
        lines.append(f"\n{label}: {url}")
        if err:
            lines.append("RESULT: FAILED — " + err)
        else:
            lines.append(f"RESULT: HTTP {status}; content-type={ctype}; body length={len(body)}")
            lines.append("Body preview: " + " ".join(body[:450].split()))
            if label == "JavaScript served by Flask":
                lines.append("Served JS contains runBattle: " + str("runBattle" in body))
                lines.append("Served JS contains duplicate-click guard: " + str("battleInProgress" in body))
                lines.append("Served JS looks like HTML error page: " + str("<html" in body[:300].lower()))
    lines.append("\n[NOTES]")
    lines.append("This report only sends GET requests. It does not create, modify, or delete any data.")
    lines.append("It does not submit a battle. It checks whether the Flask app and its front-end/API routes respond.")
    out = Path.home() / "Desktop" / "ML_Arena_Diagnostic_Report.txt"
    if not out.parent.exists():
        out = Path.home() / "OneDrive" / "Desktop" / "ML_Arena_Diagnostic_Report.txt"
    try:
        out.write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        out = Path.cwd() / "ML_Arena_Diagnostic_Report.txt"
        out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\n\nREPORT SAVED TO:\n" + str(out))
    try:
        os.startfile(str(out))
    except Exception:
        pass
    input("\nPress Enter to close this window...")

if __name__ == "__main__":
    main()
