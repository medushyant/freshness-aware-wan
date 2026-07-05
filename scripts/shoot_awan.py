"""I6 — headless verification of the A-WAN tab (additive twin of shoot.py)."""
import os
import subprocess
import time

from playwright.sync_api import sync_playwright

ROOT = "/Users/dushyantkumar/Documents/BTP_ILAC_WAN"
srv = subprocess.Popen(["/opt/homebrew/bin/python3", "-m", "http.server", "5053",
                        "--directory", "web"], cwd=ROOT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1.5)
errs = []
try:
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1280, "height": 900},
                        device_scale_factor=2)
        pg.on("console", lambda m: errs.append((m.type, m.text))
              if m.type == "error" else None)
        pg.on("pageerror", lambda e: errs.append(("pageerror", str(e))))
        pg.goto("http://localhost:5053/", wait_until="networkidle")
        pg.click("#tabs button[data-tab='awan']")
        pg.wait_for_timeout(2200)
        os.makedirs(f"{ROOT}/web/shots", exist_ok=True)
        pg.screenshot(path=f"{ROOT}/web/shots/awan.png", full_page=True)
        n_cards = pg.locator("#awanRoot .card").count()
        n_checks = pg.locator("#awChecks tr").count()
        b.close()
    ok = (not errs) and n_cards >= 5 and n_checks >= 20
    print(f"{'PASS' if ok else 'FAIL'}  I6 A-WAN tab renders "
          f"| {n_cards} cards, {n_checks} check rows, {len(errs)} console errors")
    for e in errs[:5]:
        print("  ", e)
finally:
    srv.terminate()
