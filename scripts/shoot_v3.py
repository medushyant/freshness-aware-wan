"""UI v3 — headless verification of EVERY tab (screenshots + console gate)."""
import os
import subprocess
import time

from playwright.sync_api import sync_playwright

ROOT = "/Users/dushyantkumar/Documents/BTP_ILAC_WAN"
srv = subprocess.Popen(["/opt/homebrew/bin/python3", "-m", "http.server", "5054",
                        "--directory", "web"], cwd=ROOT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1.5)
errs = []
TABS = ["overview", "journey", "swarm", "hmap", "blossom", "learned",
        "freshness", "graphs", "verify", "compare", "awan"]
try:
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1440, "height": 940}, device_scale_factor=2)
        pg.on("console", lambda m: errs.append((m.type, m.text[:160]))
              if m.type == "error" else None)
        pg.on("pageerror", lambda e: errs.append(("pageerror", str(e)[:160])))
        pg.goto("http://localhost:5054/", wait_until="networkidle")
        os.makedirs(f"{ROOT}/web/shots", exist_ok=True)
        for tab in TABS:
            pg.click(f"#tabs button[data-tab='{tab}']")
            pg.wait_for_timeout(1700)
            if tab == "journey":
                pg.mouse.wheel(0, 2600); pg.wait_for_timeout(900)
                pg.mouse.wheel(0, 2600); pg.wait_for_timeout(900)
                pg.mouse.wheel(0, -6000); pg.wait_for_timeout(400)
            if tab == "hmap":
                pg.click("#hmapStep"); pg.wait_for_timeout(800)
                pg.click("#hmapStep"); pg.wait_for_timeout(1000)
            if tab == "blossom":
                pg.click("#blShow"); pg.wait_for_timeout(900)
            pg.screenshot(path=f"{ROOT}/web/shots/v3_{tab}.png")
        n_j = pg.locator(".jn-stage").count()
        b.close()
    ok = not errs and n_j >= 10
    print(f"{'PASS' if ok else 'FAIL'}  UI-v3 all {len(TABS)} tabs "
          f"| journey stages={n_j} | console errors={len(errs)}")
    for e in errs[:8]:
        print("  ", e)
finally:
    srv.terminate()
