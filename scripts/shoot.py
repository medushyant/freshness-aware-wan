"""Headless screenshot of every tab + console-error check (preview MCP is
blocked here by a venv sandbox clash, so we verify with Playwright)."""
import subprocess, time, signal, os
from playwright.sync_api import sync_playwright

ROOT="/Users/dushyantkumar/Documents/BTP_ILAC_WAN"
srv=subprocess.Popen(["/opt/homebrew/bin/python3","-m","http.server","5052","--directory","web"],
                     cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
time.sleep(1.5)
errs=[]
TABS=["overview","swarm","hmap","blossom","learned","freshness","graphs","verify","compare"]
try:
    with sync_playwright() as p:
        b=p.chromium.launch()
        pg=b.new_page(viewport={"width":1280,"height":860},device_scale_factor=2)
        pg.on("console",lambda m:errs.append((m.type,m.text)) if m.type in("error","warning") else None)
        pg.on("pageerror",lambda e:errs.append(("pageerror",str(e))))
        pg.goto("http://localhost:5052/",wait_until="networkidle")
        os.makedirs(f"{ROOT}/web/shots",exist_ok=True)
        for tab in TABS:
            pg.click(f"#tabs button[data-tab='{tab}']")
            pg.wait_for_timeout(1600)          # let charts/animations render
            if tab=="hmap":
                pg.click("#hmapStep");pg.wait_for_timeout(700);pg.click("#hmapStep");pg.wait_for_timeout(900)
            if tab=="blossom":
                pg.click("#blShow");pg.wait_for_timeout(900)
            pg.screenshot(path=f"{ROOT}/web/shots/{tab}.png")
        b.close()
    print("SHOTS OK")
finally:
    srv.send_signal(signal.SIGTERM)
print("CONSOLE ERRORS/WARNINGS:",len(errs))
for t,m in errs[:20]:print(" -",t,m[:160])
