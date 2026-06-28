# Deploy the demo — and which host to use

This is a **fully static** site (`index.html` + `css/` + `js/` + `data.json` +
`figures/`). No build, no backend, no server code. That means it deploys
anywhere static, instantly, for free.

## Run locally first
```bash
cd web
python3 -m http.server 5050      # open http://localhost:5050
```

## Recommended host: **Vercel** (your instinct is correct)

Vercel is the best fit here — free, instant, gives you a clean public URL with
HTTPS and a global CDN, and redeploys in seconds. Two ways:

**A. Drag-and-drop (no tools):** go to https://vercel.com → "Add New… → Project"
→ choose "Deploy a static folder" / or use https://vercel.com/new and drag the
`web/` folder. Done — you get `https://<name>.vercel.app`.

**B. CLI (repeatable):**
```bash
npm i -g vercel
cd web
vercel            # first run: log in, accept defaults → preview URL
vercel --prod     # promote to your public production URL
```
No config is needed because everything is static and already in `web/`. (If you
ever want to pin it, a one-line `vercel.json` with `{ "cleanUrls": true }` is
plenty.)

## Equally fine alternatives
- **Netlify Drop** — https://app.netlify.com/drop, drag the `web/` folder, instant
  `*.netlify.app` URL. Zero account friction.
- **GitHub Pages** — push the repo, Settings → Pages → serve from the folder that
  holds `index.html`. URL: `https://<user>.github.io/<repo>/`. Good if you want the
  link tied to the source repo.
- **Cloudflare Pages** — similar to Vercel, generous free tier.

## Before you deploy
- Make sure `web/figures/` is included (the Graphs tab reads it) — it is.
- The site fetches `data.json` relatively, so it must be served over http(s)
  (any host above does this); opening `index.html` via `file://` will block the
  fetch — use a host or the local server command above.

**Bottom line:** Vercel (option A or B). It's the cleanest path to a public link
you can hand to reviewers, and redeploying after any tweak is one command.
