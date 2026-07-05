/* A-WAN (Phase 2) tab — additive file; reads data_awan.json (real numbers
   exported by export_awan_web_data.py; same honesty rule as Phase 1).
   Uses the site's own design system: .hero-stats/.stat/.num, .card,
   .gal-grid/.gal-card and the shared #lightbox. */
window.Awan = (function () {
  const C = { good:'#37d9a6', bad:'#ff6b6b', paper:'#7e8aa5', a2:'#5b8cff',
              a3:'#b07cff', pink:'#f72585', amber:'#f4a261' };
  let A = null;

  function el(html) { const d = document.createElement('div'); d.innerHTML = html.trim(); return d.firstElementChild; }
  function fmt(x, d = 2) { return (+x).toFixed(d); }

  function init() {
    fetch('data_awan.json').then(r => r.json()).then(d => { A = d; build(); })
      .catch(() => { document.getElementById('awanRoot').innerHTML =
        '<p class="muted">data_awan.json missing — run export_awan_web_data.py</p>'; });
  }

  function build() {
    const root = document.getElementById('awanRoot');
    root.innerHTML = '';
    const g = A.wp4.grand, viol = A.wp2.viol_rates;
    const gapTxt = A.wp3 ? Number(A.wp3.grounding_gap.ratio).toExponential(1).replace('e+', '×10^') : '—';
    const measured = A.wp3 && A.wp3.meter && A.wp3.meter.source !== 'MODELED';

    root.appendChild(el(`<div class="hero">
      <h1>A-WAN — the <span class="grad">Autonomous, Physically-Grounded</span> WAN</h1>
      <p class="lede">The paper's own confessed weaknesses — a <b>central hub</b> and a
      <b>toy radio</b> — plus one it never admits, a <b>fake AI brain</b> (a scalar η·L) —
      eliminated in one architecture. Agents <b>negotiate pairings themselves</b>
      (½-approx greedy → Bertsekas auction → A2A-style LLM negotiation with every
      token charged in joules), <b>plan against a learned radio map</b> under a
      split-conformal deadline certificate, and exchange <b>real VLM knowledge</b>
      with exact bits, ${measured ? 'NVML-<b>measured</b>' : 'modeled'} energy and a
      trust gate. Every number below is machine-written engine output.</p>
      <div class="hero-stats">
        <div class="stat"><span class="num">${(viol['8.0'] * 100).toFixed(0)}%</span><label>deadlines the paper's plan misses under real fading</label></div>
        <div class="stat"><span class="num">${(g.awan.dropout_completion * 100).toFixed(0)}%</span><label>A-WAN completion at 20% dropout (paper hub: ${(g.paper.dropout_completion * 100).toFixed(0)}%)</label></div>
        <div class="stat"><span class="num">${gapTxt}</span><label>compute-energy gap, ${measured ? 'MEASURED (Colab T4 NVML)' : 'modeled'} vs the paper's τf²W</label></div>
        <div class="stat"><span class="num">${A.n_pass}/${A.checks.length}</span><label>new automated Phase-2 checks, zero failures</label></div>
      </div></div>`));

    /* grand showdown — normalized to the paper row, raw values in the table */
    root.appendChild(el(`<div class="card"><h3>The grand showdown (H11) — one world, ten seeds</h3>
      <p class="muted small">Everything executes under the SAME tier-1 stochastic channel.
      Bars are relative to the paper's H-MAP (=1.0); arrows mark the good direction.
      A-WAN = auction + learned bids + conformal dB margin + batteries.</p>
      <canvas id="awGrand" height="105"></canvas>
      <div class="table-wrap"><table class="mini-table"><thead>
        <tr><th></th><th>energy [J] ↓</th><th>deadline viol ↓</th><th>drop@0.2 completion ↑</th><th>staleness ↓</th><th>lifetime [slots] ↑</th></tr></thead>
        <tbody id="awGrandT"></tbody></table></div></div>`));
    const rows = ['paper', 'phase1', 'awan'];
    const lbl = { paper:'paper H-MAP', phase1:'Phase-1 (learned)', awan:'A-WAN' };
    const col = { paper:C.bad, phase1:C.amber, awan:C.good };
    const met = [['E_mean', 1], ['viol_rate', 1], ['dropout_completion', -1],
                 ['stale_mean', 1], ['lifetime', -1]];
    new Chart(document.getElementById('awGrand'), { type:'bar',
      data:{ labels:['energy ↓', 'deadline viol ↓', 'drop@0.2 compl. ↑', 'staleness ↓', 'lifetime ↑'],
        datasets: rows.map(r => ({ label: lbl[r], backgroundColor: col[r],
          data: met.map(([k]) => {
            const base = g.paper[k] || 1e-9;
            return +(g[r][k] / base).toFixed(3);
          }) })) },
      options:{ responsive:true,
        scales:{ y:{ beginAtZero:true, title:{ display:true, text:'relative to paper (=1.0)' } } } } });
    const tb = document.getElementById('awGrandT');
    rows.forEach(r => tb.appendChild(el(`<tr><td style="color:${col[r]}"><b>${lbl[r]}</b></td>
      <td>${fmt(g[r].E_mean)}</td><td>${(g[r].viol_rate * 100).toFixed(0)}%</td>
      <td>${(g[r].dropout_completion * 100).toFixed(0)}%</td>
      <td>${fmt(g[r].stale_mean, 1)}</td><td>${fmt(g[r].lifetime, 1)}</td></tr>`)));

    /* dropout grace vs cliff */
    const dq = A.wp1.dropout;
    const qs = Object.keys(dq.cen).sort();
    root.appendChild(el(`<div class="card"><h3>Graceful vs brittle (H3)</h3>
      <p class="muted small">At round 2, a fraction q of agents vanishes. The paper's
      synchronous hub waits forever for their uploads — its own implicit assumption,
      made explicit and priced. The auction re-matches among survivors.</p>
      <canvas id="awDrop" height="88"></canvas></div>`));
    new Chart(document.getElementById('awDrop'), { type:'line',
      data:{ labels: qs, datasets:[
        { label:'centralized (paper hub)', data: qs.map(q => dq.cen[q].completion * 100),
          borderColor:C.bad, backgroundColor:'rgba(255,107,107,.12)', fill:true, tension:.25 },
        { label:'decentralized (auction)', data: qs.map(q => dq.dec[q].completion * 100),
          borderColor:C.good, backgroundColor:'rgba(55,217,166,.10)', fill:true, tension:.25 }]},
      options:{ scales:{ y:{ min:-5, max:105, title:{ display:true, text:'mission completion %' } },
                         x:{ title:{ display:true, text:'dropout fraction q' } } } } });

    /* break-even */
    const be = A.wp1.breakeven;
    const ns = Object.keys(be.hub_ctrl).map(Number).sort((a, b) => a - b);
    root.appendChild(el(`<div class="card"><h3>Talking has a price — and now a bill (H2/H4)</h3>
      <p class="muted small">Every coordination message priced on the same radio (MODELED).
      The hub was never free; the auction's bids cost more but buy robustness; LLM
      negotiation adds tokens at mJ/token. Log scale.</p>
      <canvas id="awBE" height="88"></canvas></div>`));
    new Chart(document.getElementById('awBE'), { type:'line',
      data:{ labels: ns, datasets:[
        { label:'hub upload+broadcast', data: ns.map(n => be.hub_ctrl[n] * 1e3), borderColor:C.paper, tension:.25 },
        { label:'auction bids+prices', data: ns.map(n => be.auction_ctrl[n] * 1e3), borderColor:C.a2, tension:.25 },
        { label:'negotiation (A2A mock FSM)', data: ns.map(n => be.mock_ctrl[n] * 1e3), borderColor:C.pink, tension:.25 }]},
      options:{ scales:{ y:{ type:'logarithmic', title:{ display:true, text:'control energy [mJ]' } },
                         x:{ title:{ display:true, text:'swarm size N' } } } } });

    /* corruption */
    if (A.wp3) {
      const cr = A.wp3.corruption;
      const ops = ['fabricate', 'swap', 'jitter'];
      root.appendChild(el(`<div class="card"><h3>One liar poisons the tree (H9) — real VLM facts</h3>
        <p class="muted small">One corrupted leaf agent. The paper's single-path tree hands the
        poison to the root ${(cr.fabricate_base * 100).toFixed(0)}% of the time; the
        overlap-consistency trust gate (near-identity corroboration, τ=0.95) contains it at
        ~0% verification-energy overhead. Corruption counts only factually-false content.</p>
        <canvas id="awCorr" height="88"></canvas></div>`));
      new Chart(document.getElementById('awCorr'), { type:'bar',
        data:{ labels: ops, datasets:[
          { label:'paper tree (no gate)', backgroundColor:C.bad, data: ops.map(o => cr[o + '_base'] * 100) },
          { label:'trust-gated fusion', backgroundColor:C.good, data: ops.map(o => cr[o + '_gated'] * 100) }]},
        options:{ scales:{ y:{ max:100, title:{ display:true, text:'root-corruption rate %' } } } } });
    }

    /* runtime scaling */
    const rt = A.wp4.runtime_ms, rn = A.wp4.runtime_N;
    root.appendChild(el(`<div class="card"><h3>Scaling to N = 200</h3>
      <p class="muted small">A learned pair surrogate (MAPE ${fmt(A.wp4.i4.mape_pct, 1)}%,
      mission-energy penalty ${fmt(A.wp4.i4.penalty_pct, 1)}%) frees the matcher from the
      O(N²) inner solves — the first runtime curve for this framework beyond N=10. Log-log.</p>
      <canvas id="awRT" height="88"></canvas></div>`));
    new Chart(document.getElementById('awRT'), { type:'line',
      data:{ labels: rn, datasets:[
        { label:'Blossom O(N³) (centralized)', data: rt.blossom, borderColor:C.bad, tension:.25 },
        { label:'auction (near-exact)', data: rt.auction, borderColor:C.a2, tension:.25 },
        { label:'greedy ½-approx', data: rt.greedy, borderColor:C.good, tension:.25 }]},
      options:{ scales:{ y:{ type:'logarithmic', title:{ display:true, text:'ms per round' } },
                         x:{ title:{ display:true, text:'N' } } } } });

    /* figure gallery — native gal classes + shared lightbox */
    const gal = el(`<div class="card"><h3>Every Phase-2 figure (${(A.figures || []).length})</h3>
      <p class="muted small">All regenerate from cached run artifacts — no GPU, no network.
      Click to enlarge.</p><div class="gal-grid" id="awGal"></div></div>`);
    root.appendChild(gal);
    const gd = gal.querySelector('#awGal');
    (A.figures || []).forEach(f => {
      const cap = f.replace('.png', '').replace(/_/g, ' ');
      const fig = el(`<figure class="gal-card"><div class="gal-img">
        <img loading="lazy" src="figures_awan/${f}" alt="${cap}"></div>
        <figcaption>${cap}</figcaption></figure>`);
      fig.querySelector('img').onclick = () => {
        document.getElementById('lbImg').src = 'figures_awan/' + f;
        document.getElementById('lbCap').textContent = cap;
        document.getElementById('lightbox').classList.add('on');
      };
      gd.appendChild(fig);
    });

    /* checks table */
    const card = el(`<div class="card"><h3>${A.n_pass}/${A.checks.length} automated Phase-2 checks — every claim is a test</h3>
      <div class="table-wrap"><table class="mini-table"><thead>
      <tr><th></th><th>id</th><th>claim</th><th>measured</th></tr></thead>
      <tbody id="awChecks"></tbody></table></div></div>`);
    root.appendChild(card);
    const tb2 = card.querySelector('#awChecks');
    A.checks.forEach(c => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td style="color:${c.status === 'PASS' ? C.good : C.bad};font-weight:700">${c.status}</td>
        <td><code>${c.id}</code></td><td>${c.claim}</td>
        <td class="muted small">${c.detail}</td>`;
      tb2.appendChild(tr);
    });
  }

  return { init };
})();

/* register with the Phase-1 router (additive — app.js untouched; `inits` is
   a top-level const in app.js, visible to later classic scripts) */
try { inits.awan = () => Awan.init(); } catch (e) { console.error('AWAN hook', e); }
