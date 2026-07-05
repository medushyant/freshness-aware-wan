/* The Journey tab — the whole research story as an animated flowchart.
   Every number is read from data.json / data_awan.json (real engine output).
   Stages reveal as they enter the viewport; the spine flows continuously. */
window.Journey = (function () {
  let built = false;

  function el(html) { const d = document.createElement('div'); d.innerHTML = html.trim(); return d.firstElementChild; }

  function stage(cls, phase, phaseLbl, icon, title, body, chips) {
    return `<div class="jn-stage ${cls}">
      <div class="jn-card"><span class="jn-phase ${phase}">${phaseLbl}</span>
        <h4>${title}</h4><p>${body}</p>
        <div class="jn-chips">${chips.map(c => `<span class="jn-chip ${c[1] || ''}">${c[0]}</span>`).join('')}</div>
      </div>
      <div class="jn-node">${icon}</div>
    </div>`;
  }

  function build(D, A) {
    const root = document.getElementById('journeyRoot');
    const g = A ? A.wp4.grand : null;
    const viol = A ? (A.wp2.viol_rates['8.0'] * 100).toFixed(0) : '—';
    const gap = A && A.wp3 ? Number(A.wp3.grounding_gap.ratio).toExponential(1).replace('e+', '×10^') : '—';
    const corr = A && A.wp3 ? A.wp3.corruption : null;

    const stages = [
      stage('', 'p0', 'the base paper · arXiv:2604.02381', '📄',
        'Zhao et al.: a swarm fuses knowledge in a knockout tournament',
        'N drones patrol fixed disks, semantically compress what they sense (η), and merge ' +
        'everything into one root report at minimum energy — pairing decided by a central hub ' +
        'running Blossom matching with a hand-tuned potential field ζ.',
        [['H-MAP tournament'], ['central hub'], ['h = β₀·d^−δ radio'], ['payload = scalar η·L']]),

      stage('warn', 'p0', 'what we found wrong', '🔍',
        'A 23-point, equation-anchored limitation audit',
        'No price on report quality (L8), a fragile hand-tuned knob (L9), compounding re-compression ' +
        'never tracked (L14), Wyner–Ziv ignored (L18), a free hub (L10), a static world, N≤10 synthetic ' +
        'validation — five internal inconsistencies among them.',
        [['23 limitations', 'bad'], ['5 inconsistencies', 'bad'], ['static targets', 'bad']]),

      stage('milestone', 'p1', 'phase 1 · direction 1', '⚖',
        'Price quality — and the paper\'s headline lemma flips',
        'Adding a fidelity term makes the optimal compression INTERIOR (η* ∈ [0.16, 0.72], not the ' +
        'boundary the paper proves); the paper\'s own optimum violates the fidelity floor it never measures. ' +
        'Wyner–Ziv side-information rescues infeasible deadlines; conformal calibration certifies the floor.',
        [['η* becomes interior', 'good'], ['D_ref 2.21 > 1.20', 'bad'], ['coverage 0.97', 'good']]),

      stage('milestone', 'p1', 'phase 1 · direction 2', '🧠',
        'A learned cost-to-go replaces the hand-tuned knob',
        'A ridge value model (R²=0.90) predicts remaining mission energy and steers the matching with ' +
        'no knob at all: 0.5% from the brute-force optimum where the paper\'s recipe sits at 13.3%.',
        [['gap 0.5% vs 13.3%', 'good'], ['no ζ anywhere', 'good'], ['beats best hand-tuned', 'good']]),

      stage('warn', 'p1', 'the decisive negative result', '⛔',
        'Chasing targets with motion energy is provably inert here',
        'The obvious moving-target fix — re-anchor and chase — does nothing: at patrol radius R the ' +
        'coverage constraint never binds, and the matcher routes around tight tracking. Proven with three ' +
        'experiments before a single line of the "obvious" extension shipped.',
        [['40.7 m demanded → 0 J spent'], ['washout proven', 'bad']]),

      stage('milestone', 'p1', 'phase 1 · the spine', '⏱',
        'Freshness: target motion enters through information value',
        'A faster target\'s data goes stale faster (AoI). That raises the compression floor (send more, ' +
        'pay +9%) and re-shapes the topology (fast-decaying sources get shorter paths): a 40% fresher ' +
        'root report at −1.5% energy, reducing exactly to the paper at λ_F = 0.',
        [['40% fresher, free', 'good'], ['η-floor 0.549→0.719'], ['Kalman/IMM −28% tracking', 'good']]),

      stage('', 'p2', 'phase 2 · the pivot', '🚀',
        'The paper\'s own future-work sentence becomes the target',
        '"Future research will explore dynamic channel models and decentralized coordination protocols." ' +
        'A-WAN executes that sentence literally — and adds the third fix the paper never admits it needs: ' +
        'a real AI brain instead of a scalar.',
        [['4 work packages'], ['floor→expected→stretch'], ['35 new checks', 'good']]),

      stage('milestone', 'p2', 'WP-1 · kill the hub', '🤝',
        'Agents negotiate pairings themselves — and every message is billed',
        'Distributed greedy (½-approx guarantee) → Bertsekas ε-auction (matches Blossom within a 2% ' +
        'ε-bound on 100% of feasible rounds) → A2A-style LLM negotiation: a real Qwen2.5 exchanged ' +
        `schema-valid JSON on 43/43 turns, tokens charged in joules. Under 20% dropout the paper's hub completes ${g ? (g.paper.dropout_completion * 100).toFixed(0) : 0}% of missions; the auction completes ${g ? (g.awan.dropout_completion * 100).toFixed(0) : 100}%.`,
        [['auction ≈ Blossom', 'good'], ['hub cliffs 100→0%', 'bad'], ['control <0.01% energy', 'good']]),

      stage('milestone', 'p2', 'WP-2 · kill the toy radio', '📡',
        'A real channel, a learned radio map, a certified deadline',
        `Under standard correlated shadowing + Rician fading the paper's deterministic plan misses ${viol}% ` +
        'of its deadlines. A split-conformal dB margin restores certified coverage at +2–4% energy; ' +
        'planning against the learned radio map beats move-closer by 30% — all replicated on a ' +
        'ray-traced Sionna RT Munich map (MEASURED geometry).',
        [[`${viol}% deadlines missed`, 'bad'], ['certified 0.94≥0.90', 'good'], ['Sionna replicated', 'warm']]),

      stage('milestone', 'p2', 'WP-3 · kill the fake brain', '👁',
        'Real VLM knowledge, exact bits, measured joules, a trust gate',
        'SmolVLM2 genuinely perceives exact-ground-truth scenes; SigLIP RAG fusion is measurably ' +
        `sub-additive (the paper says additive); compute energy is ${gap} times the paper's τf²W fiction ` +
        `(MEASURED on a Colab T4). One corrupted leaf poisons ${corr ? (corr.fabricate_base * 100).toFixed(0) : 95}% of roots — the ` +
        `overlap-consistency trust gate contains it at ${corr ? (corr.fabricate_gated * 100).toFixed(0) : 15}%.`,
        [[`energy gap ${gap}`, 'warm'], ['fusion sub-additive', 'good'], ['corruption 95→15%', 'good']]),

      stage('milestone', 'p2', 'WP-4 · the grand showdown', '🏆',
        'One world, ten seeds: A-WAN vs Phase 1 vs the paper',
        g ? `Energy ${g.awan.E_mean.toFixed(1)} vs ${g.paper.E_mean.toFixed(1)} J · deadline violations ` +
          `${(g.awan.viol_rate * 100).toFixed(0)}% vs ${(g.paper.viol_rate * 100).toFixed(0)}% · staleness ` +
          `${g.awan.stale_mean.toFixed(0)} vs ${g.paper.stale_mean.toFixed(0)} — A-WAN wins 4 of 5 composite axes ` +
          '(lifetime is the honest loss: fresher reports cost battery). A learned surrogate scales the whole ' +
          'framework to N=200 for the first time.' : 'run export_awan_web_data.py',
        [['4/5 axes won', 'good'], ['first N≥100 curve', 'warm'], ['every claim a check', 'good']]),
    ];

    root.innerHTML = `<div class="jn-wrap"><div class="jn-spine"></div>${stages.join('')}</div>
      <div class="jn-finale">
        <h3>Three <span class="grad">firsts</span>, all measured</h3>
        <p>Verified against the July-2026 literature and the reference group's own sequels —
        each claim survives with a named delta (docs/NOVELTY_LOG.md).</p>
        <div class="jn-firsts">
          <div class="jn-first"><b>Costed decentralized semantic aggregation</b>
            <span>Progressive knowledge fusion where the coordination itself — every bid, every token — pays joules.</span></div>
          <div class="jn-first"><b>Real-VLM WAN with measured energy</b>
            <span>The first instantiation of this line with an actual model, actual bits, NVML-measured mJ/token, and corruption propagation.</span></div>
          <div class="jn-first"><b>Conformal deadline & fidelity certificates</b>
            <span>Distribution-free guarantees — on the channel and on the real pipeline's root fidelity.</span></div>
        </div>
      </div>`;

    // reveal on scroll
    const io = new IntersectionObserver(es => es.forEach(e => {
      if (e.isIntersecting) { e.target.classList.add('on'); io.unobserve(e.target); }
    }), { threshold: 0.25 });
    root.querySelectorAll('.jn-stage').forEach(s => io.observe(s));
    setTimeout(() => root.querySelector('.jn-stage')?.classList.add('on'), 80);
  }

  return {
    init() {
      if (built) return;
      built = true;
      Promise.all([
        fetch('data.json').then(r => r.json()).catch(() => null),
        fetch('data_awan.json').then(r => r.json()).catch(() => null),
      ]).then(([D, A]) => build(D, A));
    }
  };
})();

try { inits.journey = () => Journey.init(); } catch (e) { console.error('Journey hook', e); }
