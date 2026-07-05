/* UI v3 theme layer — loaded after app.js, before any chart instantiates.
   Global Chart.js polish (rounded bars, soft grids, styled tooltips) and
   GSAP micro-interactions. No data touched. */
(function () {
  if (window.Chart) {
    const mut = '#a89fc7';
    Chart.defaults.color = mut;
    Chart.defaults.font.family = "'Inter',sans-serif";
    Chart.defaults.font.size = 12;
    Chart.defaults.borderColor = 'rgba(168,160,220,.10)';
    Chart.defaults.elements.bar.borderRadius = 9;
    Chart.defaults.elements.bar.borderSkipped = false;
    Chart.defaults.elements.line.tension = 0.32;
    Chart.defaults.elements.line.borderWidth = 2.5;
    Chart.defaults.elements.point.radius = 3.2;
    Chart.defaults.elements.point.hoverRadius = 5.5;
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.pointStyle = 'rectRounded';
    Chart.defaults.plugins.legend.labels.boxWidth = 12;
    Chart.defaults.plugins.legend.labels.padding = 14;
    Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(24,19,44,.96)';
    Chart.defaults.plugins.tooltip.borderColor = 'rgba(168,160,220,.25)';
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.cornerRadius = 12;
    Chart.defaults.plugins.tooltip.padding = 12;
    Chart.defaults.plugins.tooltip.titleFont = { family: "'Sora'", weight: '700' };
    Chart.defaults.animation.duration = 800;
    Chart.defaults.animation.easing = 'easeOutQuart';
  }

  /* staggered card reveal on every tab switch */
  const reveal = tab => {
    if (!window.gsap) return;
    const p = document.getElementById('tab-' + tab);
    if (!p) return;
    const items = p.querySelectorAll(
      ':scope > .card, .vizrow > .card, .grid2 > .card, .tourgrid > .tcard, ' +
      '.hero-stats > .stat, .checkgrid > .cchip, .ba-cards > .ba-card, #awanRoot > *');
    if (!items.length) return;
    gsap.fromTo(items, { y: 24, opacity: 0 },
      { y: 0, opacity: 1, duration: .55, stagger: .06, ease: 'power3.out',
        overwrite: 'auto', clearProps: 'transform,opacity' });
  };
  document.querySelectorAll('#tabs button').forEach(b =>
    b.addEventListener('click', () => setTimeout(() => reveal(b.dataset.tab), 30)));
  addEventListener('DOMContentLoaded', () => setTimeout(() => reveal('overview'), 120));

  /* gentle pointer tilt on stage cards */
  document.querySelectorAll('.stage.card').forEach(card => {
    card.addEventListener('pointermove', e => {
      const r = card.getBoundingClientRect();
      const rx = ((e.clientY - r.top) / r.height - .5) * -1.6;
      const ry = ((e.clientX - r.left) / r.width - .5) * 1.6;
      card.style.transform = `perspective(1100px) rotateX(${rx}deg) rotateY(${ry}deg)`;
    });
    card.addEventListener('pointerleave', () => { card.style.transform = ''; });
  });
})();
