(function () {
  const PAGES = [
    { href: 'leaderboard.html',  label: 'Leaderboard' },
    { href: 'dashboard.html',    label: 'Dashboard' },
    { href: 'top1_model.html',   label: 'Top Model' },
    { href: 'v2_rf_v22.html',    label: 'Signal Chart' },
  ];

  const current = location.pathname.split('/').pop() || 'leaderboard.html';

  const style = document.createElement('style');
  style.textContent = `
    #app-nav {
      position: sticky;
      top: 0;
      z-index: 9999;
      display: flex;
      align-items: center;
      gap: 2px;
      padding: 0 16px;
      background: #0f1117;
      border-bottom: 1px solid #2B2B43;
      height: 36px;
      flex-shrink: 0;
    }
    #app-nav a {
      padding: 6px 14px;
      font-size: 12px;
      font-family: -apple-system, BlinkMacSystemFont, sans-serif;
      font-weight: 500;
      color: #787b86;
      text-decoration: none;
      border-radius: 4px;
      white-space: nowrap;
      transition: color 0.15s, background 0.15s;
    }
    #app-nav a:hover { color: #d1d4dc; background: #1e222d; }
    #app-nav a.nav-active { color: #fff; background: #2a2e39; }
  `;
  document.head.prepend(style);

  const nav = document.createElement('nav');
  nav.id = 'app-nav';
  nav.innerHTML = PAGES.map(p =>
    `<a href="${p.href}"${p.href === current ? ' class="nav-active"' : ''}>${p.label}</a>`
  ).join('');

  if (document.body) {
    document.body.prepend(nav);
  } else {
    document.addEventListener('DOMContentLoaded', () => document.body.prepend(nav));
  }
})();
