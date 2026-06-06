(function () {
  if (localStorage.getItem('theme') === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
  }

  var SUN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><line x1="12" y1="2" x2="12" y2="4"/><line x1="12" y1="20" x2="12" y2="22"/><line x1="4.93" y1="4.93" x2="6.34" y2="6.34"/><line x1="17.66" y1="17.66" x2="19.07" y2="19.07"/><line x1="2" y1="12" x2="4" y2="12"/><line x1="20" y1="12" x2="22" y2="12"/><line x1="4.93" y1="19.07" x2="6.34" y2="17.66"/><line x1="17.66" y1="6.34" x2="19.07" y2="4.93"/></svg>';
  var MOON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';

  function init() {
    if (document.querySelector('.theme-fab')) return;

    var btn = document.createElement('button');
    btn.className = 'theme-fab';
    btn.type = 'button';
    btn.setAttribute('aria-label', 'Theme umschalten');

    function isLight() {
      return document.documentElement.getAttribute('data-theme') === 'light';
    }
    function render() {
      btn.innerHTML = isLight() ? MOON : SUN;
      btn.title = isLight() ? 'Dark Mode aktivieren' : 'Light Mode aktivieren';
    }

    btn.addEventListener('click', function () {
      var next = isLight() ? 'dark' : 'light';
      if (next === 'light') {
        document.documentElement.setAttribute('data-theme', 'light');
      } else {
        document.documentElement.removeAttribute('data-theme');
      }
      localStorage.setItem('theme', next);
      render();
    });

    render();
    document.body.appendChild(btn);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
