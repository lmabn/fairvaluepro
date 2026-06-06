(function () {
  if (localStorage.getItem('theme') === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
  }

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
      btn.innerHTML = isLight() ? '☾' : '☼';
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
