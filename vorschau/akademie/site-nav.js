/* GO:BETTER · Menü für alle Seiten. Eine Quelle: Änderungen nur hier und in site-nav.css. */
(function () {
  var me = document.currentScript, page = (me && me.dataset.page) || '';
  var chev = '<svg class="gbn__chev" viewBox="0 0 10 10" aria-hidden="true"><path d="M2 3.5 5 6.5 8 3.5" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>';
  var cur = function (p) { return page === p ? ' aria-current="page"' : ''; };
  var fvpOn = page === 'fvp' || page === 'daten' || page === 'community';
  var dot = function (c) { return '<i class="gbn__dot" style="background:' + c + '"></i>'; };

  var nav = '<header class="gbn" id="gbn"><div class="gbn__in">' +
    '<a class="gbn__wm" href="../index.html" aria-label="GO:BETTER Startseite"><span>GO</span><svg viewBox="0 0 30 71" aria-hidden="true"><circle cx="15" cy="9.5" r="9.5"/><circle cx="15" cy="58" r="13"/></svg><span>BETTER</span></a>' +
    '<span class="gbn__sep" aria-hidden="true"></span>' +
    '<nav class="gbn__links" aria-label="Werkzeuge">' +
      '<div class="gbn__dd"><a href="../fair-value-pro.html" style="--c:#10C27F"' + (fvpOn ? ' aria-current="page"' : '') + ' aria-haspopup="true">' + dot('#10C27F') + 'Fair Value Pro' + chev + '</a>' +
        '<div class="gbn__ddm"><a href="../daten-quelle.html"' + cur('daten') + '>Daten-Quelle</a><a href="../community.html"' + cur('community') + '>Community</a></div></div>' +
      '<a href="../portfolio-tracker.html" style="--c:#F0B42A"' + cur('pt') + '>' + dot('#F0B42A') + 'Portfolio Tracker</a>' +
      '<a href="../investment-planner.html" style="--c:#3B7BFF"' + cur('ip') + '>' + dot('#3B7BFF') + 'Investment Planner</a>' +
    '</nav>' +
    '<div class="gbn__r"><a href="../plaene.html"' + cur('preise') + '>Preise</a><a href="index.html"' + cur('akademie') + '>Akademie</a><span class="gbn__sep gbn__sep--r" aria-hidden="true"></span><a href="https://app.gobetter.trade/fairvalue">Anmelden</a>' +
      '<a class="gbn__btn" href="https://app.gobetter.trade/fairvalue">Kostenlos registrieren</a></div>' +
    '<button class="gbn__burger" type="button" aria-label="Menü" aria-expanded="false" aria-controls="gbn-m"><span></span><span></span></button>' +
  '</div></header>' +
  '<div class="gbn-m" id="gbn-m"><div class="gbn-m__in">' +
    '<div class="gbn-m__acc' + (fvpOn ? ' is-open' : '') + '"><div class="gbn-m__row"><a href="../fair-value-pro.html"' + cur('fvp') + '>' + dot('#10C27F') + 'Fair Value Pro</a>' +
      '<button class="gbn-m__accbtn" type="button" aria-expanded="' + fvpOn + '" aria-label="Unterseiten Fair Value Pro"><svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2.5 4.5 6 8l3.5-3.5" fill="none" stroke="currentColor" stroke-width="1.5"/></svg></button></div>' +
      '<div class="gbn-m__sub"><a href="../daten-quelle.html"' + cur('daten') + '>Daten-Quelle</a><a href="../community.html"' + cur('community') + '>Community</a></div></div>' +
    '<a class="gbn-m__item" href="../portfolio-tracker.html"' + cur('pt') + '>' + dot('#F0B42A') + 'Portfolio Tracker</a>' +
    '<a class="gbn-m__item" href="../investment-planner.html"' + cur('ip') + '>' + dot('#3B7BFF') + 'Investment Planner</a>' +
    '<div class="gbn-m__sec"><a href="../plaene.html"' + cur('preise') + '>Preise</a><a href="index.html"' + cur('akademie') + '>Akademie</a><a href="https://app.gobetter.trade/fairvalue">Anmelden</a></div>' +
    '<a class="gbn-m__cta" href="https://app.gobetter.trade/fairvalue">Kostenlos registrieren</a>' +
  '</div></div>';

  if (me) me.insertAdjacentHTML('afterend', nav); else document.body.insertAdjacentHTML('afterbegin', nav);
  var R = document.documentElement, hdr = document.getElementById('gbn'), menu = document.getElementById('gbn-m');
  var burger = hdr.querySelector('.gbn__burger');

  function setOpen(o) { R.classList.toggle('gbn-open', o); burger.setAttribute('aria-expanded', String(o)); }
  burger.addEventListener('click', function () { setOpen(!R.classList.contains('gbn-open')); });
  menu.querySelectorAll('a').forEach(function (a) { a.addEventListener('click', function () { setOpen(false); }); });
  menu.querySelectorAll('.gbn-m__acc').forEach(function (acc) {
    var b = acc.querySelector('.gbn-m__accbtn');
    b.addEventListener('click', function () { var o = !acc.classList.contains('is-open'); acc.classList.toggle('is-open', o); b.setAttribute('aria-expanded', String(o)); });
  });
  var dd = hdr.querySelector('.gbn__dd');
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') { setOpen(false); dd.classList.remove('is-open'); if (document.activeElement && hdr.contains(document.activeElement)) document.activeElement.blur(); } });
  addEventListener('resize', function () { if (innerWidth >= 960) setOpen(false); });

  /* Hell oder dunkel: Untergrund am linken Rand hinter dem Menü */
  function lum(c) {
    var m = c.match(/rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+))?/); if (!m) return null;
    if (m[4] !== undefined && +m[4] < .5) return null;
    return (0.2126 * m[1] + 0.7152 * m[2] + 0.0722 * m[3]) / 255;
  }
  function ground() {
    var y = hdr.offsetHeight / 2, list = document.elementsFromPoint ? document.elementsFromPoint(2, y) : [];
    for (var i = 0; i < list.length; i++) {
      var el = list[i];
      if (hdr.contains(el) || menu.contains(el)) continue;
      for (var n = el; n && n !== R; n = n.parentElement) {
        if (n.dataset && n.dataset.nav) return n.dataset.nav === 'dark';
        var l = lum(getComputedStyle(n).backgroundColor); if (l !== null) return l < .35;
      }
      break;
    }
    var b = lum(getComputedStyle(document.body).backgroundColor); return b !== null && b < .35;
  }
  var ticking = false;
  function upd() { ticking = false; hdr.classList.toggle('gbn--dark', ground()); }
  addEventListener('scroll', function () { if (!ticking) { ticking = true; requestAnimationFrame(upd); } }, { passive: true });
  addEventListener('resize', upd); addEventListener('load', upd);
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', upd); else upd();
})();
