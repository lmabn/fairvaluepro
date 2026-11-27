/* GO:BETTER · Verhalten der Nav (Burger-Menü, Dropdown, Hell/Dunkel, Escape).
   Markup steht in _includes/site-nav.html und wird von Jekyll ausgeliefert.
   Eine Quelle je Bauteil: Markup im Include, Verhalten hier, Stil in site-nav.css. */
(function () {
  var ROOT = document.documentElement, hdr = document.getElementById('gbn'), menu = document.getElementById('gbn-m');
  if (!hdr || !menu) return;
  var burger = hdr.querySelector('.gbn__burger');

  /* 27.11.2026: theme-color für die iPhone-Statusleiste mit dem Kopf mitführen.
     #0B0D10 solange der Kopf dunkel ist (gbn--dark oder gbn-open-dark), sonst #F7F8FA.
     Werte aus site-nav.css gelesen, nicht geschätzt. */
  var themeMeta = document.querySelector('meta[name="theme-color"]');
  function syncTheme() {
    if (!themeMeta) return;
    var dark = hdr.classList.contains('gbn--dark') || ROOT.classList.contains('gbn-open-dark');
    themeMeta.setAttribute('content', dark ? '#0B0D10' : '#F7F8FA');
  }

  function setOpen(o) {
    if (o) {
      /* Zustand der Kopfzeile im Moment des Öffnens einfrieren (Ludmila 15.11.2026):
         War die Nav dunkel, bleibt das Mobilmenü dunkel offen; war sie hell, hell. */
      ROOT.classList.toggle('gbn-open-dark', hdr.classList.contains('gbn--dark'));
    } else {
      ROOT.classList.remove('gbn-open-dark');
    }
    ROOT.classList.toggle('gbn-open', o);
    burger.setAttribute('aria-expanded', String(o));
    syncTheme();
  }
  burger.addEventListener('click', function () { setOpen(!ROOT.classList.contains('gbn-open')); });
  menu.querySelectorAll('a').forEach(function (a) { a.addEventListener('click', function () { setOpen(false); }); });
  menu.querySelectorAll('.gbn-m__acc').forEach(function (acc) {
    var b = acc.querySelector('.gbn-m__accbtn');
    b.addEventListener('click', function () { var o = !acc.classList.contains('is-open'); acc.classList.toggle('is-open', o); b.setAttribute('aria-expanded', String(o)); });
  });
  var dd = hdr.querySelector('.gbn__dd');
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') { setOpen(false); if (dd) dd.classList.remove('is-open'); if (document.activeElement && hdr.contains(document.activeElement)) document.activeElement.blur(); } });
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
      for (var n = el; n && n !== ROOT; n = n.parentElement) {
        if (n.dataset && n.dataset.nav) return n.dataset.nav === 'dark';
        var l = lum(getComputedStyle(n).backgroundColor); if (l !== null) return l < .35;
      }
      break;
    }
    var b = lum(getComputedStyle(document.body).backgroundColor); return b !== null && b < .35;
  }
  var ticking = false;
  function upd() { ticking = false; hdr.classList.toggle('gbn--dark', ground()); syncTheme(); }
  addEventListener('scroll', function () { if (!ticking) { ticking = true; requestAnimationFrame(upd); } }, { passive: true });
  addEventListener('resize', upd); addEventListener('load', upd);
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', upd); else upd();
})();
