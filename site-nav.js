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

  /* 27.11.2026 Experiment — nur auf der Startseite, nur per URL-Parameter.
     Ohne Parameter bleibt die Seite exakt wie ohne dieses Stück Code.
     Weg 1 (?sb=1) ist am iPhone wirkungslos (Statusleiste blieb hell) und fällt weg;
     Code bleibt noch stehen für Rückvergleiche.
       ?sb=2 — Hintergrundfarbe des body mit dem Kopf mitführen. Safari samplet
               laut Quellen body zuerst. Nebeneffekte reparieren: .sec, .end--open
               und .end--wall bekommen einen eigenen Hintergrund, und ground()
               liest body nicht mehr aus (siehe unten), damit der Kopf nicht
               dunkel über einem hellen Abschnitt hängt.
               Bekannter Zusatz: Bug 309956 (iOS 26.4 beta) — body-Farbwechsel
               per JS triggert nicht immer einen Statusleisten-Repaint.
       ?sb=3 — baut auf Weg 2 auf (body-Farbe mitführen, Nebeneffekte repariert).
               Zusätzlich bekommt die .gbn im dunklen Zustand ein Gradient oben:
               deckend #070C10 an der Oberkante, nach unten weich transparent.
               So wirkt die Oberkante des Kopfs als Fortsetzung der Statusleiste
               und das Glas (backdrop-filter) bleibt darunter sichtbar.
               Farbwahl #070C10: passt zum Hero-Overlay und zum Bildschatten
               (gemessen rgb(7,12,16) am linken Rand).
  */
  var isHome = (location.pathname === '/' || location.pathname === '/index.html');
  var SB = isHome ? new URLSearchParams(location.search).get('sb') : null;
  var sbStripe = null;
  if (SB === '1') {
    sbStripe = document.createElement('div');
    sbStripe.setAttribute('aria-hidden', 'true');
    sbStripe.style.cssText = 'position:fixed;top:0;left:0;right:0;height:60px;z-index:40;pointer-events:none;background:#F7F8FA';
    document.body.appendChild(sbStripe);
  }

  function syncTheme() {
    var dark = hdr.classList.contains('gbn--dark') || ROOT.classList.contains('gbn-open-dark');
    var color = dark ? '#0B0D10' : '#F7F8FA';
    if (themeMeta) themeMeta.setAttribute('content', color);
    if (SB === '1' && sbStripe) sbStripe.style.background = color;
    if (SB === '2' || SB === '3') document.body.style.backgroundColor = color;
    if (SB === '3') hdr.classList.toggle('gbn--sb3-top', dark);
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
        /* Bei sb=2/3 ändern wir body-background selbst — darf nicht zurückgelesen werden */
        if ((SB === '2' || SB === '3') && n === document.body) continue;
        var l = lum(getComputedStyle(n).backgroundColor); if (l !== null) return l < .35;
      }
      break;
    }
    if (SB === '2' || SB === '3') return false;
    var b = lum(getComputedStyle(document.body).backgroundColor); return b !== null && b < .35;
  }
  var ticking = false;
  function upd() { ticking = false; hdr.classList.toggle('gbn--dark', ground()); syncTheme(); }
  addEventListener('scroll', function () { if (!ticking) { ticking = true; requestAnimationFrame(upd); } }, { passive: true });
  addEventListener('resize', upd); addEventListener('load', upd);
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', upd); else upd();
})();
