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
               liest body nicht mehr aus, damit der Kopf nicht dunkel über einem
               hellen Abschnitt hängt.
               Bekannter Rest: Bug 309956 (iOS 26.4 beta) — body-Farbwechsel per JS
               triggert nicht immer einen Statusleisten-Repaint.
       ?sb=3 — baut auf Weg 2 auf. Zusätzlich bekommt die .gbn im dunklen Zustand
               ein Gradient oben: deckend #070C10 an der Oberkante, nach unten
               weich transparent. Oberkante des Kopfs wirkt als Fortsetzung der
               Statusleiste, backdrop-filter bleibt darunter. Farbwahl #070C10:
               deckt den Hero-Overlay und den gemessenen Bildschatten
               (rgb(7,12,16) am linken Rand).
       ?sb=4 — Weg 3 PLUS Repaint-Nudges gegen Bug 309956. Nach jedem Farbwechsel
               werden mehrere Mittel kombiniert, damit Safari die Statusleiste
               neu zeichnet:
                 a) html-Hintergrund mit umschalten (manche Safari-Fassungen
                    sampeln html als Fallback vor body);
                 b) meta[name=theme-color] aus dem Head entfernen und wieder
                    einfügen — das ist stärker als ein setAttribute, weil es die
                    Preference-Beobachter neu initialisiert;
                 c) ein 1-px-hohes fixed Element am Top in der neuen Farbe für
                    einen Frame einblenden und wieder entfernen — zwingt Safari,
                    die Compositor-Ebene am Statusleisten-Rand neu zu bauen;
                 d) erzwungenes Reflow über offsetWidth-Lesen.
       ?sb=5 — Ohne Farbwechsel per Skript. body ist dauerhaft dunkel, das
               theme-color ist fest #070C10. Dunkler Kopf arbeitet wie in sb=3
               (Glas + Gradient). Bei <=719px hat der helle Kopf keinen
               backdrop-filter und ist deckend #F7F8FA, damit Safari ihn als
               festes Element mit solider Farbe erkennt und die Leiste hell
               macht. Ab 720px bleibt das Glas. Nebenwirkung an mobilen
               Überrollen beobachten und melden (body-dunkel ist auch oben und
               unten sichtbar).
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
  if (SB === '5') {
    /* body-Hintergrund dauerhaft dunkel, theme-color fest; syncTheme() macht
       dann für sb=5 nichts mehr. CSS-Hänger in site-nav.css: html.sb5 {...} */
    ROOT.classList.add('sb5');
    if (themeMeta) themeMeta.setAttribute('content', '#070C10');
  }

  function syncTheme() {
    var dark = hdr.classList.contains('gbn--dark') || ROOT.classList.contains('gbn-open-dark');
    /* sb=5 erbt den Weg-3-Verlauf am dunklen Kopf; der Rest wird für sb=5 übersprungen */
    if (SB === '3' || SB === '4' || SB === '5') hdr.classList.toggle('gbn--sb3-top', dark);
    if (SB === '5') return;
    var color = dark ? '#0B0D10' : '#F7F8FA';
    if (themeMeta) themeMeta.setAttribute('content', color);
    if (SB === '1' && sbStripe) sbStripe.style.background = color;
    if (SB === '2' || SB === '3' || SB === '4') document.body.style.backgroundColor = color;
    if (SB === '4') nudgeStatusbar(color);
  }

  /* ?sb=4 — Kombi-Nudge für die Statusleiste. Hintergrund:
       Bug 309956 — body-background-Wechsel per JS triggert in Safari iOS 26+
       nicht immer einen Statusleisten-Repaint. Workarounds aus der WebKit-
       Community kombiniert, weil kein einzelner als zuverlässig bestätigt ist. */
  function nudgeStatusbar(color) {
    ROOT.style.backgroundColor = color;
    if (themeMeta && themeMeta.parentNode) {
      var p = themeMeta.parentNode, n = themeMeta.nextSibling;
      p.removeChild(themeMeta); p.insertBefore(themeMeta, n);
    }
    var tag = document.createElement('div');
    tag.setAttribute('aria-hidden', 'true');
    tag.style.cssText = 'position:fixed;top:0;left:0;right:0;height:1px;z-index:99;pointer-events:none;background:' + color;
    document.body.appendChild(tag);
    void document.body.offsetWidth;  /* Reflow */
    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        if (tag.parentNode) tag.parentNode.removeChild(tag);
      });
    });
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
        /* Bei sb=2/3/4/5 ist body dunkel (gesetzt per JS oder CSS) — nicht zurücklesen */
        if ((SB === '2' || SB === '3' || SB === '4' || SB === '5') && n === document.body) continue;
        var l = lum(getComputedStyle(n).backgroundColor); if (l !== null) return l < .35;
      }
      break;
    }
    if (SB === '2' || SB === '3' || SB === '4' || SB === '5') return false;
    var b = lum(getComputedStyle(document.body).backgroundColor); return b !== null && b < .35;
  }
  var ticking = false;
  function upd() { ticking = false; hdr.classList.toggle('gbn--dark', ground()); syncTheme(); }
  addEventListener('scroll', function () { if (!ticking) { ticking = true; requestAnimationFrame(upd); } }, { passive: true });
  addEventListener('resize', upd); addEventListener('load', upd);
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', upd); else upd();
})();
