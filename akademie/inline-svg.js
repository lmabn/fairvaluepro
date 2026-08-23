/* Lädt <object type="image/svg+xml"> innerhalb .ak-fig via fetch und ersetzt
   sie durch inline <svg>. Nur so erben die tokenbasierten SVGs die
   CSS-Variablen der Elternseite (--bg, --neon, --text …) und ziehen den
   Dark/Light-Toggle korrekt mit. Fallback bei Fetch-Fehler: das <object>
   bleibt stehen und rendert mit seinen eingebauten Hex-Fallbacks (Dark). */
(function () {
  const nodes = document.querySelectorAll('.ak-fig object[type="image/svg+xml"]');
  if (!nodes.length) return;
  nodes.forEach(async (obj) => {
    const url = obj.getAttribute('data');
    if (!url) return;
    try {
      const res = await fetch(url, { credentials: 'same-origin' });
      if (!res.ok) return;
      const text = await res.text();
      const doc = new DOMParser().parseFromString(text, 'image/svg+xml');
      const svg = doc.documentElement;
      if (!svg || svg.nodeName.toLowerCase() !== 'svg') return;
      const aria = obj.getAttribute('aria-label');
      if (aria) svg.setAttribute('aria-label', aria);
      svg.setAttribute('role', 'img');
      svg.style.display = 'block';
      svg.style.width = '100%';
      svg.style.height = 'auto';
      obj.replaceWith(svg);
    } catch (e) {
      /* Fehler still schlucken — das <object> bleibt als Fallback sichtbar. */
    }
  });
})();
