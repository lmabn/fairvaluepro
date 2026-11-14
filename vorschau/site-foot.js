/* GO:BETTER · Footer für alle Seiten. Eine Quelle: Änderungen nur hier und in site-foot.css. */
(function () {
  var me = document.currentScript, here = location.pathname.split('/').pop();
  var cur = function (f) { return decodeURIComponent(here) === f ? ' aria-current="page"' : ''; };
  var l = function (f, t) { return '<a href="' + f + '"' + cur(f) + '>' + t + '</a>'; };
  var html = '<footer class="gbf"><div class="gbf__in">' +
    l('partner.html', 'Für Partner') +
    '<a href="https://app.gobetter.trade/kuendigen">Verträge hier kündigen</a><a href="https://app.gobetter.trade/widerrufen">Vertrag widerrufen</a>' +
    l('agb.html', 'AGB') + l('datenschutz.html', 'Datenschutz') + l('impressum.html', 'Impressum') + '<span class="gbf__sep" aria-hidden="true"></span>' +
    '<a class="gbf__yt" href="https://www.youtube.com/@gobetter-trade" target="_blank" rel="noopener"><svg viewBox="0 0 16 12" width="16" height="12" aria-hidden="true"><rect width="16" height="12" rx="3" fill="currentColor"/><path d="M6.5 3.5v5l4-2.5z" fill="#F7F8FA"/></svg>YouTube</a>' +
    '<p class="gbf__copy">© ' + new Date().getFullYear() + ' GO:BETTER. Alle Rechte vorbehalten.</p>' +
    '<p class="gbf__note"><b>Keine Anlageberatung. Alle Ergebnisse beruhen auf deinen eigenen Annahmen.</b> Alle Werte in den Ansichten sind Beispieldaten. · Bildmotive KI-generiert.<br>✱ Preise verstehen sich als Endpreise · Gemäß § 19 UStG (Kleinunternehmerregelung) wird keine Umsatzsteuer ausgewiesen.</p>' +
    '</div></footer>';
  if (me) me.insertAdjacentHTML('afterend', html); else document.body.insertAdjacentHTML('beforeend', html);
})();
