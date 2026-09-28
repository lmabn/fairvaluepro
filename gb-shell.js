/*
 * GO:BETTER — Shell (Nav Burger-Toggle)
 * gb-shell.js · v1.0
 *
 * Toggle für das mobile Burger-Menü in .gb-nav-wrap.
 * Verhält sich still, wenn die Elemente auf einer Seite fehlen.
 */
(function(){
  var burger = document.getElementById('gb-burger');
  var menu = document.getElementById('gb-mobile-menu');
  if (!burger || !menu) return;
  burger.addEventListener('click', function(){
    var open = menu.classList.toggle('is-open');
    burger.setAttribute('aria-expanded', String(open));
    burger.setAttribute('aria-label', open ? 'Menü schließen' : 'Menü öffnen');
  });
  menu.querySelectorAll('a').forEach(function(a){
    a.addEventListener('click', function(){
      menu.classList.remove('is-open');
      burger.setAttribute('aria-expanded', 'false');
    });
  });
})();
