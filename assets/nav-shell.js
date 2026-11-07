(function(){
  var burger = document.querySelector('.site-nav .burger');
  if (!burger) return;
  function setMenu(open){
    document.body.classList.toggle('is-menu-open', open);
    burger.setAttribute('aria-expanded', open);
  }
  burger.addEventListener('click', function(){
    setMenu(!document.body.classList.contains('is-menu-open'));
  });
  document.addEventListener('keydown', function(e){
    if (e.key === 'Escape'){ setMenu(false); closeDd(); }
  });
  var mql = window.matchMedia('(min-width:1100px)');
  if (mql.addEventListener) mql.addEventListener('change', function(m){ if(m.matches) setMenu(false); });

  var acc = document.querySelector('.site-mmenu #mm-fvp'), accBtn = acc && acc.querySelector('.mm-acc__btn');
  if (acc && accBtn){
    accBtn.addEventListener('click', function(){
      var o = !acc.classList.contains('is-open');
      acc.classList.toggle('is-open', o);
      accBtn.setAttribute('aria-expanded', o);
    });
  }

  var dd = document.querySelector('.site-nav #dd-fvp'), trig = dd && dd.querySelector('.dd__trigger'), t;
  function openDd(){ if (!dd) return; clearTimeout(t); dd.classList.add('is-open'); trig.setAttribute('aria-expanded', true); }
  function closeDd(){ if (!dd) return; dd.classList.remove('is-open'); trig.setAttribute('aria-expanded', false); }
  if (dd && trig){
    dd.addEventListener('mouseenter', openDd);
    dd.addEventListener('mouseleave', function(){ t = setTimeout(closeDd, 120); });
    dd.addEventListener('focusin', openDd);
    dd.addEventListener('focusout', function(e){ if (!dd.contains(e.relatedTarget)) closeDd(); });
  }

  var q = new URLSearchParams(location.search);
  if (q.get('menu') === 'open') setMenu(true);
  if (q.get('acc') === 'open' && accBtn) accBtn.click();
  if (q.get('dd') === 'open') openDd();
})();
