(function () {
  var bar = document.createElement('div');
  bar.id = 'nav-loader';
  document.body.appendChild(bar);

  function startLoad() {
    bar.classList.remove('done');
    bar.classList.add('loading');
  }

  function finishLoad() {
    bar.classList.remove('loading');
    bar.classList.add('done');
    setTimeout(function () { bar.classList.remove('done'); }, 400);
  }

  document.addEventListener('click', function (e) {
    var link = e.target.closest('a[href]');
    if (!link) return;
    var href = link.getAttribute('href');
    if (!href || href === '#' || href.startsWith('javascript:') || href.startsWith('mailto:')) return;
    if (link.getAttribute('target') === '_blank') return;
    if (e.ctrlKey || e.metaKey || e.shiftKey) return;
    startLoad();
  });

  document.addEventListener('submit', function (e) {
    startLoad();
  });

  window.addEventListener('pageshow', function () {
    finishLoad();
  });
})();
