// Make a sidebar group heading toggle its section (instead of navigating to the group landing page).
// Forward the click to the native <summary> so the theme's own collapse/state handling stays consistent.
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('.bd-sidebar-primary li.toctree-l1.has-children > a.reference').forEach(function (a) {
    var summary = a.parentElement.querySelector(':scope > details > summary');
    if (!summary) return;
    a.addEventListener('click', function (e) {
      e.preventDefault();
      summary.click();
    });
  });
});
