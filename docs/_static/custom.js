// Make a sidebar group heading toggle its section (instead of navigating to the group landing page).
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('.bd-sidebar-primary li.toctree-l1 > a.reference').forEach(function (a) {
    var details = a.parentElement.querySelector(':scope > details');
    if (!details) return;
    a.addEventListener('click', function (e) {
      e.preventDefault();
      details.open = !details.open;
    });
  });
});
