// Force .ipynb download links to trigger a file save dialog instead
// of opening the JSON in the browser tab.  ReadTheDocs serves .ipynb
// with Content-Type: application/json, which browsers display inline.
// The HTML5 `download` attribute overrides that behaviour.
document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll('a[href$=".ipynb"]').forEach(function (a) {
        if (!a.hasAttribute("download")) {
            // Use the filename from the href as the suggested save name
            var parts = a.href.split("/");
            a.setAttribute("download", parts[parts.length - 1]);
        }
    });
});
