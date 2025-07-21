// Show loading spinner on form submit
const form = document.querySelector('.sa-form');
const loadingOverlay = document.getElementById('sa-loading-overlay');
if (form && loadingOverlay) {
    form.addEventListener('submit', () => {
        loadingOverlay.classList.remove('d-none');
        setTimeout(() => loadingOverlay.classList.add('d-none'), 6000); // fallback hide
    });
}
// Hide spinner on page load (in case)
window.addEventListener('DOMContentLoaded', () => {
    if (loadingOverlay) loadingOverlay.classList.add('d-none');
}); 