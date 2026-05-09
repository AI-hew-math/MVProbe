document.addEventListener('DOMContentLoaded', function () {
  const copyBtn = document.getElementById('copy-bibtex');
  if (copyBtn) {
    copyBtn.addEventListener('click', function () {
      const bibtex = document.getElementById('bibtex-content').innerText;
      navigator.clipboard.writeText(bibtex).then(function () {
        const original = copyBtn.innerText;
        copyBtn.innerText = 'Copied!';
        setTimeout(function () { copyBtn.innerText = original; }, 1500);
      });
    });
  }
});
