"use strict";

const copyButton = document.querySelector("#copy-bibtex");
const citation = document.querySelector("#citation");
const copyStatus = document.querySelector("#copy-status");

copyButton.addEventListener("click", async () => {
  try {
    if (!navigator.clipboard) throw new Error("Clipboard unavailable");
    await navigator.clipboard.writeText(citation.textContent.trim());
    copyStatus.textContent = "Citation copied.";
    copyButton.textContent = "Copied";
  } catch {
    const range = document.createRange();
    range.selectNodeContents(citation);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    copyStatus.textContent = "Citation selected. Press Ctrl+C or ⌘C to copy.";
  }
});
