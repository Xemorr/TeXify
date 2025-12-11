export function typeset() {
    return window.MathJax?.typeset();
}
export function typesetPromise() {
    return window.MathJax?.typesetPromise();
}
export function clearTypeset(elements) {
    return window.MathJax?.startup.document.clearMathItemsWithin(elements);
}