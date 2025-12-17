/**
 * MathJax Custom Element
 * Usage: <math-jax>$E=mc^2$</math-jax>
 */

// 1. Configure MathJax (Must be set before the library loads)
window.MathJax = {
    loader: {load: ['[tex]/upgreek', '[tex]/autoload']},
    tex: {
        packages: {'[+]': ['upgreek', 'autoload']},
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']]
    },
    startup: {
        // We disable automatic typesetting because the component handles it
        typeset: false
    }
};

// 2. Dynamically inject the MathJax CDN script
(function() {
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js';
    script.async = true;
    document.head.appendChild(script);
})();

// 3. Define the Custom Element Class
class MathJaxElement extends HTMLElement {
    constructor() {
        super();
        this._observer = null;
    }

    connectedCallback() {
        this.render();

        // Observer for dynamic content changes (e.g. AJAX or JS updates)
        this._observer = new MutationObserver(() => this.render());
        this._observer.observe(this, {
            childList: true,
            subtree: true,
            characterData: true
        });
    }

    disconnectedCallback() {
        if (this._observer) {
            this._observer.disconnect();
        }
    }

    render() {
        // Check if MathJax is loaded and the promise interface is available
        if (window.MathJax && window.MathJax.typesetPromise) {
            // Use typesetPromise for smooth rendering
            window.MathJax.typesetPromise([this])
                .catch((err) => console.error('MathJax rendering error:', err));
        } else {
            // If MathJax hasn't loaded yet, retry in 50ms
            setTimeout(() => this.render(), 50);
        }
    }
}

// 4. Register the element
// Check if already defined to prevent errors if script is imported twice
if (!customElements.get('math-jax')) {
    customElements.define('math-jax', MathJaxElement);
}