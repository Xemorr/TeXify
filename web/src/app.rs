use leptos::prelude::*;
use leptos_meta::{provide_meta_context, MetaTags, Stylesheet, Title};
use leptos_router::{
    components::{Route, Router, Routes},
    StaticSegment,
};
use crate::app::classifier::Classifier;
use crate::app::footer::Footer;
use crate::app::install_button::InstallButton;

mod classifier;
mod footer;
mod install_button;

pub fn shell(options: LeptosOptions) -> impl IntoView {
    view! {
        <!DOCTYPE html>
        <html lang="en">
            <head>
                <script src="https://cdn.jsdelivr.net/npm/@membrane/wasm-stack-trace@0.1.1/dist/index.js" integrity="sha256-4ven6yDmBLbFNTJ3e+BT6LHZgVL9XbXNKYDUhAa/S0Y=" crossorigin="anonymous"></script>
                <script type="module" src="include-mathjax-component.js"></script>
                <meta charset="utf-8"/>
                <meta name="viewport" content="width=device-width, initial-scale=1"/>
                <meta name="description" content="Draw LaTeX symbols and get instant recognition with our machine learning classifier. Works offline!"/>
                <link rel="manifest" href="/manifest.json"/>
                <link rel="icon" href="/latex-logo-trimmed-filled-in.webp"/>
                <link rel="apple-touch-icon" href="/latex-logo-trimmed-filled-in.webp"/>
                <meta name="theme-color" content="#000000"/>
                <meta name="apple-mobile-web-app-capable" content="yes"/>
                <meta name="apple-mobile-web-app-status-bar-style" content="black"/>
                <meta name="apple-mobile-web-app-title" content="TypeIt"/>
                <AutoReload options=options.clone() />
                <HydrationScripts options/>
                <MetaTags/>
            </head>
            <body>
                <App/>
                <script>"if ('serviceWorker' in navigator) { navigator.serviceWorker.register('/sw.js'); }"</script>
            </body>
        </html>
    }
}

#[component]
pub fn App() -> impl IntoView {
    // Provides context that manages stylesheets, titles, meta tags, etc.
    provide_meta_context();

    view! {
        // injects a stylesheet into the document <head>
        // id=leptos means cargo-leptos will hot-reload this stylesheet
        <Stylesheet id="leptos" href="/classifier.css"/>

        // sets the document title
        <Title text="LaTeX Symbol Classifier"/>

        // content for this welcome page
        <Router>
            <main>
                <Routes fallback=|| "Page not found.".into_view()>
                    <Route path=StaticSegment("") view=HomePage/>
                </Routes>
            </main>
        </Router>
    }
}

/// Renders the home page of your application.
#[component]
fn HomePage() -> impl IntoView {
    view! {
        <>
            <h1 style="display: flex; align-items: center; gap: 10px;">
                <img src="latex-logo-trimmed.webp" style="height: 1em;" alt="LaTeX"></img>
                "Symbol Classifier"
            </h1>
            <InstallButton />
            <Classifier />
            <Footer />
        </>
    }
}

