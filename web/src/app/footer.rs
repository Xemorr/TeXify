use leptos::prelude::{ElementChild, StyleAttribute};
use leptos::{component, IntoView};
use leptos::prelude::ClassAttribute;
use crate::app::view;

#[component]
pub fn footer() -> impl IntoView {
    view! {
        <div class="footer">
            <h1>"FAQ"</h1>
            <h2>"Inspiration & Credit"</h2>
            <p>"Credit to the " <a href="https://detexify.kirelabs.org/classify.html">"original detexify app"</a> ", their data was used to train this classifier."</p>
            <h2>"How does this app work?"</h2>
            <p>"It's written in Rust using Leptos, compiled to web assembly in order to have great performance running a ML model from the browser. This leads to lower latency, and if you install this as an app, you can continue to use it even if this website goes down!"</p>
        </div>
    }
}