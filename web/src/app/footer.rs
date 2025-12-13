use leptos::prelude::ElementChild;
use leptos::{component, IntoView};
use leptos::prelude::ClassAttribute;
use crate::app::view;

#[component]
pub fn footer() -> impl IntoView {
    view! {
        <div class="container">
            <p>"Credit to the " <a href="https://detexify.kirelabs.org/classify.html">original detexify app</a>", their data was used to train this classifier."</p>
        </div>
    }
}