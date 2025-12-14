use leptos::prelude::ElementChild;
use leptos::{component, IntoView};
use leptos::prelude::ClassAttribute;
use crate::app::view;

#[component]
pub fn footer() -> impl IntoView {
    view! {
        <div class="container">
            <p>"Credit to the " <a href="https://detexify.kirelabs.org/classify.html">"original detexify app"</a>", their data was used to train this classifier."</p>
            <br/>
            <h2>"Why use this version?"</h2>
            <p>"It performs classification locally, so performs faster, can be installed as an app on any device and uses modern ML techniques (CNN)"</p>
        </div>
    }
}