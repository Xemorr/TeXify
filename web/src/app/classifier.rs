use crate::app::classifier::state::{build_and_load_model, MyB};
use burn::backend::ndarray::NdArrayDevice;
use burn::tensor::activation::softmax;
use leptos::control_flow::For;
use leptos::prelude::{signal, Effect, Get, NodeRef, NodeRefAttribute, OnAttribute, Set, Show, StyleAttribute, IntoAny};
use leptos::prelude::{ClassAttribute, ElementChild};
use leptos::wasm_bindgen::JsCast;
use leptos::{component, view, IntoView};
use leptos::html::{Custom, InnerHtmlAttribute};
use shared::item::{HEIGHT, WIDTH};
use shared::model::Model;
use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::iter::zip;
use std::rc::Rc;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};
use crate::app::classifier::model::{Prediction, SharedModel};

mod keys;
mod state;
mod model;

fn get_logical_coords(e: &web_sys::MouseEvent, canvas: &HtmlCanvasElement) -> (f64, f64) {
    let rect = canvas.get_bounding_client_rect();
    let scale_x = canvas.width() as f64 / rect.width();
    let scale_y = canvas.height() as f64 / rect.height();

    let x = (e.client_x() as f64 - rect.left()) * scale_x;
    let y = (e.client_y() as f64 - rect.top()) * scale_y;

    (x, y)
}

fn get_touch_coords(e: &web_sys::TouchEvent, canvas: &HtmlCanvasElement) -> Option<(f64, f64)> {
    let touch = e.touches().get(0)?;
    let rect = canvas.get_bounding_client_rect();
    let scale_x = canvas.width() as f64 / rect.width();
    let scale_y = canvas.height() as f64 / rect.height();

    let x = (touch.client_x() as f64 - rect.left()) * scale_x;
    let y = (touch.client_y() as f64 - rect.top()) * scale_y;

    Some((x, y))
}

#[component]
pub fn Classifier() -> impl IntoView {
    let model = Rc::new(RefCell::new(SharedModel::new()));
    let canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    let processed_canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    let (drawing, set_drawing) = signal(false);
    let (prediction, set_predictions) = signal(None::<Vec<Prediction>>);
    let (classifying, set_classifying) = signal(false);

    // Initialize canvas drawing on mount
    let setup_canvas = move || {
        if let Some(canvas) = canvas_ref.get() {
            let canvas: HtmlCanvasElement = canvas.clone().into();
            let ctx = canvas
                .get_context("2d")
                .unwrap()
                .unwrap()
                .dyn_into::<CanvasRenderingContext2d>()
                .unwrap();

            ctx.set_line_width(1.5);
            ctx.set_line_cap("round");
            ctx.set_stroke_style_str("black");
        }
    };

    let perform_classification = Rc::new(move || {
        if classifying.get() {
            return; // Skip if already classifying
        }
        set_classifying.set(true);
        if let Some(canvas) = canvas_ref.get() {
            // Spawn async task to avoid blocking the canvas
            let model_inner = Rc::clone(&model);
            wasm_bindgen_futures::spawn_local(async move {
                let canvas: HtmlCanvasElement = canvas.clone().into();
                let ctx = canvas
                    .get_context("2d")
                    .unwrap()
                    .unwrap()
                    .dyn_into::<CanvasRenderingContext2d>()
                    .unwrap();

                // Get image data from canvas
                let image_data_obj = ctx
                    .get_image_data(0.0, 0.0, canvas.width() as f64, canvas.height() as f64)
                    .unwrap();

                let data = image_data_obj.data();
                let canvas_width = canvas.width() as usize;
                let canvas_height = canvas.height() as usize;

                // Create a temporary ImageBuffer for resizing
                let img_buffer = image::ImageBuffer::from_fn(canvas_width as u32, canvas_height as u32, |x, y| {
                    let luma_idx = (y as usize * canvas_width + x as usize) * 4 + 3; // * 4 + 3 to get the luma component
                    image::Luma([data.0[luma_idx]])
                });

                // Resize to 32x32
                let resized = image::imageops::resize(&img_buffer, 32, 32, image::imageops::FilterType::Triangle);

                // Convert to [[f32; 32]; 32] and normalize
                let mut image_array = [[0.0f32; WIDTH]; HEIGHT];
                for y in 0..HEIGHT {
                    for x in 0..WIDTH {
                        let pixel = resized.get_pixel(x as u32, y as u32);

                        // Normalize to 0.0-1.0
                        image_array[y][x] = (pixel[0] as f32) / 255.0;
                    }
                }

                // Run inference
                match model_inner.borrow_mut().inference(image_array).await {
                    predictions => {
                        set_predictions.set(Some(predictions));
                    }
                }
                set_classifying.set(false);
            });
        }
    });

    let on_mouse_down = move |e: web_sys::MouseEvent| {
        set_drawing.set(true);
        if let Some(canvas) = canvas_ref.get() {
            let canvas: HtmlCanvasElement = canvas.clone().into();
            let ctx = canvas
                .get_context("2d")
                .unwrap()
                .unwrap()
                .dyn_into::<CanvasRenderingContext2d>()
                .unwrap();

            let rect = canvas.get_bounding_client_rect();
            let (x, y) = get_logical_coords(&e, &canvas);

            ctx.begin_path();
            ctx.move_to(x, y);
        }
    };

    let perform_classification_clone = Rc::clone(&perform_classification);
    let on_mouse_move = move |e: web_sys::MouseEvent| {
        if drawing.get() {
            if let Some(canvas) = canvas_ref.get() {
                let canvas: HtmlCanvasElement = canvas.clone().into();
                let ctx = canvas
                    .get_context("2d")
                    .unwrap()
                    .unwrap()
                    .dyn_into::<CanvasRenderingContext2d>()
                    .unwrap();

                let rect = canvas.get_bounding_client_rect();
                let (x, y) = get_logical_coords(&e, &canvas);

                ctx.line_to(x, y);
                ctx.stroke();
            }
        }
    };

    let perform_classification_clone2 = Rc::clone(&perform_classification);
    let on_mouse_up = move |_| {
        set_drawing.set(false);
        // Final classification when done drawing
        perform_classification_clone2();
    };

    // Touch event handlers
    let on_touch_start = move |e: web_sys::TouchEvent| {
        e.prevent_default();
        set_drawing.set(true);
        if let Some(canvas) = canvas_ref.get() {
            let canvas: HtmlCanvasElement = canvas.clone().into();
            let ctx = canvas
                .get_context("2d")
                .unwrap()
                .unwrap()
                .dyn_into::<CanvasRenderingContext2d>()
                .unwrap();

            if let Some((x, y)) = get_touch_coords(&e, &canvas) {
                ctx.begin_path();
                ctx.move_to(x, y);
            }
        }
    };

    let on_touch_move = move |e: web_sys::TouchEvent| {
        e.prevent_default();
        if drawing.get() {
            if let Some(canvas) = canvas_ref.get() {
                let canvas: HtmlCanvasElement = canvas.clone().into();
                let ctx = canvas
                    .get_context("2d")
                    .unwrap()
                    .unwrap()
                    .dyn_into::<CanvasRenderingContext2d>()
                    .unwrap();

                if let Some((x, y)) = get_touch_coords(&e, &canvas) {
                    ctx.line_to(x, y);
                    ctx.stroke();
                }
            }
        }
    };

    let perform_classification_clone3 = Rc::clone(&perform_classification);
    let on_touch_end = move |e: web_sys::TouchEvent| {
        e.prevent_default();
        set_drawing.set(false);
        // Final classification when done drawing
        perform_classification_clone3();
    };

    let clear_canvas = move |_| {
        if let Some(canvas) = canvas_ref.get() {
            let canvas: HtmlCanvasElement = canvas.clone().into();
            let ctx = canvas
                .get_context("2d")
                .unwrap()
                .unwrap()
                .dyn_into::<CanvasRenderingContext2d>()
                .unwrap();

            ctx.clear_rect(0.0, 0.0, canvas.width() as f64, canvas.height() as f64);
        }
        set_predictions.set(None);
    };

    Effect::new(move |_| {
        setup_canvas();
    });

    view! {
        <div class="container">
            <div class="canvas-section">
                <h3>Draw Here</h3>
                <canvas
                    node_ref=canvas_ref
                    width="32"
                    height="32"
                    on:mousedown=on_mouse_down
                    on:mousemove=on_mouse_move
                    on:mouseup=on_mouse_up.clone()
                    on:mouseleave=on_mouse_up
                    on:touchstart=on_touch_start
                    on:touchmove=on_touch_move
                    on:touchend=on_touch_end.clone()
                    on:touchcancel=on_touch_end
                ></canvas>
                <div class="button-row">
                    <button on:click=clear_canvas>Clear</button>
                </div>
            </div>
                
            <div class="predictions">
                <Show
                    when=move || prediction.get().is_some()
                    fallback=|| view! { <p>Predictions will appear here!</p> }
                >
                    {move || {
                        let predicted = prediction.get().unwrap().clone();
                        view! {
                            <For
                                each=move || predicted.clone()
                                key=|state| state.clone()
                                children=move |child| {
                                    view! {
                                        <PredictionItem prediction=child/>
                                    }
                                }
                            />
                        }
                    }}
                </Show>
            </div>
        </div>
    }
}

#[component]
fn PredictionItem(prediction: Prediction) -> impl IntoView {
    let url = format!("/symbols/{}.png", prediction.symbol);
    let split = prediction.symbol.clone().split("_").map(|it| it.to_string()).collect::<Vec<String>>();
    let symbol = split.get(1).unwrap_or(&split[0]).clone();
    let package = split.get(0).map(|s| s.split("-").next().unwrap_or("").to_string()).unwrap_or_default();
    view! {
        <div class="prediction-item">
            <div style="display: flex; align-items: center; flex-direction: column;">
                <p><strong>"\\usepackage{"{package.clone()}"}"</strong></p>
                <p><strong>"\\"{symbol.clone()}</strong></p>
                <p>{format!("{:.2}%", prediction.probability)}</p>
            </div>
            {
                // Packages that should fallback to PNG
                let png_fallback_packages = ["stmaryrd", "dsfont", "textcomp", "mathdots", "wasysym", "marvosym", "gensymb", "tipa"];
                // Specific symbols that should fallback to PNG
                let png_fallback_symbols = ["textquestiondown", "textordfeminine", "dj", "copyright", "textbackslash", "textgreater", "guilsinglright", "textasciicircum"];

                let use_png = png_fallback_packages.contains(&package.as_str())
                    || png_fallback_symbols.contains(&symbol.as_str());

                if use_png {
                    view! {
                        <div>
                            <img src={url.clone()} style="width: 35px; height: 35px;" />
                        </div>
                    }.into_any()
                } else {
                    let latex_content = format!(r#"$\{symbol}$"#, symbol=symbol);
                    view! {
                        <div inner_html={format!(r#"<math-jax style="font-size: 35px;">{}</math-jax>"#, latex_content)}></div>
                    }.into_any()
                }
            }
        </div>
    }
}