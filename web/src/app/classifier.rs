use crate::app::classifier::state::{build_and_load_model, MyB};
use burn::backend::ndarray::NdArrayDevice;
use leptos::prelude::{ClassAttribute, ElementChild};
use leptos::prelude::{signal, Effect, Get, NodeRef, NodeRefAttribute, OnAttribute, Set, Show, StyleAttribute};
use leptos::{component, view, IntoView};
use shared::item::{HEIGHT, WIDTH};
use shared::model::Model;
use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::iter::zip;
use std::rc::Rc;
use leptos::control_flow::For;

mod keys;
mod state;
pub struct SharedModel {
    model: Option<Model<MyB>>,
    device: NdArrayDevice,
}

#[derive(Clone)]
pub struct Prediction {
    pub symbol: String,
    pub probability: f32
}

impl PartialEq for Prediction {
    fn eq(&self, other: &Self) -> bool {
        self.symbol == other.symbol
    }
}

impl Eq for Prediction {}

impl Hash for Prediction {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.symbol.hash(state);
    }
}

impl SharedModel {
    pub fn new() -> Self {
        Self {
            model: None,
            device: NdArrayDevice::default(),
        }
    }

    pub async fn inference(&mut self, image: [[f32; WIDTH]; HEIGHT]) -> Vec<Prediction> {
        use burn::prelude::*;
        use burn::record::Recorder;

        // Lazy-load the model
        if self.model.is_none() {
            self.model = Some(build_and_load_model().await);
        }

        let model = self.model.as_ref().unwrap();

        // Create tensor and reshape to [batch, height, width]
        let tensor = Tensor::<MyB, 2>::from_floats(image, &self.device)
            .unsqueeze();

        // Run forward pass
        let output: Tensor<MyB, 2> = model.forward(tensor);

        // Get argmax asynchronously (the predicted class index)
        let topk = output
            .topk_with_indices(5, 1);
        let predicted_idx = topk.1
            .to_data_async()
            .await
            .to_vec::<i32>()
            .unwrap();
        let predicted_values = topk.0
            .to_data_async()
            .await
            .to_vec::<f32>().unwrap();

        let predictions: Vec<Prediction> = zip(predicted_idx, predicted_values)
            .map(|(idx, value)| Prediction { symbol: keys::KEYS[idx as usize].to_string(), probability: value })
            .collect::<Vec<_>>();

        predictions
    }
}

#[component]
pub fn Classifier() -> impl IntoView {
    use leptos::wasm_bindgen::JsCast;
    use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

    let model = Rc::new(RefCell::new(SharedModel::new()));
    let canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    let processed_canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    let (drawing, set_drawing) = signal(false);
    let (prediction, set_predictions) = signal(None::<Vec<Prediction>>);

    fn get_logical_coords(e: &web_sys::MouseEvent, canvas: &HtmlCanvasElement) -> (f64, f64) {
        let rect = canvas.get_bounding_client_rect();
        let scale_x = canvas.width() as f64 / rect.width();
        let scale_y = canvas.height() as f64 / rect.height();

        let x = (e.client_x() as f64 - rect.left()) * scale_x;
        let y = (e.client_y() as f64 - rect.top()) * scale_y;

        (x, y)
    }

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

            ctx.set_line_width(3.0);
            ctx.set_line_cap("round");
            ctx.set_stroke_style_str("black");
        }
    };

    let perform_classification = Rc::new(move || {
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

            ctx.set_line_width(1.5);
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

                ctx.set_line_width(1.5);
                ctx.line_to(x, y);
                ctx.stroke();

                // Perform classification after drawing
                // perform_classification_clone();
            }
        }
    };

    let on_mouse_up = move |_: web_sys::MouseEvent| {
        set_drawing.set(false);
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

    let classify = move |_| {
        perform_classification();
    };

    Effect::new(move |_| {
        setup_canvas();
    });

    view! {
        <div class="classifier-container">
            <div class="canvas-section">
                <h3>Draw Here</h3>
                <canvas
                    node_ref=canvas_ref
                    width="32"
                    height="32"
                    on:mousedown=on_mouse_down
                    on:mousemove=on_mouse_move
                    on:mouseup=on_mouse_up
                    on:mouseleave=on_mouse_up
                ></canvas>
                <div class="button-row">
                    <button on:click=clear_canvas>Clear</button>
                    <button on:click=classify>Classify</button>
                </div>
                <p>Draw a LaTeX symbol and click Classify</p>
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
                                    let url = format!("/symbols/{}.png", child.symbol);
                                    let symbol = child.symbol;
                                    view! {
                                        <div class="prediction-item">
                                            <p><strong>{symbol}</strong></p>
                                            <p>{format!("{:.2}%", child.probability * 100.0)}</p>
                                            <img src={url} />
                                        </div>
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