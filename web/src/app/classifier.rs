use burn::backend::wgpu::WgpuDevice;
use burn::backend::{NdArray, Wgpu};
use burn::record::{BinBytesRecorder, FullPrecisionSettings};
use leptos::prelude::ElementChild;
use leptos::prelude::{signal, Effect, Get, NodeRef, NodeRefAttribute, OnAttribute, Set, Show, StyleAttribute};
use leptos::{component, view, IntoView};
use shared::item::{HEIGHT, WIDTH};
use shared::model::{Model, ModelConfig};
use std::cell::RefCell;
use std::rc::Rc;
use burn::backend::ndarray::NdArrayDevice;
use crate::app::classifier::state::{build_and_load_model, MyB};

mod keys;
mod state;
pub struct SharedModel {
    model: Option<Model<MyB>>,
    device: NdArrayDevice,
}

impl SharedModel {
    pub fn new() -> Self {
        Self {
            model: None,
            device: NdArrayDevice::default(),
        }
    }

    pub async fn inference(&mut self, image: [[f32; WIDTH]; HEIGHT]) -> Result<String, String> {
        use burn::prelude::*;
        use burn::record::Recorder;

        // Lazy-load the model
        if self.model.is_none() {
            self.model = Some(build_and_load_model().await);
        }

        let model = self.model.as_ref().unwrap();

        // Flatten the 32x32 image into 1D slice
        let image_flat: Vec<f32> = image.iter().flat_map(|row| row.iter().copied()).collect();

        // Create tensor and reshape to [batch, height, width]
        let tensor = Tensor::<MyB, 3>::from_floats(image_flat.as_slice(), &self.device)
            .reshape([1, HEIGHT, WIDTH]);

        // Run forward pass
        let output: Tensor<MyB, 2> = model.forward(tensor);

        // Get argmax asynchronously (the predicted class index)
        let predicted_idx: usize = output
            .argmax(1)
            .flatten::<1>(0, 1)
            .into_scalar_async()
            .await as usize;

        Ok(keys::KEYS[predicted_idx].to_string())
    }
}

#[component]
pub fn Classifier() -> impl IntoView {
    use leptos::wasm_bindgen::JsCast;
    use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

    let model = Rc::new(RefCell::new(SharedModel::new()));
    let canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    let (drawing, set_drawing) = signal(false);
    let (predicted_symbol, set_predicted_symbol) = signal(None::<String>);

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
            let x = e.client_x() as f64 - rect.left();
            let y = e.client_y() as f64 - rect.top();

            ctx.begin_path();
            ctx.move_to(x, y);
        }
    };

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
                let x = e.client_x() as f64 - rect.left();
                let y = e.client_y() as f64 - rect.top();

                ctx.line_to(x, y);
                ctx.stroke();
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
        set_predicted_symbol.set(None);
    };

    let model_clone = Rc::clone(&model);
    let classify = move |_| {
        if let Some(canvas) = canvas_ref.get() {
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

            // Preprocess: resize to 32x32 and convert to grayscale
            let canvas_width = canvas.width() as usize;
            let canvas_height = canvas.height() as usize;

            // Create a temporary ImageBuffer for resizing
            let img_buffer = image::ImageBuffer::from_fn(canvas_width as u32, canvas_height as u32, |x, y| {
                let idx = (y as usize * canvas_width + x as usize) * 4;
                // Invert: white background (255) -> 0, black drawing (0) -> 255
                let gray_value = 255 - data[idx];
                image::Luma([gray_value])
            });

            // Resize to 32x32
            let resized = image::imageops::resize(&img_buffer, 32, 32, image::imageops::FilterType::Lanczos3);

            // Convert to [[f32; 32]; 32] and normalize
            let mut image_array = [[0.0f32; WIDTH]; HEIGHT];
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    let pixel = resized.get_pixel(x as u32, y as u32);
                    // Normalize to 0.0-1.0
                    image_array[y][x] = pixel[0] as f32 / 255.0;
                }
            }

            leptos::logging::log!("Image preprocessed to 32x32");

            // Run inference
            let model_inner = Rc::clone(&model_clone);
            wasm_bindgen_futures::spawn_local(async move {
                match model_inner.borrow_mut().inference(image_array).await {
                    Ok(symbol) => {
                        leptos::logging::log!("Predicted: {}", symbol);
                        set_predicted_symbol.set(Some(symbol));
                    }
                    Err(e) => {
                        leptos::logging::error!("Inference error: {}", e);
                        set_predicted_symbol.set(Some(format!("Error: {}", e)));
                    }
                }
            });
        }
    };

    Effect::new(move |_| {
        setup_canvas();
    });

    view! {
        <div style="display: flex; flex-direction: column; align-items: center; gap: 1rem;">
            <canvas
                node_ref=canvas_ref
                width="300"
                height="300"
                style="border: 2px solid #333; cursor: crosshair; background: white;"
                on:mousedown=on_mouse_down
                on:mousemove=on_mouse_move
                on:mouseup=on_mouse_up
                on:mouseleave=on_mouse_up
            />
            <div style="display: flex; gap: 1rem;">
                <button on:click=clear_canvas>"Clear"</button>
                <button on:click=classify>"Classify"</button>
            </div>
            <Show
                when=move || predicted_symbol.get().is_some()
                fallback=|| view! { <p>"Draw a LaTeX symbol and click Classify"</p> }
            >
                {move || {
                    let symbol = predicted_symbol.get().unwrap();
                    view! {
                        <p style="font-size: 1.5rem; font-weight: bold;">
                            "Predicted: " {symbol}
                        </p>
                    }
                }}
            </Show>
        </div>
    }
}