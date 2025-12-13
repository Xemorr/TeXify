use std::hash::{Hash, Hasher};
use std::iter::zip;
use burn::backend::ndarray::NdArrayDevice;
use burn::tensor::activation::softmax;
use shared::item::{HEIGHT, WIDTH};
use shared::model::Model;
use crate::app::classifier::keys;
use crate::app::classifier::state::{build_and_load_model, MyB};

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
        let output: Tensor<MyB, 1> = model.forward(tensor).squeeze();
        let probabilities = softmax(output.clone(), 0);

        let topk = probabilities
            .topk_with_indices(5, 0);

        let predicted_idx = topk.1
            .to_data_async()
            .await
            .to_vec::<i32>()
            .unwrap();
        let predicted_values = (topk.0 * 100)
            .to_data_async()
            .await
            .to_vec::<f32>().unwrap();

        let predictions: Vec<Prediction> = zip(predicted_idx, predicted_values)
            .map(|(idx, value)| Prediction { symbol: keys::KEYS[idx as usize].to_string(), probability: value })
            .collect::<Vec<_>>();

        predictions
    }
}