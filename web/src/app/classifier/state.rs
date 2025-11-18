use burn::backend::NdArray;
use burn::{
    module::Module,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
};
use shared::model::{Model, ModelConfig};

static STATE_ENCODED: &[u8] = include_bytes!("../../../../over90top5/model.bin");

pub type MyB = NdArray<f32, i32>;


/// Builds and loads trained parameters into the model.
pub async fn build_and_load_model() -> Model<MyB> {
    let model: Model<MyB> = ModelConfig::new(1098, 512)
        .init(&Default::default());
    let record = BinBytesRecorder::<FullPrecisionSettings, &'static [u8]>::default()
        .load(STATE_ENCODED, &Default::default())
        .expect("Failed to decode state");

    model.load_record(record)
}