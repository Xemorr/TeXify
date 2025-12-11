use burn::backend::{Wgpu};
use burn::module::Module;
use burn::record::{FullPrecisionSettings, Recorder};
use shared::model::{Model, ModelConfig};

fn main() {
    // Load the model from .mpk file
    let model_path = "over90top5/model.mpk";

    // Initialize the model with the correct config
    let config = ModelConfig::new(1098, 512);
    let device = Default::default();

    // Load the record
    let record = burn::record::NamedMpkFileRecorder::<FullPrecisionSettings>::new()
        .load(model_path.into(), &device)
        .expect("Failed to load model");

    // Initialize model with record
    let model: Model<Wgpu> = config.init(&device).load_record(record);

    // Save as .bin file
    let bin_path = "over90top5/model.bin";
    burn::record::BinFileRecorder::<FullPrecisionSettings>::new()
        .record(model.into_record(), bin_path.into())
        .expect("Failed to save model as .bin");

    println!("Model converted successfully from .mpk to .bin");
}