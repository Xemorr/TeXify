use crate::data::DetexifyBatcher;
use crate::training::TrainingConfig;
use burn::config::Config;
use burn::data::dataloader::batcher::Batcher;
use burn::module::Module;
use burn::prelude::Backend;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use shared::item::DetexifyItem;

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: DetexifyItem) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");
    let record = BinFileRecorder::<FullPrecisionSettings>::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist; run train first");

    let model = config.model.init::<B>(&device).load_record(record);

    let label = item.label;
    let batcher = DetexifyBatcher::default();
    let batch = batcher.batch(vec![item], &device);
    let output = model.forward(batch.images);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("Predicted: {}, Actual: {}", predicted, label);
}