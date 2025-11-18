use crate::data::{DetexifyBatch, DetexifyBatcher};
use crate::dataset::DetexifyDataset;
use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, AdamWConfig};
use burn::prelude::Backend;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Int;
use burn::train::metric::{AccuracyMetric, LossMetric, TopKAccuracyMetric};
use burn::train::{ClassificationOutput, LearnerBuilder, LearningStrategy, TrainOutput, TrainStep, ValidStep};
use burn::Tensor;
use shared::model::{Model, ModelConfig};

pub trait ForwardClassification<B: Backend> {
    fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>
    ) -> ClassificationOutput<B>;
}

impl<B: Backend> ForwardClassification<B> for Model<B> {
    fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .with_smoothing(Some(0.1))
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<DetexifyBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: DetexifyBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<DetexifyBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: DetexifyBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamWConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64
}

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).unwrap();
    std::fs::create_dir_all(artifact_dir).unwrap();
}


pub fn train<Backend: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: Backend::Device, dataset_train: DetexifyDataset, dataset_eval: DetexifyDataset) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully!");

    Backend::seed(&device, config.seed);

    let batcher = DetexifyBatcher::default();

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_train.dataset);

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_eval.dataset);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(TopKAccuracyMetric::new(5))
        .metric_valid_numeric(TopKAccuracyMetric::new(5))
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .learning_strategy(LearningStrategy::SingleDevice(device.clone()))
        .num_epochs(config.num_epochs)
        .build(
            config.model.init::<Backend>(&device),
            config.optimizer.init(),
            config.learning_rate
        );

    let result = learner.fit(dataloader_train, dataloader_test);

    result.model.save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully!");
}