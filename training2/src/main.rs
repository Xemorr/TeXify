#![recursion_limit = "256"]
mod data;
mod training;
mod infer;
mod dataset;

use crate::training::TrainingConfig;
use burn::backend::{Autodiff, Vulkan};
use burn::data::dataset::InMemDataset;
use burn::optim::AdamWConfig;
use sqlx::postgres::PgPoolOptions;
use sqlx::Error;
use std::collections::{HashMap, HashSet};
use std::ops::Add;
use shared::image_processing::{rasterize_strokes, save_image};
use shared::item::{DetexifyItem, HEIGHT, WIDTH};
use shared::model::ModelConfig;
use crate::dataset::DetexifyDataset;

use std::sync::Arc;

use wordchipper::{
    concurrency::rayon::{ParallelRayonDecoder, ParallelRayonEncoder},
    decoders::TokenDictDecoder,
    encoders::DefaultTokenEncoder,
    pretrained::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN,
    training::{BinaryPairVocabTrainer, BinaryPairVocabTrainerOptions},
    vocab::{ByteMapVocab, UnifiedTokenVocab, io::save_tiktoken_vocab_path},
};

fn example<I, S>(
    vocab_size: usize,
    batches: I,
    tiktoken_save_path: Option<String>,
) where
    I: IntoIterator,
    I::Item: AsRef<[S]>,
    S: AsRef<str>,
{
    // We can pick any unsigned integer type > vocab_size;
    // See [`wordchipper::types::TokenType`].
    type T = u32;
    type K = String;
    type C = u64;

    let options = BinaryPairVocabTrainerOptions::new(OA_GPT3_CL100K_WORD_PATTERN, vocab_size);

    let mut trainer: BinaryPairVocabTrainer<K, C> = options.init();

    for batch in batches {
        // The trainer has no parallelism.
        // The perceived benefits of parallelism in the trainer
        // are insignificant if the IO for the sample source is
        // fed by another thread.
        trainer.update_from_samples(batch.as_ref());
    }

    let byte_vocab: ByteMapVocab<T> = Default::default();

    let vocab: UnifiedTokenVocab<T> =
        trainer.train(byte_vocab.clone()).expect("training failed");

    if let Some(path) = tiktoken_save_path {
        save_tiktoken_vocab_path(&vocab.span_vocab().span_map(), &path)
            .expect("failed to save tiktoken vocab");
        println!("- tiktoken vocab: {path:?}");
    }

    let encoder: DefaultTokenEncoder<T> = DefaultTokenEncoder::new(vocab.clone(), None);
    let encoder = ParallelRayonEncoder::new(Arc::new(encoder));

    let decoder = TokenDictDecoder::from_unified_vocab(vocab.clone());
    let decoder = ParallelRayonDecoder::new(Arc::new(decoder));
}

#[tokio::main]
async fn main() -> Result<(), Error> {


    let pretty = false;
    tokenizer
        .train_from_files(
            &mut trainer,
            vec!["path/to/vocab.txt".to_string()],
        )?
        .save("tokenizer.json", pretty)?;

    let key_to_value: HashMap<&String, usize> = distinct_keys
        .iter()
        .enumerate()
        .map(|(i, key)| (key, i))
        .collect();

    let items: Vec<DetexifyItem> = rows.iter()
        .map(|row| {
            let key = row.key.as_ref().expect("Key should be present for all rows");
            let strokes: Vec<Vec<[f32; 3]>> = row.strokes.as_ref()
                .expect("Strokes should be present for all rows")
                .as_array().expect("Strokes should be an array")
                .iter()
                .map(|stroke| {
                    stroke.as_array().expect("Stroke should be an array")
                        .iter()
                        .map(|point| {
                            let p = point.as_array().expect("Point should be an array");
                            [
                                p[0].as_f64().expect("x must be number") as f32,
                                p[1].as_f64().expect("y must be number") as f32,
                                p[2].as_f64().expect("t must be number") as f32,
                            ]
                        })
                        .collect()
                }).collect();
            DetexifyItem {
                image: rasterize_strokes(&strokes),
                label: *key_to_value.get(&key).unwrap() as u32
            }
        }).collect();

    let label_to_find = *key_to_value.get(&&"latex2e-OT1-_sigma".to_string()).unwrap() as u32;
    let sigma = items.iter().find(|item| item.label == label_to_find).unwrap();
    save_image(&sigma.image, distinct_keys[sigma.label as usize].clone());

    let dataset = DetexifyDataset { dataset: InMemDataset::new(items.clone()) };
    let (dataset_train, dataset_eval) = dataset.split(0.8);

    type MyBackend = Vulkan;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = Default::default();
    let model = ModelConfig::new(number_of_classes as usize, 256).init::<MyBackend>(&device);
    let artifact_dir = "./models";

    training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(
            ModelConfig::new(number_of_classes as usize, 256),
            AdamWConfig::new().with_cautious_weight_decay(true)
        ).with_num_epochs(1),
        device.clone(),
        dataset_train,
        dataset_eval
    );

    println!("{model}");

    infer::infer::<MyBackend>(artifact_dir, device, sigma.clone());

    Ok(())
}
