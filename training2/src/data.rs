use burn::data::dataloader::batcher::Batcher;
use burn::prelude::{Backend, ElementConversion, TensorData};
use burn::tensor::Int;
use burn::Tensor;
use shared::item::{DetexifyItem, HEIGHT, WIDTH};

#[derive(Clone, Default)]
pub struct DetexifyBatcher {}

#[derive(Clone, Debug)]
pub struct DetexifyBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>
}

impl<B: Backend> Batcher<B, DetexifyItem, DetexifyBatch<B>> for DetexifyBatcher {
    fn batch(&self, items: Vec<DetexifyItem>, device: &B::Device) -> DetexifyBatch<B> {
        let images = items
            .iter()
            .map(|item| TensorData::from(item.image).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 2>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, HEIGHT, WIDTH]))
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data([(item.label as i64).elem::<B::IntElem>()], device)
            }).collect();

        let images = Tensor::cat(images, 0);
        let targets = Tensor::cat(targets, 0);

        DetexifyBatch { images, targets }
    }
}