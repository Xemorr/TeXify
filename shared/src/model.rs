use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d, Relu};
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig};
use burn::prelude::Backend;
use burn::Tensor;
use crate::basicblock::BasicBlock;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    basicblock1: BasicBlock<B>,
    basicblock2: BasicBlock<B>,
    basicblock3: BasicBlock<B>,
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();

        let x = images.reshape([batch_size, 1, height, width]);

        let x = self.basicblock1.forward(x);
        let x = self.basicblock2.forward(x);
        let x = self.basicblock3.forward(x);

        let x = self.pool.forward(x);
        let x = x.reshape([batch_size, 128 * 8 * 8]);
        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        self.linear2.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
    hidden_size: usize,
    #[config(default = "0.3")]
    dropout: f64
}


impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            basicblock1: BasicBlock::new(1, 32, 1, device),
            basicblock2: BasicBlock::new(32, 64, 1, device),
            basicblock3: BasicBlock::new(64, 128, 1, device),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: Relu::new(),
            linear1: LinearConfig::new(128 * 8 * 8, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            dropout: DropoutConfig::new(self.dropout).init()
        }
    }
}