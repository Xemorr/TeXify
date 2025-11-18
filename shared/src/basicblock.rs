use burn::nn::{conv::{Conv2d, Conv2dConfig}, BatchNorm, BatchNormConfig, PaddingConfig2d, Relu};
use burn::prelude::*;
use burn::module::Module;

#[derive(Module, Debug)]
pub struct BasicBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,

    // Optional projection for skip path
    projection: Option<(Conv2d<B>, BatchNorm<B>)>,

    activation: Relu,
}

impl<B: Backend> BasicBlock<B> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        device: &B::Device,
    ) -> Self {

        let conv1 = Conv2dConfig::new([in_channels, out_channels], [3, 3])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        let bn1 = BatchNormConfig::new(out_channels).init(device);

        let conv2 = Conv2dConfig::new([out_channels, out_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        let bn2 = BatchNormConfig::new(out_channels).init(device);

        // If dimensions change, create a projection skip connection
        let projection = if in_channels != out_channels || stride != 1 {
            let proj_conv = Conv2dConfig::new([in_channels, out_channels], [1, 1])
                .with_stride([stride, stride])
                .with_padding(PaddingConfig2d::Valid)
                .init(device);

            let proj_bn = BatchNormConfig::new(out_channels).init(device);

            Some((proj_conv, proj_bn))
        } else {
            None
        };

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            projection,
            activation: Relu::new(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let identity = match &self.projection {
            Some((proj_conv, proj_bn)) => {
                proj_bn.forward(proj_conv.forward(x.clone()))
            }
            None => x.clone(),
        };

        let out = self.conv1.forward(x);
        let out = self.bn1.forward(out);
        let out = self.activation.forward(out);

        let out = self.conv2.forward(out);
        let out = self.bn2.forward(out);

        // Residual addition
        let out = out + identity;

        self.activation.forward(out)
    }
}