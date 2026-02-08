use burn::config::Config;
use burn::module::Module;
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::transformer::{TransformerDecoder, TransformerDecoderConfig, TransformerDecoderInput};
use burn::nn::{conv::Conv2d, conv::Conv2dConfig, Embedding, EmbeddingConfig, Gelu, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

#[derive(Module, Debug)]
pub struct LatexOCR<B: Backend> {
    cnn: CnnEncoder<B>,
    transformer: TransformerDecoder<B>,
    embedding: Embedding<B>,
    output_linear: Linear<B>,
    feat_projection: Linear<B>, // Projects CNN features to Transformer dimension
}

#[derive(Config)]
#[derive(Debug)]
pub struct LatexOCRConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub image_channels: usize,
}

impl LatexOCRConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LatexOCR<B> {
        let cnn = CnnEncoder::new(self.image_channels, self.d_model, device);

        let transformer = TransformerDecoderConfig::new(
            self.d_model,
            self.d_model * 4,
            self.n_heads,
            self.n_layers,
        )
            .with_dropout(0.1)
            .with_norm_first(true)
            .with_quiet_softmax(true)
            .init(device);

        let embedding = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);
        let output_linear = LinearConfig::new(self.d_model, self.vocab_size).init(device);
        let feat_projection = LinearConfig::new(self.d_model, self.d_model).init(device);

        LatexOCR {
            cnn,
            transformer,
            embedding,
            output_linear,
            feat_projection,
        }
    }
}

impl<B: Backend> LatexOCR<B> {
    pub fn forward(&self, images: Tensor<B, 4>, targets: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        // 1. Extract Image Features [Batch, Channels, H, W] -> [Batch, SeqLen, d_model]
        let features = self.cnn.forward(images);
        let [batch, channels, h, w] = features.dims();
        let features = features
            .reshape([batch, channels, h * w])
            .swap_dims(1, 2); // [Batch, H*W, Channels]

        let memory = self.feat_projection.forward(features);

        // 2. Embed Target Tokens (for Teacher Forcing during training)
        let target_embeddings = self.embedding.forward(targets);

        // 3. Decode
        let decoded = self.transformer.forward(TransformerDecoderInput::new(target_embeddings, memory));

        // 4. Project to Vocabulary
        self.output_linear.forward(decoded)
    }
}

// Simple CNN Backbone Helper
#[derive(Module, Debug)]
pub struct CnnEncoder<B: Backend> {
    conv1: Conv2d<B>,
    max_pool2d: MaxPool2d,
    conv2: Conv2d<B>,
    gelu: Gelu
}

impl<B: Backend> CnnEncoder<B> {
    pub fn new(in_channels: usize, out_channels: usize, device: &B::Device) -> Self {
        let conv1 = Conv2dConfig::new([in_channels, out_channels / 2], [3, 3]).init(device);
        let max_pool2d = MaxPool2dConfig::new([2, 2]).init();
        let conv2 = Conv2dConfig::new([out_channels / 2, out_channels / 4], [3, 3]).init(device);
        let gelu = Gelu::new();
        Self { conv1, max_pool2d, conv2, gelu }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv1.forward(x);
        let x = self.gelu.forward(x);
        let x = self.conv2.forward(x);
        self.gelu.forward(x)
    }
}