use burn::data::dataset::{Dataset, InMemDataset};
use shared::item::DetexifyItem;

pub struct DetexifyDataset {
    pub dataset: InMemDataset<DetexifyItem>
}

impl DetexifyDataset {
    pub fn split(self, split_ratio: f64) -> (Self, Self) {
        let len = self.dataset.len();
        let split_index = (len as f64 * split_ratio) as usize;

        let items: Vec<DetexifyItem> = (0..len)
            .map(|i| self.dataset.get(i).unwrap())
            .collect();

        let train_items = items[..split_index].to_vec();
        let eval_items = items[split_index..].to_vec();

        (
            DetexifyDataset { dataset: InMemDataset::new(train_items) },
            DetexifyDataset { dataset: InMemDataset::new(eval_items) }
        )
    }
}