pub const WIDTH: usize = 32;
pub const HEIGHT: usize = 32;

#[derive(Debug, Clone)]
pub struct DetexifyItem {
    pub image: [[f32; WIDTH]; HEIGHT],
    pub label: u32
}