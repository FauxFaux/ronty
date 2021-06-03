#[derive(Copy, Clone, Default)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Rect {
    pub(crate) fn centre(&self) -> (u32, u32) {
        (
            self.x + self.w / 2,
            self.y + self.h / 2,
        )
    }
}
