pub(crate) struct Window {
    pub(crate) window: Vec<(usize, usize)>,
    pub(crate) len_x: usize,
    pub(crate) len_y: usize,
}

impl Window {
    pub(crate) fn new(len_x: usize, len_y: usize) -> Self {
        Self {
            window: vec![(usize::MAX, usize::MIN); len_x],
            len_x,
            len_y,
        }
    }

    pub(crate) fn mark_visited(&mut self, x: usize, y: usize) {
        let (left, right) = self.window[x];
        self.window[x] = (left.min(y), right.max(y + 1).min(self.len_y));
    }
}
