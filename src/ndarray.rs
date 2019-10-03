pub struct NDArray<T> {
    pub data: Vec<T>,
    pub dtype: &'static str,
    pub size: usize,
    pub ndim: usize,
    pub shape: Vec<usize>,
}

impl<T> NDArray<T> {
    pub fn new(data: Vec<T>, dtype: &'static str, shape: Vec<usize>) -> Self {
        Self {
            data: data,
            dtype: dtype,
            size: shape.iter().product(),
            ndim: shape.len(),
            shape: shape,
        }
    }

    pub fn reshape(&mut self, shape: Vec<usize>) -> &mut Self {
        self.shape = shape;
        self
    }
}
