#[derive(Debug, Copy, Clone, Default)]
/// Vector dimensions of 1024
pub struct X1024;
#[derive(Debug, Copy, Clone, Default)]
/// Vector dimensions of 768
pub struct X768;
#[derive(Debug, Copy, Clone, Default)]
/// Vector dimensions of 512
pub struct X512;

/// Dimension specification information.
pub trait Dim: Default {
    /// The size of the dim.
    ///
    /// This is used for validation, not compute.
    fn size() -> usize;
}

impl Dim for X1024 {
    fn size() -> usize {
        1024
    }
}

impl Dim for X768 {
    fn size() -> usize {
        768
    }
}

impl Dim for X512 {
    fn size() -> usize {
        512
    }
}
