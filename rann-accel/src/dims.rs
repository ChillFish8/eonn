#[derive(Debug, Copy, Clone, Default)]
/// Vector dimensions of 1024
pub struct X1024;
#[derive(Debug, Copy, Clone, Default)]
/// Vector dimensions of 768
pub struct X768;
#[derive(Debug, Copy, Clone, Default)]
/// Vector dimensions of 512
pub struct X512;
#[derive(Debug, Copy, Clone, Default)]
/// Vector dimensions of any size.
pub struct XAny;

/// Dimension specification information.
pub trait Dim: Default {
    /// The size of the dim.
    ///
    /// This is used for validation, not compute.
    fn const_size() -> Option<usize>;
}

impl Dim for X1024 {
    fn const_size() -> Option<usize> {
        Some(1024)
    }
}

impl Dim for X768 {
    fn const_size() -> Option<usize> {
        Some(768)
    }
}

impl Dim for X512 {
    fn const_size() -> Option<usize> {
        Some(512)
    }
}

impl Dim for XAny {
    fn const_size() -> Option<usize> {
        None
    }
}
