/// Vector dimensions of 1024
pub struct X1024;
/// Vector dimensions of 768
pub struct X768;
/// Vector dimensions of 512
pub struct X512;

impl X1024 {
    pub const OFFSET: usize = 1024;
}

impl X768 {
    pub const OFFSET: usize = 768;
}

impl X512 {
    pub const OFFSET: usize = 512;
}
