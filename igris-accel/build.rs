use std::path::Path;
use std::process::Command;


static KERNEL_FILES: &[&str] = &[
    "dot_product",
    "cosine",
];

static FORTRAN_FOLDER: &str = "fortran/";
static BUILD_FOLDER: &str = "build/";
static LIB_NAME: &str = "libigriskernels";
static LINK_LIB_NAME: &str = "igriskernels";


fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=fortran/");
    println!("cargo:rerun-if-changed=src/");
    
    let build_folder = Path::new(BUILD_FOLDER)
        .canonicalize()
        .unwrap();
    let kernel_path = build_folder
        .join(LIB_NAME)
        .with_extension(if cfg!(windows) { "dll" } else { "a" });
    let _ = std::fs::create_dir_all(&build_folder);

    let mut combine_files = Vec::new();
    for op in KERNEL_FILES {
        let fortran_file = format!("{FORTRAN_FOLDER}{op}.f95");
        let export_file = format!("{BUILD_FOLDER}{op}.o");
                
        Command::new("gfortran")
            .arg("-O3")
            .arg("-fforce-addr")
            .arg("-ffast-math")
            .arg("-fstrength-reduce")
            .arg("-funroll-loops")
            .arg("-fcaller-saves")
            .arg("-fexpensive-optimizations")
            .arg("-mavx2")
            .arg("-mfma")
            .arg("-march=native")  // todo change
            .arg("-c")
            .arg(&fortran_file)
            .arg("-o")
            .arg(&export_file)
            .status()
            .unwrap_or_else(|_| panic!("Compile file: {fortran_file}"));
        
        combine_files.push(export_file);
    }
    
    // We dynamically link on windows due to it being somewhat of a pain to
    // build a DLL.
    if cfg!(windows) {
        Command::new("gfortran")
            .arg("-shared")
            .arg("-o")
            .args(&kernel_path);
    } else {
        Command::new("ar")
            .arg("r")
            .arg(&kernel_path)
            .args(combine_files)
            .status()
            .expect("Combine kernel files");
    }
        
    println!("cargo:rustc-link-search={}", build_folder.display());
    println!("cargo:rustc-link-lib=static={LINK_LIB_NAME}");
}

