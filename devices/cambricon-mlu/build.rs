use std::{env,process};
use std::path::PathBuf;

fn main() {
    use build_script_cfg::Cfg;
    use search_neuware_tools::find_neuware_home;

    let neuware = Cfg::new("detected_neuware");
    if find_neuware_home().is_some() {
        neuware.define();
    }

    let infini_root = env::var("INFINI_ROOT").unwrap_or_else(|_| {
        eprintln!("INFINI_ROOT environment variable is not set.");
        process::exit(1)
    });

    // Tell cargo to tell rustc to link the shared library.
    println!("cargo:rustc-link-search=native={}", infini_root);

    // Link the operators library without including the prefix lib and the suffix .so
    println!("cargo:rustc-link-lib=dylib=operators");
    // Link the cnnl_extra library
    println!("cargo:rustc-link-lib=dylib=cnnl_extra");
    // Link the OpenMP library
    println!("cargo:rustc-link-lib=dylib=gomp");    

    // The bindgen::Builder is the main entry point to bindgen,
    // and lets you build up options for the resulting bindings.
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}/include", infini_root))
        // Generate rust style enums.
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: true,
        })     
        // Tell cargo to invalidate the built crate whenever the wrapper changes
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Disable layout tests because bitfields might cause issues
        .layout_tests(false)
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");      
}
