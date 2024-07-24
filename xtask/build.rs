use std::{env, process};

fn main() {
    use build_script_cfg::Cfg;
    use search_cuda_tools::{find_cuda_root, find_nccl_root};
    use search_neuware_tools::find_neuware_home;

    let cuda = Cfg::new("detected_cuda");
    let nccl = Cfg::new("detected_nccl");
    if cfg!(feature = "nvidia") && find_cuda_root().is_some() {
        cuda.define();
        if find_nccl_root().is_some() {
            nccl.define();
        }
    }

    let neuware = Cfg::new("detected_neuware");
    if cfg!(feature = "cambricon") && find_neuware_home().is_some() {
        neuware.define();
    }

    let infini_root = env::var("INFINI_ROOT").unwrap_or_else(|_| {
        eprintln!("INFINI_ROOT environment variable is not set.");
        process::exit(1)
    });
    
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", infini_root);  
}
