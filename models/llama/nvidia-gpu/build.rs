fn main() {
    use build_script_cfg::Cfg;
    use search_cuda_tools::{find_cuda_root, find_nccl_root};

    let cfg = Cfg::new("hw_detected");
    let nccl = Cfg::new("nccl_detected");
    if find_cuda_root().is_some() {
        cfg.define();
        if find_nccl_root().is_some() {
            nccl.define();
        }
    }
}
