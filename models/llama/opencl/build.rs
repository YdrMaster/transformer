fn main() {
    use build_script_cfg::Cfg;
    use search_cl_tools::find_opencl;

    let cfg = Cfg::new("hw_detected");
    if find_opencl().is_some() {
        cfg.define();
    }
}
