fn main() {
    use build_script_cfg::Cfg;
    use search_infini_tools::find_infini_rt;

    let cfg = Cfg::new("detected");
    if find_infini_rt().is_some() {
        cfg.define();
    }
}
