fn main() {
    use build_script_cfg::Cfg;
    use search_infini_tools::{find_infini_ccl, find_infini_op, find_infini_rt};

    let cfg = Cfg::new("detected");
    if find_infini_rt().is_some() && find_infini_op().is_some() && find_infini_ccl().is_some() {
        cfg.define();
    }
}
