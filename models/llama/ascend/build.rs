fn main() {
    use build_script_cfg::Cfg;
    use search_ascend_tools::find_ascend_toolkit_home;

    let cfg = Cfg::new("hw_detected");
    if find_ascend_toolkit_home().is_some() {
        cfg.define();
    }
}
