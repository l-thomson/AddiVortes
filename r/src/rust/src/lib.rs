use extendr_api::prelude::*;

/// Version of the vendored core crate.
#[extendr]
fn core_version() -> &'static str {
    thiessen::VERSION
}

extendr_module! {
    mod thiessen;
    fn core_version;
}
