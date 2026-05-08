//! CI drift check for `dist/softfloat64.metal`.
//!
//! Runs the gen-msl-header binary in --check mode. Fails if the
//! checked-in dist file doesn't match what the generator produces from
//! the current `shaders/softfloat.metal`.

#[test]
fn dist_softfloat64_metal_is_up_to_date() {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let status = std::process::Command::new(env!("CARGO"))
        .arg("run")
        .arg("--quiet")
        .arg("--bin")
        .arg("gen-msl-header")
        .arg("--")
        .arg("--check")
        .current_dir(&manifest)
        .status()
        .expect("spawn gen-msl-header");
    assert!(
        status.success(),
        "dist/softfloat64.metal is stale — re-run `cargo run --bin gen-msl-header`"
    );
}
