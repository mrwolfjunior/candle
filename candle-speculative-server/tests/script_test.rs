use std::fs;
use std::process::Command;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

#[test]
fn test_deployment_script_syntax() {
    let script_path = if std::path::Path::new("run-speculative-server.sh").exists() {
        std::path::PathBuf::from("run-speculative-server.sh")
    } else {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("run-speculative-server.sh")
    };

    let output = Command::new("bash")
        .arg("-n")
        .arg(&script_path)
        .output()
        .expect("Failed to execute bash syntax check");

    assert!(
        output.status.success(),
        "run-speculative-server.sh has bash syntax errors: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_deployment_script_permissions() {
    let script_path = if std::path::Path::new("run-speculative-server.sh").exists() {
        std::path::PathBuf::from("run-speculative-server.sh")
    } else {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("run-speculative-server.sh")
    };

    let metadata = fs::metadata(&script_path).expect("run-speculative-server.sh must exist");

    #[cfg(unix)]
    {
        let permissions = metadata.permissions();
        assert!(
            permissions.mode() & 0o111 != 0,
            "run-speculative-server.sh must be executable, got mode: {:o}",
            permissions.mode()
        );
    }
}
