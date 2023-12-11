use std::process::Command;

use crate::{project_root, DynError};

pub fn check_formatting() -> Result<(), DynError> {
    let poetry_res = Command::new("poetry")
        .current_dir(project_root())
        .args(["install"])
        .status()?;

    if !poetry_res.success() {
        return Err("Failed to run poetry install!")?;
    }

    // Check formatting with Black
    let black_res = Command::new("poetry")
        .current_dir(project_root())
        .args(["run", "black", "--check", "."])
        .status()?;

    // Check formatting with cargo fmt
    let cargo_fmt_res = Command::new("cargo")
        .current_dir(project_root())
        .args(["fmt", "--check"])
        .status()?;

    let mut err = "".to_owned();

    if !black_res.success() {
        err.push_str("Black formatting check failed!\n");
    }
    if !cargo_fmt_res.success() {
        err.push_str("Cargo fmt run failed!\n")
    }

    if !err.is_empty() {
        return Err(err)?;
    }

    Ok(())
}
