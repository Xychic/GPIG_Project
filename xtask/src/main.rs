use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

use ci::*;

mod ci;

type DynError = Box<dyn std::error::Error>;

fn main() {
    if let Err(e) = try_main() {
        eprintln!("\n\nSome errors occured in the xtask:\n\n{}", e);
        std::process::exit(1)
    }
}

fn try_main() -> Result<(), DynError> {
    match env::args().nth(1).as_deref() {
        Some("dev") => start_dev()?,
        Some("ci") => ci()?,
        _ => print_help(),
    }

    Ok(())
}

fn print_help() {
    eprintln!("Available commands: dev ci")
}

fn start_dev() -> Result<(), DynError> {
    let res = Command::new("docker-compose").args(["up", "-d"]).status()?;

    if !res.success() {
        return Err("Failed to start Docker Compose")?;
    }

    Ok(())
}

fn ci() -> Result<(), DynError> {
    let build_res = Command::new("cargo")
        .current_dir(project_root())
        .args(["build", "--release", "-p", "node_lookup"])
        .status()?;

    let clippy_res = Command::new("cargo")
        .current_dir(project_root())
        .args(["clippy", "-p", "node_lookup", "--", "-Dwarnings"])
        .status()?;

    let mut err = "".to_owned();

    if !build_res.success() {
        err.push_str("Cargo build failed.\n");
    }

    if !clippy_res.success() {
        eprintln!("Clippy run failed");
        err.push_str("Clippy run failed.\n");
    }

    if let Err(formatting_error) = check_formatting() {
        err.push_str(formatting_error.to_string().as_str());
    }

    if !err.is_empty() {
        return Err(err)?;
    }
    Ok(())
}

fn project_root() -> PathBuf {
    Path::new(&env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(1)
        .unwrap()
        .to_path_buf()
}
