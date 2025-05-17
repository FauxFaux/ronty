use anyhow::{anyhow, Context, Error, Result};
use itertools::Itertools;
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use v4l::framesize::FrameSizeEnum;
use v4l::video::Capture;
use v4l::Device;
use v4l::FourCC;

pub fn print_camera_info() -> Result<()> {
    for path in fs::read_dir("/dev")? {
        let path = match path.ok() {
            Some(path) => path,
            None => continue,
        };
        let path = match path.file_name().to_str() {
            Some(name) if name.starts_with("video") => path,
            _ => continue,
        };
        let path = path.path();
        if let Err(e) =
            print_device_info(&path).with_context(|| anyhow!("Camera device {}", path.display()))
        {
            eprintln!("Error: {e:?}");
            continue;
        }
    }

    Ok(())
}

fn print_device_info(path: impl AsRef<Path>) -> Result<()> {
    println!("Device: {}", path.as_ref().display());
    let device = Device::with_path(path)?;
    let mut fours = HashSet::with_capacity(4);
    for format in Capture::enum_formats(&device)? {
        println!(" - {:?}", format);
        fours.insert(format.fourcc.repr);
    }

    for four in fours {
        println!(
            "  - {:?}: {}",
            String::from_utf8_lossy(&four),
            Capture::enum_framesizes(&device, FourCC::new(&four))?
                .iter()
                .filter_map(|sz| match &sz.size {
                    FrameSizeEnum::Discrete(size) => Some((size.width, size.height)),
                    _ => None,
                })
                .sorted_by_key(|&(w, h)| u64::from(w) * u64::from(h))
                .map(|(w, h)| format!("{w}x{h}"))
                .join(", ")
        );
    }

    println!(
        "selected: {:?}",
        device
            .format()
            .with_context(|| anyhow!("asking for format"))?
    );
    Ok(())
}
