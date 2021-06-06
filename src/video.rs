use std::io;
use std::sync::Arc;
use std::thread;
use std::thread::JoinHandle;

use anyhow::Context;
use anyhow::Result;
use image::codecs::jpeg::JpegDecoder;
use image::ImageDecoder;
use image::RgbImage;
use v4l::buffer::Type;
use v4l::io::traits::CaptureStream;
use v4l::io::mmap::Stream as MmapStream;
use v4l::video::Capture;
use v4l::Device;
use v4l::FourCC;

use crate::latest::Latest;

pub fn make_frames() -> (Arc<Latest<RgbImage>>, JoinHandle<Result<()>>) {
    let ret = Arc::new(Latest::default());
    let frames = Arc::clone(&ret);
    let thread = thread::spawn(move || -> anyhow::Result<()> {
        let mut dev = Device::new(0).with_context(|| "Failed to open device")?;
        let mut format = dev.format().with_context(|| "reading format")?;

        let cam_width = 1024;
        let cam_height = 768;

        format.width = cam_width;
        format.height = cam_height;
        format.fourcc = FourCC::new(b"MJPG");

        dev.set_format(&format)?;

        let mut stream = MmapStream::with_buffers(&mut dev, Type::VideoCapture, 4)
            .with_context(|| "Failed to create buffer stream")?;

        loop {
            let (buf, _meta) = stream.next()?;
            let decoded = JpegDecoder::new(io::Cursor::new(buf))?;
            let mut image = image::ImageBuffer::new(cam_width, cam_height);
            decoded.read_image(image.as_mut())?;
            frames.put(image);
        }
    });

    (ret, thread)
}
