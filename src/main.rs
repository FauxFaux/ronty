use std::io;

use crate::faces::Comms;
use anyhow::{Context, Result};
use dlib_face_recognition::*;
use image::codecs::jpeg::JpegDecoder;
use image::imageops::{crop_imm, resize, FilterType};
use image::*;
use itertools::Itertools;
use minifb::{Key, Window, WindowOptions};
use std::sync::Arc;
use v4l::buffer::Type;
use v4l::context::enum_devices;
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture;
use v4l::FourCC;

mod faces;
mod img;
mod latest;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

fn main() -> Result<()> {
    let comms = Arc::new(Comms::default());
    let theirs = Arc::clone(&comms);
    std::thread::spawn(move || faces::main(theirs));

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

    let mut buffer: Vec<u32> = vec![0; (WIDTH * HEIGHT) as usize];

    let mut window = Window::new(
        "Test - ESC to exit",
        WIDTH as usize,
        HEIGHT as usize,
        WindowOptions::default(),
    )?;

    // Limit to max ~60 fps update rate
    window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let (buf, _meta) = stream.next()?;

        let decoded = JpegDecoder::new(io::Cursor::new(buf))?;
        let mut image = image::ImageBuffer::new(cam_width, cam_height);
        decoded.read_image(image.as_mut())?;

        comms.input.put(image.clone());

        let boxes = comms.output.lock().expect("panicked").clone();

        // left eye
        let mut bx = boxes.left_eye;
        bx.y -= bx.h / 2;
        bx.h *= 2;
        let image = crop_imm(&image, bx.x, bx.y, bx.w, bx.h);

        let (w, h) = image.dimensions();
        let xscale = (w as f32) / (WIDTH as f32);
        let yscale = (h as f32) / (HEIGHT as f32);
        let scale = xscale.max(yscale);

        let image = resize(
            &image,
            ((w as f32) / scale) as u32,
            ((h as f32) / scale) as u32,
            FilterType::Triangle,
        );

        for h in 0..image.height().min(HEIGHT) {
            for w in 0..image.width().min(WIDTH) {
                let p = image.get_pixel(w, h).0;
                buffer[(h * WIDTH + w) as usize] =
                    p[2] as u32 + 256 * p[1] as u32 + 256 * 256 * p[0] as u32;
            }
        }

        window.update_with_buffer(&buffer, WIDTH as usize, HEIGHT as usize)?;
    }
    Ok(())
}

fn image_from_yuyv(buf: &[u8]) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut image = image::ImageBuffer::new(WIDTH as u32, HEIGHT as u32);

    for i in 0..((WIDTH * HEIGHT) as usize).min(buf.len() / 2) {
        let p = buf[i * 2];
        image.put_pixel(
            (i % WIDTH as usize) as u32,
            (i / WIDTH as usize) as u32,
            Rgb([p, p, p]),
        );
    }
    image
}

fn draw_rectangle(image: &mut RgbImage, rect: &Rectangle, colour: Rgb<u8>) {
    for x in rect.left..rect.right {
        image.put_pixel(x as u32, rect.top as u32, colour);
        image.put_pixel(x as u32, rect.bottom as u32, colour);
    }

    for y in rect.top..rect.bottom {
        image.put_pixel(rect.left as u32, y as u32, colour);
        image.put_pixel(rect.right as u32, y as u32, colour);
    }
}

fn draw_point(image: &mut RgbImage, point: &Point, colour: Rgb<u8>) {
    image.put_pixel(point.x() as u32, point.y() as u32, colour);
    image.put_pixel(point.x() as u32 + 1, point.y() as u32, colour);
    image.put_pixel(point.x() as u32 + 1, point.y() as u32 + 1, colour);
    image.put_pixel(point.x() as u32, point.y() as u32 + 1, colour);
}
