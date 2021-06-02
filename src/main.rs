use std::io;

use anyhow::{Context, Result};
use dlib_face_recognition::*;
use image::codecs::jpeg::JpegDecoder;
use image::imageops::{crop_imm, resize, FilterType};
use image::*;
use minifb::{Key, Window, WindowOptions};
use v4l::buffer::Type;
use v4l::context::enum_devices;
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture;
use v4l::FourCC;

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 768;

fn main() -> Result<()> {
    let mut dev = Device::new(0).with_context(|| "Failed to open device")?;
    let mut format = dev.format().with_context(|| "reading format")?;

    let red = Rgb([255, 0, 0]);
    let green = Rgb([0, 255, 0]);
    let blue = Rgb([32, 32, 255]);

    format.width = WIDTH as u32;
    format.height = HEIGHT as u32;
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

    let det = FaceDetector::default();
    // let det = FaceDetectorCnn::default();
    let landmarks = LandmarkPredictor::default();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        for i in buffer.iter_mut() {
            *i = 0; // write something more funny here!
        }

        let (buf, meta) = stream.next()?;

        // let mut image = image_from_yuyv(&buf);

        let decoded = JpegDecoder::new(io::Cursor::new(buf))?;
        let mut image = image::ImageBuffer::new(WIDTH as u32, HEIGHT as u32);
        decoded.read_image(image.as_mut())?;

        let matrix = dlib_face_recognition::ImageMatrix::from_image(&image);
        let locs = det.face_locations(&matrix);
        let r = match locs.iter().next() {
            Some(r) => r,
            None => continue,
        };
        // for r in locs.iter()
        //     draw_rectangle(&mut image, &r, red);

        let landmarks = landmarks.face_landmarks(&matrix, &r);
        let centre = landmarks[33];
        let xs = landmarks.iter().map(|p| p.x()).collect::<Vec<_>>();
        let ys = landmarks.iter().map(|p| p.y()).collect::<Vec<_>>();

        // guaranteed by clamp
        let min_x = xs
            .iter()
            .min()
            .expect("known length")
            .clone()
            .clamp(0, i64::from(WIDTH)) as u32;
        let max_x = xs
            .iter()
            .max()
            .expect("known length")
            .clone()
            .clamp(0, i64::from(WIDTH)) as u32;
        let min_y = ys
            .iter()
            .min()
            .expect("known length")
            .clone()
            .clamp(0, i64::from(HEIGHT)) as u32;
        let max_y = ys
            .iter()
            .max()
            .expect("known length")
            .clone()
            .clamp(0, i64::from(HEIGHT)) as u32;

        let image = crop_imm(&image, min_x, min_y, max_x - min_x, max_y - min_y);

        // for (x, point) in landmarks.iter().enumerate() {
        //     let colour = match x {
        //         36..=47 => green,
        //         33 => blue,
        //         _ => red,
        //     };
        //     draw_point(&mut image, &point, colour);
        // }

        let image = resize(&image, WIDTH, HEIGHT, FilterType::Triangle);

        for h in 0..image.height() {
            for w in 0..image.width() {
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

    for i in 0..((WIDTH * HEIGHT) as usize).min((buf.len() / 2)) {
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
