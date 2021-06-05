use std::convert::TryFrom;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use image::imageops::{crop_imm, flip_horizontal, flip_vertical, grayscale, resize, FilterType};
use image::{GenericImageView, ImageBuffer, Rgb, RgbImage};
use minifb::{Key, Window, WindowOptions};

use crate::faces::{Boxen, Fleek};
use crate::img::{Pt, Rect};
use crate::video::make_frames;
use dlib_face_recognition::{Point, Rectangle};

mod faces;
mod img;
mod latest;
mod video;

fn main() -> Result<()> {
    let mut threads = Vec::new();

    let (frames, frame_thread) = make_frames();
    threads.push(frame_thread);

    let boxen = Arc::new(Mutex::new(Boxen::default()));
    {
        let frames = Arc::clone(&frames);
        let boxen = Arc::clone(&boxen);
        threads.push(std::thread::spawn(move || -> Result<()> {
            faces::main(frames, boxen)
        }));
    }

    let mut left = Win::new("left", 320, 240, 100, 100)?;
    let mut right = Win::new("right", 320, 240, 500, 100)?;
    let mut mouth = Win::new("mouth", 720, 320, 100, 400)?;
    let mut debug = Win::new("debug", 1024, 768, 800, 400)?;

    let mut distances = Ring::with_capacity(25);

    let mut left_centres = Ring::with_capacity(5);
    let mut right_centres = Ring::with_capacity(5);
    let mut mouth_centres = Ring::with_capacity(5);

    loop {
        for win in [&left, &right, &mouth] {
            if !win.inner.is_open() || win.inner.is_key_down(Key::Q) {
                return Ok(());
            }
        }

        let boxes = boxen.lock().expect("panicked").clone();

        if boxes.landmarks.is_empty() {
            continue;
        }

        let left_eye = boxes.landmarks[0].left_eye().centre();
        let right_eye = boxes.landmarks[0].right_eye().centre();
        let mouth_centre = boxes.landmarks[0].mouth().centre();

        let image = frames.peek();

        let pupil_distance = distance(left_eye.tuple(), right_eye.tuple());

        distances.push(pupil_distance);

        // println!("{} {}", pupil_distance, distances.mean());
        let pupil_distance = distances.mean();

        let bounds = image.dimensions();

        let s = (80. * 8.) / pupil_distance;

        let scale = |w: u32, h: u32| ((w as f32 / s) as u32, (h as f32 / s) as u32);

        left_centres.push(left_eye.tuple());
        right_centres.push(right_eye.tuple());
        mouth_centres.push(mouth_centre.tuple());

        for (c, win) in [
            (left_centres.mean(), &mut left),
            (right_centres.mean(), &mut right),
            (mouth_centres.mean(), &mut mouth),
        ] {
            let bx = pick(c, scale(win.w, win.h), bounds);
            let image = crop_imm(&image, bx.x, bx.y, bx.w, bx.h);
            let image = resize(&image, win.w, win.h, FilterType::Triangle);
            win.update(&image)?;
        }

        let mut image = image;

        for pt in boxes.landmarks[0].as_ref() {
            let red = Rgb([255, 0, 0]);
            draw_point(&mut image, *pt, red);
        }

        debug.update(&image)?;
    }
}

struct Win {
    inner: Window,
    buffer: Box<[u32]>,
    w: u32,
    h: u32,
}

impl Win {
    fn new(hint: &str, w: u32, h: u32, x: isize, y: isize) -> Result<Win> {
        let uw = usize::try_from(w).expect("usize compatible width please");
        let uh = usize::try_from(h).expect("usize compatible height please");
        let mut inner = Window::new(
            hint,
            uw,
            uh,
            WindowOptions {
                borderless: true,
                ..WindowOptions::default()
            },
        )?;

        let ul = uw.checked_mul(uh).expect("too big for memory");

        inner.set_position(x, y);

        // Limit to max ~60 fps update rate
        inner.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

        Ok(Win {
            inner,
            buffer: vec![0u32; ul].into_boxed_slice(),
            w,
            h,
        })
    }

    fn update<I: GenericImageView<Pixel = Rgb<u8>>>(&mut self, image: &I) -> Result<()> {
        assert_eq!(
            (self.w, self.h),
            (image.width(), image.height()),
            "invalid image dimesions"
        );

        for h in 0..image.height() {
            for w in 0..image.width() {
                let p = image.get_pixel(w, h).0;
                self.buffer[(h * self.w + w) as usize] =
                    p[2] as u32 + 256 * p[1] as u32 + 256 * 256 * p[0] as u32;
            }
        }
        Ok(self
            .inner
            .update_with_buffer(&self.buffer, self.w as usize, self.h as usize)?)
    }
}

fn distance((ax, ay): (u32, u32), (bx, by): (u32, u32)) -> f32 {
    let l = ax.min(bx);
    let r = ax.max(bx);
    let t = ay.min(by);
    let b = ay.max(by);

    let w = r - l;
    let h = b - t;

    ((w * w + h * h) as f32).sqrt()
}

fn pick((cx, cy): (u32, u32), (w, h): (u32, u32), (bw, bh): (u32, u32)) -> Rect {
    let x = cx - w / 2;
    let y = cy - h / 2;
    // TODO: in bounds
    Rect { x, y, w, h }
}

#[cfg(never)]
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

fn draw_rectangle(image: &mut RgbImage, rect: Rect, colour: Rgb<u8>) {
    let r = rect.x + rect.w;
    let b = rect.y + rect.h;
    for x in rect.x..r {
        image.put_pixel(x, rect.y, colour);
        image.put_pixel(x, b, colour);
    }

    for y in rect.y..b {
        image.put_pixel(rect.x, y, colour);
        image.put_pixel(r, y, colour);
    }
}

fn draw_point(image: &mut RgbImage, point: Pt, colour: Rgb<u8>) {
    image.put_pixel(point.x, point.y, colour);
    image.put_pixel(point.x + 1, point.y, colour);
    image.put_pixel(point.x + 1, point.y + 1, colour);
    image.put_pixel(point.x, point.y + 1, colour);
}

struct Ring<T> {
    buf: Box<[T]>,
    idx: usize,
    end: usize,
}

impl<T: Default + Clone> Ring<T> {
    fn with_capacity(cap: usize) -> Ring<T> {
        Ring {
            buf: vec![T::default(); cap].into_boxed_slice(),
            idx: 0,
            end: 0,
        }
    }
}

impl<T: Clone> Ring<T> {
    fn push(&mut self, val: T) {
        if self.end < self.buf.len() {
            self.end += 1;
        }

        self.buf[self.idx] = val;
        self.idx += 1;
        if self.idx >= self.buf.len() {
            self.idx = 0;
        }
    }
}

impl<T> Ring<T> {
    fn is_empty(&self) -> bool {
        self.end == 0
    }
}

impl Ring<u32> {
    fn mean(&self) -> u32 {
        if self.is_empty() {
            return 0;
        }
        let mut sum = 0;
        for val in &self.buf[..self.end] {
            sum += val;
        }
        // lazy
        sum / (self.end as u32)
    }
}

impl Ring<f32> {
    fn mean(&self) -> f32 {
        if self.is_empty() {
            return 0.;
        }
        let mut sum = 0.;
        for val in &self.buf[..self.end] {
            sum += val;
        }
        // lazy
        sum / (self.end as f32)
    }
}

impl Ring<(u32, u32)> {
    fn mean(&self) -> (u32, u32) {
        if self.is_empty() {
            return (0, 0);
        }
        let mut sum = (0, 0);
        for val in &self.buf[..self.end] {
            sum.0 += val.0;
            sum.1 += val.1;
        }

        (sum.0 / self.end as u32, sum.1 / self.end as u32)
    }
}
