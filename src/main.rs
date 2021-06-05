use std::convert::TryFrom;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use image::{GenericImageView, ImageBuffer, Rgb};
use image::imageops::{crop_imm, FilterType, resize, flip_horizontal, flip_vertical};
use minifb::{Key, Window, WindowOptions};
use num_integer::Roots;

use crate::faces::Boxen;
use crate::video::make_frames;
use crate::img::Rect;

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

    let mut left = Win::new("left", 500, 300, 100, 100)?;
    let mut right = Win::new("right", 500, 300, 500, 100)?;
    let mut mouth = Win::new("mouth", 720, 320, 100, 400)?;
    let mut debug = Win::new("debug", 1024, 768, 800, 400)?;
    // let mut debug = Win::new("debug", 1280, 720, 800, 400)?;

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

        // booting
        if boxes.left_eye.w == 0 {
            continue;
        }
        let image = frames.peek();

        let pupil_distance = distance(
            boxes.left_eye.centre(),
            boxes.right_eye.centre(),);

        distances.push(pupil_distance);

        // println!("{} {}", pupil_distance, distances.mean());
        let pupil_distance = distances.mean();

        let bounds = image.dimensions();

        let s = (80. * 8.) / pupil_distance;

        let scale = |w: u32, h: u32| ((w as f32 / s) as u32, (h as f32 / s) as u32);

        left_centres.push(boxes.left_eye.centre());
        right_centres.push(boxes.right_eye.centre());
        mouth_centres.push(boxes.mouth.centre());

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

        debug.update(&image)?;

        // let bl = pick(boxes.left_eye.centre(), scale(left.w, left.h), bounds);
        // left.update(&crop_imm(&image, bl.x, bl.y, bl.w, bl.h))?;
        //
        // let br = pick(boxes.right_eye.centre(), scale(right.w, right.h), bounds);
        // right.update(&crop_imm(&image, br.x, br.y, br.w, br.h))?;
        //
        // let bm = pick(boxes.mouth.centre(), scale(mouth.w, mouth.h), bounds);
        // mouth.update(&crop_imm(&image, bm.x, bm.y, bm.w, bm.h))?;

        // let (w, h) = image.dimensions();
        // let xscale = (w as f32) / (left.w as f32);
        // let yscale = (h as f32) / (left.h as f32);
        // let scale = xscale.max(yscale);
        //
        // let image = resize(
        //     &image,
        //     ((w as f32) / scale) as u32,
        //     ((h as f32) / scale) as u32,
        //     FilterType::Triangle,
        // );

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
        assert_eq!((self.w, self.h), (image.width(), image.height()), "invalid image dimesions");

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
    Rect {
        x, y, w, h
    }
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

#[cfg(never)]
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

#[cfg(never)]
fn draw_point(image: &mut RgbImage, point: &Point, colour: Rgb<u8>) {
    image.put_pixel(point.x() as u32, point.y() as u32, colour);
    image.put_pixel(point.x() as u32 + 1, point.y() as u32, colour);
    image.put_pixel(point.x() as u32 + 1, point.y() as u32 + 1, colour);
    image.put_pixel(point.x() as u32, point.y() as u32 + 1, colour);
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
            return 0
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
            return 0.
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
            return (0, 0)
        }
        let mut sum = (0, 0);
        for val in &self.buf[..self.end] {
            sum.0 += val.0;
            sum.1 += val.1;
        }

        (
            sum.0 / self.end as u32,
            sum.1 / self.end as u32,
            )
    }
}
