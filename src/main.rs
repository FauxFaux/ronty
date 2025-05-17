use std::convert::TryFrom;
use std::sync::Arc;
use std::sync::Mutex;

use anyhow::{anyhow, Context, Result};
use image::imageops::flip_horizontal;
use image::imageops::flip_vertical;
use image::imageops::resize;
use image::imageops::FilterType;
use image::imageops::{crop_imm, flip_horizontal_in_place};
use image::GenericImageView;
use image::Rgb;
use image::RgbImage;
use minifb::Window;
use minifb::WindowOptions;
use minifb::{Key, KeyRepeat};

use crate::faces::Fleek;
use crate::faces::Predictor;
use crate::img::Pt;
use crate::img::Rect;
use crate::video::make_frames;

mod faces;
mod img;
mod latest;
mod video;

#[derive(Default)]
struct Flip {
    horizontal: bool,
    vertical: bool,
}

fn main() -> Result<()> {
    let mut threads = Vec::new();

    let (frames, frame_thread) = make_frames();
    threads.push(frame_thread);

    let boxen = Arc::new(Mutex::new(Default::default()));
    {
        let frames = Arc::clone(&frames);
        let boxen = Arc::clone(&boxen);
        threads.push(std::thread::spawn(move || -> Result<()> {
            faces::find_faces(frames, boxen)
        }));
    }

    let landmark_predictor = Predictor::default();

    let mut left = Win::new("left", 320, 240, 100, 100).context("left")?;
    let mut right = Win::new("right", 320, 240, 500, 100).context("right")?;
    let mut mouth = Win::new("mouth", 720, 320, 100, 400).context("mouth")?;
    let mut debug = Win::new("debug", 1024, 768, 2560 + 100, 100).context("debug")?;

    let mut distances = Ring::with_capacity(25);

    let mut left_centres = Ring::with_capacity(5);
    let mut right_centres = Ring::with_capacity(5);
    let mut mouth_centres = Ring::with_capacity(5);

    let left_flip = Flip::default();
    let right_flip = Flip::default();
    let mouth_flip = Flip::default();

    let mut relative_zoom = 8.;

    loop {
        for win in [&left, &right, &mouth, &debug] {
            if !win.inner.is_open() || win.inner.is_key_down(Key::Q) {
                return Ok(());
            }

            if win.inner.is_key_pressed(Key::Key0, KeyRepeat::Yes) {
                relative_zoom += 0.4;
            }
            if win.inner.is_key_pressed(Key::Key9, KeyRepeat::Yes) {
                relative_zoom -= 0.4;
            }
        }

        let boxes = boxen.lock().expect("panicked").clone();

        let image = frames.peek();

        let landmarks = landmark_predictor.landmarks_from_faces(&image, &boxes);

        if let Some(landmarks) = landmarks.get(0) {
            let left_eye = landmarks.left_eye().centre();
            let right_eye = landmarks.right_eye().centre();
            let mouth_centre = landmarks.mouth().centre();
            let pupil_distance = distance(left_eye.tuple(), right_eye.tuple());

            distances.push(pupil_distance);
            left_centres.push(left_eye.tuple());
            right_centres.push(right_eye.tuple());
            mouth_centres.push(mouth_centre.tuple());
        }

        let pupil_distance = distances.mean();

        let bounds = image.dimensions();

        // 80px is our "reference" pupil distance (it's very arbitrary)
        let s = (80. * relative_zoom) / pupil_distance;

        let scale = |w: u32, h: u32| ((w as f32 / s) as u32, (h as f32 / s) as u32);

        for (c, win, flip) in [
            (left_centres.mean(), &mut left, &left_flip),
            (right_centres.mean(), &mut right, &right_flip),
            (mouth_centres.mean(), &mut mouth, &mouth_flip),
        ] {
            if (0, 0) == c {
                continue;
            }
            let bx = pick(c, scale(win.w, win.h), bounds);
            let image = crop_imm(&image, bx.x, bx.y, bx.w, bx.h);
            let mut image = resize(&*image, win.w, win.h, FilterType::Triangle);
            if flip.horizontal {
                flip_horizontal_in_place(&mut image);
            }
            if flip.vertical {
                flip_horizontal_in_place(&mut image);
            }
            win.update(&image)?;
        }

        let mut image = image;

        for picked in landmarks {
            for pt in picked.as_ref() {
                let red = Rgb([255, 0, 0]);
                draw_point(&mut image, *pt, red);
            }
        }

        let image = resize(&image, debug.w, debug.h, FilterType::Nearest);
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
                // borderless: true,
                ..WindowOptions::default()
            },
        )
        .map_err(|e| anyhow!("{e:?}"))
        .context("raw window call")?;

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
    let mut x = cx.saturating_sub(w / 2);
    let mut y = cy.saturating_sub(h / 2);

    let r = x + w;
    let b = y + h;

    // bw: 700
    // r: 750
    // move x 50 px left

    x -= r.saturating_sub(bw);
    y -= b.saturating_sub(bh);

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
    let (bw, bh) = image.dimensions();
    let mut poke = |x: u32, y: u32| image.put_pixel(x.clamp(0, bw - 1), y.clamp(0, bh - 1), colour);

    poke(point.x, point.y);
    poke(point.x + 1, point.y);
    poke(point.x + 1, point.y + 1);
    poke(point.x, point.y + 1);
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
