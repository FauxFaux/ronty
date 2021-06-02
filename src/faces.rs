use std::sync::{Arc, Mutex};

use dlib_face_recognition::{
    FaceDetector, FaceDetectorTrait, LandmarkPredictor, LandmarkPredictorTrait, Point,
};
use image::{ImageBuffer, Rgb};
use itertools::Itertools;

use crate::img::Rect;
use crate::latest::Latest;

#[derive(Copy, Clone, Default)]
pub struct Boxen {
    pub left_eye: Rect,
    pub right_eye: Rect,
    pub mouth: Rect,
}

#[derive(Default)]
pub struct Comms {
    pub input: Latest<ImageBuffer<Rgb<u8>, Vec<u8>>>,
    pub output: Mutex<Boxen>,
}

pub fn main(comms: Arc<Comms>) {
    let det = FaceDetector::default();
    let landmarks = LandmarkPredictor::default();

    loop {
        let image = comms.input.get();

        let matrix = dlib_face_recognition::ImageMatrix::from_image(&image);
        let locs = det.face_locations(&matrix);
        let r = match locs.iter().next() {
            Some(r) => r,
            None => continue,
        };
        // for r in locs.iter()
        //     draw_rectangle(&mut image, &r, red);

        let landmarks = landmarks.face_landmarks(&matrix, &r);

        // left eye
        let mut bx = bounding_box(&landmarks[36..42]);
        bx.y -= bx.h / 2;
        bx.h *= 2;

        comms.output.lock().expect("panicked").left_eye = bx;
    }
}

fn bounding_box(marks: &[Point]) -> Rect {
    let (l, r) = minmax_by(marks, |p| p.x());
    let (t, b) = minmax_by(marks, |p| p.y());

    let x = l.x();
    let y = t.y();
    let w = r.x() - x;
    let h = b.y() - y;

    Rect {
        x: force_u32(x),
        y: force_u32(y),
        w: force_u32(w),
        h: force_u32(h),
    }
}

fn force_u32(v: i64) -> u32 {
    v.clamp(0, i64::from(u32::MAX)) as u32
}

fn minmax_by<T, F: FnMut(&&T) -> i64>(items: &[T], key: F) -> (&T, &T) {
    items
        .iter()
        .minmax_by_key(key)
        .into_option()
        .expect("not empty")
}
