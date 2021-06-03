use std::io;
use std::io::Write as _;
use std::sync::{Arc, Mutex};

use anyhow::Result;

use dlib_face_recognition::{
    FaceDetector, FaceDetectorTrait, LandmarkPredictor, LandmarkPredictorTrait, Point,
};
use image::{ImageBuffer, Rgb};
use itertools::Itertools;

use crate::img::Rect;
use crate::latest::{Latest, Token};

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

pub fn main(
    input: Arc<Latest<ImageBuffer<Rgb<u8>, Vec<u8>>>>,
    output: Arc<Mutex<Boxen>>,
) -> Result<()> {
    let det = FaceDetector::default();
    let landmarks = landmark_from_include();

    let mut token = Token::default();
    loop {
        let (current, image) = input.when_changed_from(token);
        token = current;

        let matrix = dlib_face_recognition::ImageMatrix::from_image(&image);
        let locs = det.face_locations(&matrix);
        let r = match locs.iter().next() {
            Some(r) => r,
            None => continue,
        };
        // for r in locs.iter()
        //     draw_rectangle(&mut image, &r, red);

        let landmarks = landmarks.face_landmarks(&matrix, &r);

        let left_eye = bounding_box(&landmarks[36..=41]);
        let right_eye = bounding_box(&landmarks[42..=47]);
        let mouth = bounding_box(&landmarks[48..]);

        let mut output = output.lock().expect("panicked");
        output.left_eye = left_eye;
        output.right_eye = right_eye;
        output.mouth = mouth;
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

fn landmark_from_include() -> LandmarkPredictor {
    let mut t = tempfile::NamedTempFile::new().expect("creating temporary file");
    io::copy(
        &mut io::Cursor::new(&include_bytes!("../data/shape_predictor_68_face_landmarks.dat")[..]),
        &mut t,
    )
    .expect("writing file");
    t.flush().expect("flushing file");
    let landmarks = LandmarkPredictor::new(t.path()).expect("static data");
    t.close().expect("removing temporary file");
    landmarks
}
