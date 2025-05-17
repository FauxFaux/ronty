use std::convert::TryInto;
use std::io;
use std::io::Write as _;
use std::sync::Arc;
use std::sync::Mutex;

use anyhow::Result;
use dlib_face_recognition::FaceDetector;
use dlib_face_recognition::FaceDetectorTrait;
use dlib_face_recognition::LandmarkPredictor;
use dlib_face_recognition::LandmarkPredictorTrait;
use dlib_face_recognition::Point;
use dlib_face_recognition::Rectangle;
use image::RgbImage;
use itertools::Itertools;

use crate::img::{Pt, Rect};
use crate::latest::{Latest, Token};

#[derive(Copy, Clone)]
pub struct Landmarks {
    inner: [Pt; 68],
}

pub fn find_faces(input: Arc<Latest<RgbImage>>, output: Arc<Mutex<Vec<Rectangle>>>) -> Result<()> {
    let det = FaceDetector::default();

    let mut token = Token::default();
    loop {
        let (current, image) = input.when_changed_from(token);
        let gap = current.gap_since(token);
        if gap > 1 {
            println!("{} frame drop at face detector", gap - 1);
        }
        token = current;

        let matrix = dlib_face_recognition::ImageMatrix::from_image(&image);
        let locs = det.face_locations(&matrix);
        let mut output = output.lock().expect("panicked");
        *output = locs.to_vec();
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

pub struct Predictor {
    inner: LandmarkPredictor,
}

impl Predictor {
    pub fn from_include() -> Self {
        Predictor {
            inner: landmark_from_include(),
        }
    }

    pub fn landmarks_from_faces(&self, image: &RgbImage, faces: &[Rectangle]) -> Vec<Landmarks> {
        let matrix = dlib_face_recognition::ImageMatrix::from_image(image);
        faces
            .iter()
            .map(|rect| self.inner.face_landmarks(&matrix, rect))
            .map(|l| Landmarks::from_dlib(&l))
            .collect()
    }
}

impl Default for Predictor {
    fn default() -> Self {
        Self::from_include()
    }
}

fn landmark_from_include() -> LandmarkPredictor {
    let mut t = tempfile::NamedTempFile::new().expect("creating temporary file");
    io::copy(
        &mut io::Cursor::new(&include_bytes!("../data/shape_predictor_68_face_landmarks.dat")[..]),
        &mut t,
    )
    .expect("writing file");
    t.flush().expect("flushing file");
    let landmarks = LandmarkPredictor::open(t.path()).expect("static data");
    t.close().expect("removing temporary file");
    landmarks
}

impl Landmarks {
    fn from_dlib(dlib: &[Point]) -> Landmarks {
        assert_eq!(68, dlib.len());
        let inner = dlib
            .into_iter()
            .map(|p| Pt {
                x: force_u32(p.x()),
                y: force_u32(p.y()),
            })
            .collect::<Vec<_>>()
            .try_into()
            .expect("invalid number of landmarks");
        Landmarks { inner }
    }

    pub fn left_eye(&self) -> &[Pt] {
        &self.inner[36..=41]
    }

    pub fn right_eye(&self) -> &[Pt] {
        &self.inner[42..=47]
    }

    pub fn mouth(&self) -> &[Pt] {
        &self.inner[48..]
    }
}

impl AsRef<[Pt]> for Landmarks {
    fn as_ref(&self) -> &[Pt] {
        &self.inner
    }
}

pub trait Fleek {
    fn centre(&self) -> Pt;
}

impl Fleek for &[Pt] {
    fn centre(&self) -> Pt {
        let xs: u32 = self.iter().map(|&v| v.x).sum();
        let ys: u32 = self.iter().map(|&v| v.y).sum();
        Pt {
            x: xs / self.len() as u32,
            y: ys / self.len() as u32,
        }
    }
}
