use std::sync::{Condvar, LockResult, Mutex};

#[derive(Default)]
pub struct Latest<T> {
    mutex: Mutex<Option<T>>,
    cv: Condvar,
}

impl<T> Latest<T> {
    pub fn get(&self) -> T {
        let mut opt = self.mutex.lock().expect("panicked");
        loop {
            if opt.is_some() {
                return (*opt).take().expect("just checked");
            }
            opt = self.cv.wait(opt).expect("panicked");
        }
    }

    pub fn put(&self, val: T) {
        (*self.mutex.lock().expect("panicked")) = Some(val);
        self.cv.notify_one();
    }
}
