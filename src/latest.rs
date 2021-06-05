use std::sync::{Condvar, Mutex};

#[derive(Copy, Clone, Default, Debug, Eq, PartialEq)]
pub struct Token(usize);

#[derive(Default)]
pub struct Latest<T> {
    mutex: Mutex<(Token, Option<T>)>,
    cv: Condvar,
}

impl<T: Clone> Latest<T> {
    #[cfg(never)]
    pub fn take(&self) -> T {
        let mut opt = self.mutex.lock().expect("panicked");
        loop {
            if opt.1.is_some() {
                return (*opt).1.take().expect("just checked");
            }
            opt = self.cv.wait(opt).expect("panicked");
        }
    }

    pub fn peek(&self) -> T {
        let mut opt = self.mutex.lock().expect("panicked");
        loop {
            match &opt.1 {
                Some(t) => return t.clone(),
                None => opt = self.cv.wait(opt).expect("panicked"),
            }
        }
    }

    pub fn when_changed_from(&self, previous: Token) -> (Token, T) {
        let mut opt = self.mutex.lock().expect("panicked");
        loop {
            match &opt.1 {
                Some(t) if opt.0 != previous => return (opt.0, t.clone()),
                _ => opt = self.cv.wait(opt).expect("panicked"),
            }
        }
    }

    pub fn put(&self, val: T) {
        {
            let mut guard = self.mutex.lock().expect("panicked");
            guard.0 = Token(guard.0 .0.wrapping_add(1));
            guard.1 = Some(val);
        }
        self.cv.notify_all();
    }
}
