```
# 70MB
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# 5s
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

mkdir -p data
mv shape_predictor_68_face_landmarks.dat data/

# 3 minutes
RUSTFLAGS='-C target-cpu=native' cargo run --release
```
