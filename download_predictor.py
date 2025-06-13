import urllib.request
import bz2
import os

def download_landmarks_predictor():
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    bz2_fname = "shape_predictor_68_face_landmarks.dat.bz2"
    dat_fname = "shape_predictor_68_face_landmarks.dat"

    if os.path.exists(dat_fname):
        print(f"{dat_fname} already exists, skipping download.")
        return

    print("Downloading facial landmarks predictor...")
    urllib.request.urlretrieve(url, bz2_fname)
    
    print("Extracting file...")
    with bz2.BZ2File(bz2_fname) as fr, open(dat_fname, "wb") as fw:
        fw.write(fr.read())
    
    os.remove(bz2_fname)
    print("Done!") 