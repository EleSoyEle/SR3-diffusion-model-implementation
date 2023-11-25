# SR3-diffusion-model-implementation
Implementation of SR3 to increase the resolution of images from 64x64 to 512x512 with images of cats.



The comments are in Spanish.

The code is the implementation of SR3 [<https://iterative-refinement.github.io/>].

File structure
|File| use|
|-----:|---------|
|train.py| Train the model| 
|test.py| Get a prediction from the model|
|animation.py| Create a video of how you are reducing noise to the image|

## Test
### Use of test.py
The way to use the file is:
```bash
python3 test.py -p <path image> -i <inference steps> -t <steps model> -o <out file>
```

Example:
``` bash
python3 test.py -p image.jpg -i 100 -o out_image.jpg
```

### Use of animation.py
The way to use the file is: 
```bash
python3 animation.py -p <path image> -i <inference steps> -t <steps model> -o <out image> -v <out video>
```

Example:
```bash
python3 animation.py -p image.jpg -i 100 -o out_img.jpg -v video.mp4
```