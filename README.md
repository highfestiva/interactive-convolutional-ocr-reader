# Interactive OCR reading
Draw a low-res digit and see if the computer understands it. OCR reading is python/tensorflow,
the drawing is html/canvas/jQuery.

The first time you need to train the model, enter 20000 for a fairly good training. The model
is saved automatically when the training is done, so you don't need to redo it again. 20000
training iterations takes about an hour. Using 10000 seems to be sufficient to get half of the
digits right half of the time, at least in my first attempt.

Have fun!

PS. It's mainly copy-pasted from TensorFlow's convolutional [tutorial](https://www.tensorflow.org/tutorials/layers).
