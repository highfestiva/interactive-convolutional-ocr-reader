#!/usr/bin/env python3

import base64, flask, io, PIL, struct
import numpy as np
import tensorflow as tf


app = flask.Flask(__name__, static_folder='')


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                    labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


@app.route('/')
def root():
    return app.send_static_file('ocr-draw.html')


@app.route('/ocr-char', methods=['POST'])
def content():
    img_base64 = flask.request.values.get('img')
    img = to_float_img(img_base64.partition(',')[2])
    eval_data = np.array([img]*10, dtype=np.float32)
    eval_labels = np.array(range(10), dtype=np.float32)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            batch_size=100,
            num_epochs=1,
            shuffle=False)
    classification = mnist_classifier.predict(input_fn=eval_input_fn)
    probs = next(classification)['probabilities']
    result = max(zip(probs, eval_labels))
    return str(int(result[1]))


def to_float_img(img_base64):
    # use the PNG alpha mask for RGB color; extract R and use as 0-16 grayscale
    data = base64.b64decode(img_base64)
    bmp = io.BytesIO()
    png = PIL.Image.open(io.BytesIO(data))
    white_bkg = PIL.Image.new('RGB', png.size, (255,255,255))
    white_bkg.paste(png, mask=png.split()[3])
    white_bkg.convert('RGB').save(bmp, 'BMP')
    out = bmp.getvalue()
    w = struct.unpack("<L", out[18:22])[0]
    h = struct.unpack("<L", out[22:26])[0]
    return [(255-b)*16/255 for i in range(h-1,-1,-1) for b in out[54+i*w*3:54+(i+1)*w*3:3]]


def runweb():
    import time, threading, webbrowser
    def delayed_browse():
        time.sleep(1)
        webbrowser.open_new_tab('http://localhost:9003/')
    threading.Thread(target=delayed_browse).start()
    app.run(host='0.0.0.0', port=9003, threaded=True)


if __name__ == "__main__":
    # Load training and eval data
    print('fetching mnist data...')
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

    train_additional_steps = input("\nHow many additional steps do you want to train? (0=none, if you've already trained sufficiently) ")
    train_additional_steps = int(train_additional_steps)
    if train_additional_steps:
        print('training %i additional steps...' % train_additional_steps)
        tf.logging.set_verbosity(tf.logging.INFO)
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": train_data},
                y=train_labels,
                batch_size=100,
                num_epochs=None,
                shuffle=True)
        mnist_classifier.train(
                input_fn=train_input_fn,
                steps=train_additional_steps)

    runweb()
