import tensorflow as tf

def _img_string_to_tensor(image_string, image_size=(28, 28)):
    image_decoded = tf.image.decode_jpeg(image_string, channels=1)
    # Convert from full range of uint8 to range [0,1] of float32.
    image_decoded_as_float = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
    # Resize to expected
    image_resized = tf.image.resize_images(image_decoded_as_float, size=image_size)
    image = tf.squeeze(image_resized, -1)
    return image

def input_fn(images, labels, epochs, batch_size):
    # https://medium.com/tensorflow/multi-gpu-training-with-estimators-tf-keras-and-tf-data-ba584c3134db

    # Convert the inputs to a Dataset. (E)
    ds = tf.data.Dataset.from_tensor_slices(({'image': images}, labels))
    # Shuffle, repeat, and batch the examples. (T)
    SHUFFLE_SIZE = 5000
    ds = ds.shuffle(SHUFFLE_SIZE).repeat(epochs).batch(batch_size)
    # check https://cs230-stanford.github.io/tensorflow-input-data.html
    ds = ds.prefetch(2)
    # Return the dataset. (L)
    return ds

def serving_input_receiver_fn():
    # https://www.damienpontifex.com/2018/05/10/tensorflow-serving-with-a-variable-batch-size/

    feature_spec = {
        'image': tf.FixedLenFeature([], dtype=tf.string)
    }
    
    default_batch_size = 1
    
    serialized_tf_example = tf.placeholder(
        dtype=tf.string, shape=[default_batch_size], 
        name='input_image_tensor')
    
    received_tensors = { 'images': serialized_tf_example }
    features = tf.parse_example(serialized_tf_example, feature_spec)
    
    fn = lambda image: _img_string_to_tensor(image)
    
    features['image'] = tf.map_fn(fn, features['image'], dtype=tf.float32)
    
    return tf.estimator.export.ServingInputReceiver(features, received_tensors)


def model_fn(features, labels, mode, params):
    features = features['image']
    inputs = tf.reshape(features, [-1, 28*28])

    net = tf.layers.dense(inputs, params['hidden_h1'], activation=tf.nn.relu)
    logits = tf.layers.dense(net, params['label_size'], activation=tf.nn.sigmoid)

    # important
    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(labels=tf.argmax(labels,1), predictions=tf.argmax(logits,1))
        logging_hook = tf.train.LoggingTensorHook({"loss" : loss, 'accuracy': accuracy[1]}, every_n_iter=100)
        eval_metric_ops = {"accuracy": accuracy}
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, predictions=logits, training_hooks=[logging_hook], eval_metric_ops=eval_metric_ops)
    else:
        return tf.estimator.EstimatorSpec(mode, predictions=logits)
