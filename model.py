import tensorflow as tf

# https://medium.com/tensorflow/multi-gpu-training-with-estimators-tf-keras-and-tf-data-ba584c3134db
def input_fn(images, labels, epochs, batch_size):
    # Convert the inputs to a Dataset. (E)
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    # Shuffle, repeat, and batch the examples. (T)
    SHUFFLE_SIZE = 5000
    ds = ds.shuffle(SHUFFLE_SIZE).repeat(epochs).batch(batch_size)
    # check https://cs230-stanford.github.io/tensorflow-input-data.html
    ds = ds.prefetch(2)
    # Return the dataset. (L)
    return ds

def model_fn(features, labels, mode, params):
    inputs = tf.reshape(features, [-1, 28*28])

    net = tf.layers.dense(inputs, params['hidden_h1'], activation=tf.nn.relu)
    logits = tf.layers.dense(net, params['label_size'], activation=tf.nn.sigmoid)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    accuracy = tf.metrics.accuracy(labels=tf.argmax(labels,1), predictions=tf.argmax(logits,1))
    logging_hook = tf.train.LoggingTensorHook({"loss" : loss, 'accuracy': accuracy[1]}, every_n_iter=100)
    eval_metric_ops = {"accuracy": accuracy}

    # https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, predictions=logits, training_hooks=[logging_hook], eval_metric_ops=eval_metric_ops)
