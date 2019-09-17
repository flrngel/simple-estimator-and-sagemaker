import os
import argparse
import tensorflow as tf
import model
import numpy as np
#import sagemaker

if __name__ =='__main__':
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--hidden_h1', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.1)

    # input data and model directories
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--debug', type=bool, default=False)

    args, _ = parser.parse_known_args()

    if args.num_gpus > 0:
        strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=args.num_gpus)
        config = tf.estimator.RunConfig(train_distribute=strategy)
    else:
        config = tf.estimator.RunConfig()

    estimator = tf.estimator.Estimator(model_fn=model.model_fn,
            model_dir=args.model_dir,
            params={
                'learning_rate': args.learning_rate,
                'hidden_h1': args.hidden_h1,
                'label_size': 10
                },
            config=config
            )

    if args.debug == True:
        tf.logging.set_verbosity(tf.logging.INFO)

    # load data from keras
    train, test = tf.keras.datasets.mnist.load_data()

    train_x, train_y = train
    train_x = np.array(train_x, dtype=np.float32)
    train_y = tf.keras.utils.to_categorical(train_y, 10)

    estimator.train(input_fn=lambda:model.input_fn(train_x, train_y, args.epochs, args.batch_size))

    # exporting model
    estimator.export_savedmodel('saved_model', model.serving_input_receiver_fn)
