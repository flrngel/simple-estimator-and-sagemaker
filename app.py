import os
import argparse
import tensorflow as tf
import model
import numpy as np

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

    args, _ = parser.parse_known_args()

    estimator = tf.estimator.Estimator(model_fn=model.model_fn,
            params={
                'learning_rate': args.learning_rate,
                'hidden_h1': args.hidden_h1,
                'label_size': 10
                }
            )

    # load data from keras
    train, test = tf.keras.datasets.mnist.load_data()
    train_x, train_y = train
    train_y = tf.keras.utils.to_categorical(train_y, 10)

    estimator.train(input_fn=lambda:model.input_fn(train_x, train_y, args.epochs, args.batch_size))
