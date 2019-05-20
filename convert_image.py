from PIL import Image
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

def _bytes_feature(value):
    """
    Returns a bytes_list from a string / byte.
    """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    #return value

def load_image( infilename ) :
    #img = Image.open( infilename )
    #img.load()
    #data = np.asarray(img, dtype="int32")
    data = tf.read_file(infilename)
    return data

def save_image( npdata, outfilename ) :
    np.save(outfilename, np.array([npdata]))

def serialize_example(image):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
          'image': _bytes_feature(image),
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

save_image(serialize_example(load_image('./sexy.jpg')), 'sexy.npy')
