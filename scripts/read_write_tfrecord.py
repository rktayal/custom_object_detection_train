import tensorflow as tf

# Create example data
data = {
    'Age': 29,
    'Movie': ['The Shawshank Redemption', 'Fight Club'],
    'Movie Ratings': [9.0, 9.7],
    'Suggestion': 'Inception',
    'Suggestion Purchased': 1.0,
    'Purchase Price': 9.99
}

print(data)

# Create the Example
example = tf.train.Example(features=tf.train.Features(feature={
    'Age': tf.train.Feature(
        int64_list=tf.train.Int64List(value=[data['Age']])),
    'Movie': tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[m.encode('utf-8') for m in data['Movie']])),
    'Movie Ratings': tf.train.Feature(
        float_list=tf.train.FloatList(value=data['Movie Ratings'])),
    'Suggestion': tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[data['Suggestion'].encode('utf-8')])),
    'Suggestion Purchased': tf.train.Feature(
        float_list=tf.train.FloatList(
            value=[data['Suggestion Purchased']])),
    'Purchase Price': tf.train.Feature(
        float_list=tf.train.FloatList(value=[data['Purchase Price']]))
}))

print(example)

# Write TFrecord file
with tf.python_io.TFRecordWriter('customer_1.tfrecord') as writer:
    writer.write(example.SerializeToString())

# Read and print data:
sess = tf.InteractiveSession()

# Read TFRecord file
reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(['customer_1.tfrecord'])

_, serialized_example = reader.read(filename_queue)

# Define features
read_features = {
    'Age': tf.FixedLenFeature([], dtype=tf.int64),
    'Movie': tf.VarLenFeature(dtype=tf.string),
    'Movie Ratings': tf.VarLenFeature(dtype=tf.float32),
    'Suggestion': tf.FixedLenFeature([], dtype=tf.string),
    'Suggestion Purchased': tf.FixedLenFeature([], dtype=tf.float32),
    'Purchase Price': tf.FixedLenFeature([], dtype=tf.float32)}

# Extract features from serialized data
read_data = tf.parse_single_example(serialized=serialized_example,
                                    features=read_features)

# Many tf.train functions use tf.train.QueueRunner,
# so we need to start it before we read
tf.train.start_queue_runners(sess)

# Print features
for name, tensor in read_data.items():
    print('{}: {}'.format(name, tensor.eval()))
