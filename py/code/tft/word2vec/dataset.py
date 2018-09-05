import tensorflow as tf

from tensorflow.python.ops import lookup_ops


def build_dataset(data_file, vocab_file, batch_size=128, t=3, prefetch_size=1):
    vocab_table = lookup_ops.index_table_from_file(vocab_file, default_value=0)
    dataset = tf.data.TextLineDataset(data_file)
    dataset = dataset.map(lambda line: tf.string_split([line]).values, num_parallel_calls=t)
    dataset = dataset.map(lambda words: vocab_table.lookup(words), num_parallel_calls=t)
    dataset = dataset.map(lambda words: (words[0], words[1]), num_parallel_calls=t)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_size)
    return dataset, vocab_table