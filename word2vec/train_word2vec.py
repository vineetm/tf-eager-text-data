import argparse
import numpy as np
import time
import tensorflow as tf
from tensorflow.python.ops import lookup_ops
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()
np.random.seed(42)
tf.set_random_seed(42)

logging = tf.logging
logging.set_verbosity(logging.INFO)


SENTENCES = 'wiki.1M.txt.tokenized'
WINDOW_SIZE = 5

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-vocab')
    parser.add_argument('-d', type=int, default=128)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-num_samples', type=int, default=5)
    parser.add_argument('-lr', default=0.01, type=float)
    parser.add_argument('-log_freq', type=int, default=1000)
    return parser.parse_args()


def skip_gram_src_tgt_generator(sentences_file=SENTENCES, window_size=WINDOW_SIZE):
    with open(sentences_file) as fr:
        for line in fr:
            words = line.split()
            word_tensors = tf.convert_to_tensor(words)
            for word_index in range(len(words)):
                lo = word_index - window_size
                lo = lo if lo >= 0 else 0
                ro = word_index + window_size
                ro = ro if ro < len(words) else len(words)

                if word_index > lo:
                    yield word_tensors[word_index], word_tensors[np.random.randint(lo, high=word_index)]
                if ro > word_index + 1:
                    yield word_tensors[word_index], word_tensors[np.random.randint(word_index + 1, high=ro)]


def build_dataset(vocab_file, batch_size):
    vocab_table = lookup_ops.index_table_from_file(vocab_file, default_value=0)
    dataset = tf.data.Dataset.from_generator(generator=skip_gram_src_tgt_generator, output_types=(tf.string, tf.string))
    dataset = dataset.map(lambda src_word, tgt_word: (vocab_table.lookup(src_word), vocab_table.lookup(tgt_word)))
    dataset = dataset.batch(batch_size)
    return dataset, vocab_table


class Word2Vec(object):
    def __init__(self, V, d, num_sampled=5):
        self.W = tfe.Variable(tf.random_uniform([V, d]))
        self.nce_W = tfe.Variable(tf.truncated_normal([V, d]))
        self.nce_b = tfe.Variable(tf.zeros(V))
        self.V = V
        self.num_sampled = num_sampled

    def compute_loss(self, src_words, tgt_words):
        word_vectors = tf.nn.embedding_lookup(self.W, src_words)
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.nce_W, biases=self.nce_b,
                                             labels=tf.expand_dims(tgt_words, axis=1),
                                             inputs=word_vectors,
                                             num_sampled=self.num_sampled, num_classes=self.V))
        return loss


def main():
    args = setup_args()
    logging.info(args)

    dataset, vocab_table = build_dataset(args.vocab, args.batch_size)
    V = vocab_table.size()

    model = Word2Vec(V, args.d)
    grad_fun = tfe.implicit_value_and_gradients(model.compute_loss)
    opt = tf.train.GradientDescentOptimizer(learning_rate=args.lr)

    train_step = 0
    total_loss = 0.0
    start_time = time.time()
    for src_words, tgt_words in dataset:
        batch_loss, gradients = grad_fun(src_words, tgt_words)
        opt.apply_gradients(gradients)
        total_loss += batch_loss
        train_step += 1
        if train_step % args.log_freq == 0:
            time_taken = time.time() - start_time
            logging.info(f'Step: {train_step} Loss: {total_loss/args.log_freq} Time: {time_taken}')
            start_time = time.time()
            total_loss = 0.



if __name__ == '__main__':
    main()