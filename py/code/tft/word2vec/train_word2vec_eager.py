import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tft.word2vec.dataset import build_dataset
from tft.utils import log_msg

tf.enable_eager_execution()
np.random.seed(42)
tf.set_random_seed(42)

logging = tf.logging
logging.set_verbosity(logging.INFO)


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', help='Src/Tgt Word file')
    parser.add_argument('-vocab', help='Vocab file')
    parser.add_argument('-batch_size', type=int, default=1024)
    parser.add_argument('-d', type=int, default=128, help='Word2Vec dimension')

    parser.add_argument('-window_size', default=5, type=int)

    parser.add_argument('-nce_samples', type=int, default=5)
    parser.add_argument('-lr', default=0.01, type=float)
    parser.add_argument('-log_freq', type=int, default=1000)

    parser.add_argument('-t', default=3, type=int)
    parser.add_argument('-prefetch', default=1, type=int)
    return parser.parse_args()


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

    dataset, vocab_table = build_dataset(data_file=args.data, vocab_file=args.vocab, batch_size=args.batch_size,
                                         t=args.t, prefetch_size=args.prefetch)
    V = vocab_table.size()

    model = Word2Vec(V, args.d)
    grad_fun = tfe.implicit_value_and_gradients(model.compute_loss)
    opt = tf.train.GradientDescentOptimizer(learning_rate=args.lr)

    train_step = 0
    total_loss = 0.0
    for src_words, tgt_words in dataset:
        batch_loss, gradients = grad_fun(src_words, tgt_words)
        opt.apply_gradients(gradients)
        total_loss += batch_loss
        train_step += 1
        if train_step % args.log_freq == 0:
            log_msg(f'Step: {train_step} Loss: {total_loss/args.log_freq}')
            total_loss = 0.
    log_msg(f'Num steps: {train_step} Done!')

if __name__ == '__main__':
    main()