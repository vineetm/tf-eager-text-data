import argparse
import numpy as np
import tensorflow as tf
import time
import tensorflow.contrib.eager as tfe

from tensorflow.python.ops import lookup_ops

tf.enable_eager_execution()

logging = tf.logging
logging.set_verbosity(logging.INFO)

tf.set_random_seed(42)


def log_msg(msg):
    logging.info(f'{time.ctime()}: {msg}')


def create_dataset(data_file, vocab_table, batch_size, eos, t):
    dataset = tf.data.TextLineDataset(data_file)
    dataset = dataset.map(lambda sentence: tf.string_split([sentence]).values, num_parallel_calls=t)
    dataset = dataset.map(lambda words: (words, tf.concat([words[1:], [eos]], axis=0)), num_parallel_calls=t)
    dataset = dataset.map(lambda src_words, tgt_words: (vocab_table.lookup(src_words), vocab_table.lookup(tgt_words)), num_parallel_calls=t)
    dataset = dataset.map(lambda src_words, tgt_words: (src_words, tgt_words, tf.size(src_words)), num_parallel_calls=t)

    #src_words, tgt_words, and num of words.
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None], [None], []))
    dataset = dataset.prefetch(1)
    return dataset


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train')
    parser.add_argument('-valid')
    parser.add_argument('-vocab')

    parser.add_argument('-t', default=4, type=int)

    parser.add_argument('-nd', default=128, type=int)
    parser.add_argument('-nh', default=128, type=int)

    parser.add_argument('-cell', default='rnn', help='rnn|lstm')
    parser.add_argument('-bs', default=32, type=int)

    parser.add_argument('-eval_freq', default=5000, type=int)
    parser.add_argument('-stats_freq', default=1000, type=int)

    parser.add_argument('-unk', default='<unk>')
    parser.add_argument('-unk_index', default=0, type=int)
    parser.add_argument('-eos', default='<eos>')
    parser.add_argument('-max_epochs', default=100, type=int)

    parser.add_argument('-lr', default=0.01, type=float)
    parser.add_argument('-opt', default='sgd')
    return parser.parse_args()


class LanguageModel(tf.keras.Model):
    def __init__(self, d, h, V, cell):
        super(LanguageModel, self).__init__()
        self.W = tfe.Variable(tf.random_uniform(minval=-1.0, maxval=1.0, shape=[V, d]))
        if cell == 'lstm':
            self.rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=h)
        else:
            self.rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=h)

        # We do this to get away with specifying input!
        self.output_layer = tf.keras.layers.Dense(V, kernel_initializer=tf.random_uniform_initializer(-1.0, 1.0))

    def call(self, inputs):
        # This is BS x T x d
        word_vectors = tf.nn.embedding_lookup(self.W, inputs[0])

        # This is [BS x d] list
        word_vectors_time = tf.unstack(word_vectors, axis=1)
        state = self.rnn_cell.zero_state(tf.shape(word_vectors)[0], dtype=tf.float32)
        outputs, final_state = tf.nn.static_rnn(self.rnn_cell, word_vectors_time, state, sequence_length=inputs[2])
        # Outputs is [BS x h], convert to BS X T X h
        outputs = tf.stack(outputs, axis=1)
        outputs = self.output_layer(outputs)
        return outputs


def loss_fn(model, datum):
    pred_logits = model(datum)
    batch_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_logits, labels=datum[1]))
    return batch_loss / tf.cast(tf.reduce_sum(datum[2]), dtype=tf.float32)


def compute_ppl(model, dataset):
    batch_losses = []
    num_elems = []
    for datum in dataset:
        batch_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model(datum), labels=datum[1]))
        batch_losses.append(batch_loss.numpy())
        num_elems.append(np.sum(datum[2]))

    total_loss = np.sum(np.array(batch_losses))
    length = np.sum(np.array(num_elems))
    return np.exp(total_loss / length)


def main():
    args = setup_args()
    log_msg(args)

    vocab_table = lookup_ops.index_table_from_file(args.vocab, default_value=args.unk_index)
    train_dataset = create_dataset(args.train, vocab_table, args.bs, args.eos, args.t)
    valid_dataset = create_dataset(args.valid, vocab_table, args.bs, args.eos, args.t)

    grad_fun = tfe.implicit_value_and_gradients(loss_fn)

    V = vocab_table.size()
    lm = LanguageModel(args.nd, args.nh, V, args.cell)

    valid_ppl = compute_ppl(lm, valid_dataset)
    log_msg(f'Start PPL: {valid_ppl: 0.4f}')

    if args.opt == 'gd':
        opt = tf.train.GradientDescentOptimizer(learning_rate=args.lr)
    else:
        opt = tf.train.AdamOptimizer(learning_rate=args.lr)

    for epoch_num in range(args.max_epochs):
        log_msg(f'Epoch: {epoch_num} START')

        for train_step, train_datum in enumerate(train_dataset, start=1):
            loss_value, gradients = grad_fun(lm, train_datum)

            if train_step % args.eval_freq == 0:
                ppl = compute_ppl(lm, valid_dataset)
                if ppl < valid_ppl:
                    log_msg(f'Epoch: {epoch_num} Step: {train_step} ppl better:{ppl: 0.4f}/{valid_ppl: 0.4f}')
                    valid_ppl = ppl
                else:
                    log_msg(f'Epoch: {epoch_num} Step: {train_step} ppl worse:{ppl: 0.4f}/{valid_ppl: 0.4f}')

            if train_step % args.stats_freq == 0:
                log_msg(f'Epoch: {epoch_num} Step: {train_step} Train_Loss:{loss_value: 0.4f} ppl:{valid_ppl: 0.4f}')

            opt.apply_gradients(gradients)

        ppl = compute_ppl(lm, valid_dataset)
        if ppl < valid_ppl:
            valid_ppl = ppl
        log_msg(f'Epoch: {epoch_num} END ppl: {valid_ppl: 0.4f}')


if __name__ == '__main__':

    main()

