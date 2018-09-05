import tensorflow as tf
import time

logging = tf.logging
logging.set_verbosity(logging.INFO)


def log_msg(msg):
    logging.info(f'{time.ctime()}: {msg}')