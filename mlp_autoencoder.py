import tensorflow as tf
import matplotlib.pyplot as plt
import random


class MLPAutoEncoder(object):
    def __init__(self, input_x_dim, input_y_dim, code_dim):
        self._input_x_dim = input_x_dim
        self._input_y_dim = input_y_dim
        self._code_dim = code_dim

        code_dim_2 = self._code_dim ** 2
        input_dim = self._input_x_dim * self._input_y_dim
        hidden_dim = int((input_dim + code_dim_2) / 2)
        self._params = {
            "enc_W0": tf.Variable(tf.random.normal((hidden_dim, input_dim))),
            "enc_b0": tf.Variable(tf.zeros((hidden_dim, 1))),
            "enc_W1": tf.Variable(tf.random.normal((code_dim_2, hidden_dim))),
            "enc_b1": tf.Variable(tf.zeros((code_dim_2, 1))),
            "dec_W0": tf.Variable(tf.random.normal((hidden_dim, code_dim_2))),
            "dec_b0": tf.Variable(tf.zeros((hidden_dim, 1))),
            "dec_W1": tf.Variable(tf.random.normal((input_dim, hidden_dim))),
            "dec_b1": tf.Variable(tf.zeros((input_dim, 1))),
        }

    def encode(self, inputs):
        reshaped_inputs = tf.reshape(inputs, (len(inputs), self._input_x_dim * self._input_y_dim))
        hidden_layer = tf.nn.leaky_relu(tf.matmul(self._params["enc_W0"], reshaped_inputs, transpose_b=True)
                                        + self._params["enc_b0"])
        encoded = tf.transpose(tf.nn.sigmoid(tf.matmul(self._params["enc_W1"], hidden_layer)
                                             + self._params["enc_b1"]))
        return encoded

    def decode(self, encoded):
        hidden_layer = tf.nn.leaky_relu(tf.matmul(self._params["dec_W0"], encoded, transpose_b=True)
                                        + self._params["dec_b0"])
        decoded = tf.transpose(tf.nn.sigmoid(tf.matmul(self._params["dec_W1"], hidden_layer)
                                             + self._params["dec_b1"]))
        reshaped_outputs = tf.reshape(decoded, (len(decoded), self._input_x_dim, self._input_y_dim))
        return reshaped_outputs

    def loss(self, inputs):
        inputs_non_sparsity = tf.reduce_mean(tf.reduce_sum(inputs, axis=1))
        encoded = self.encode(inputs)
        #non_sparsity_penalty = tf.tanh(tf.reduce_mean(tf.reduce_sum(encoded, axis=1)) / self._code_dim - 1)
        decoded = self.decode(encoded)
        mean_square_error = tf.reduce_mean(tf.losses.mean_squared_error(inputs, decoded))
        outputs_non_sparsity = tf.reduce_mean(tf.reduce_sum(decoded, axis=1))
        non_sparsity_penalty = tf.tanh((inputs_non_sparsity - outputs_non_sparsity) ** 2)
        return (mean_square_error + non_sparsity_penalty) / 2

    def _training_step(self, inputs, learning_rate):
        with tf.GradientTape() as t:
            current_loss = self.loss(inputs)
        grads = t.gradient(current_loss, self._params)
        for param in self._params.keys():
            self._params[param].assign_sub(learning_rate * grads[param])

    def train(self, inputs, num_epochs, batch_size, learning_rate):
        for epoch_num in range(1, num_epochs + 1):
            for batch_num in range(int(inputs.shape[0] / batch_size)):
                inputs_batch = inputs[batch_num * batch_size: batch_num * batch_size + batch_size]
                self._training_step(inputs_batch, learning_rate)
            print("Epoch {} of {} completed, loss = {}".format(epoch_num, num_epochs, self.loss(inputs)))


def main():
    assert tf.version.VERSION >= "2.0.0", "TensorFlow 2.0.0 or newer required, %s installed" % tf.version.VERSION

    # Load the dataset, like last time
    # This time we're not gonna use the y data (because we're making an autoencoder not a classifier)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    print("\n{} examples in the training set, {} examples in the test set".format(x_train.shape[0], x_test.shape[0]))

    assert x_train.shape[1] == x_test.shape[1]
    assert x_train.shape[2] == x_test.shape[2]
    input_x_dim = x_train.shape[1]
    input_y_dim = x_train.shape[2]
    print("\nInput dimension: {}x{}\n".format(input_x_dim, input_y_dim))

    hidden_code_dim = 16
    model = MLPAutoEncoder(input_x_dim, input_y_dim, hidden_code_dim)
    model.train(x_train, 50, 128, 0.1)

    plt.title("Input Image")
    test_case = x_test[random.randrange(x_test.shape[0])]
    plt.imshow(test_case, cmap='Greys')
    plt.show()

    plt.title("Hidden Representation")
    encoded = model.encode([test_case])
    plt.imshow(tf.reshape(encoded[0], (hidden_code_dim, hidden_code_dim)) * 255.0, cmap='Greys')
    plt.show()

    plt.title("Output Image")
    decoded = model.decode([encoded])
    plt.imshow(decoded[0] * 255.0, cmap='Greys')
    plt.show()


if __name__ == "__main__":
    main()
