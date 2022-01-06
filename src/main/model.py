import tensorflow as tf


class Dense_Block(tf.keras.layers.Layer):
    def __init__(self, units=9, dropout=False, activation=None):
        super(Dense_Block, self).__init__()
        self.Dense = tf.keras.layers.Dense(units=units,
                                           use_bias=False,
                                           activation=activation)
        if dropout:
            self.Dropout = tf.keras.layers.Dropout(rate=0.25)
            self.model = tf.keras.Sequential([self.Dense, self.Dropout])
        else:
            self.model = tf.keras.Sequential([self.Dense])

    def call(self, input_tensor, training=False):
        return self.model(input_tensor)


class Network(tf.keras.Model):
    def __init__(self,
                 units=[9, 9, 5],
                 dropout=[True, True, False],
                 activation=["relu", "relu", "relu"]):
        super(Network, self).__init__()

        self.dense_net = self._create_dense_layers(units, dropout, activation)
        self.model = tf.keras.Sequential([dense_block for dense_block in self.dense_net])

    def call(self, input_tensor, training=False):
        return self.model(input_tensor)


    def _create_dense_layers(self, units, dropout, activation):
        dense_blocks = []
        for i in range(len(units)):
            dense_block_i = Dense_Block(units=units[i],
                                        dropout=dropout[i],
                                        activation=activation[i])
            dense_blocks.append(dense_block_i)
        # output layer
        dense_blocks.append(Dense_Block(units=1,
                                        dropout=False,
                                        activation="sigmoid"))

        return dense_blocks
