import tensorflow as tf
import time


class Optimizer:
    def __init__(self,
                 model,
                 mb=8,
                 lr=0.001,
                 loss=tf.keras.losses.BinaryCrossentropy(),
                 opt=tf.keras.optimizers.Adam):

        self.model = model
        self.loss = loss(from_logits=False)
        self.optimizer = opt(learning_rate=lr)
        self.mb = mb
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

        self.train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy', threshold=0.5)
        self.test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy', threshold=0.5)

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.model(x)
            loss = self.loss(y, predictions)
        # Calculate gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # Back propagate through the model
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(y, predictions)

    @tf.function
    def test_step(self, x, y):
        predictions = self.model(x)
        loss = self.loss(y, predictions)
        self.test_loss(loss)
        self.test_accuracy(y, predictions)

    def train(self):
        for mbX, mbY in self.train_ds:
            self.train_step(mbX, mbY)

    def test(self):
        for mbX, mbY in self.test_ds:
            self.test_step(mbX, mbY)

    def run(self, dataX, dataY, testX, testY, epochs, verbose=2):
        historyTR_loss = []
        historyTS_loss = []

        historyTR_acc = []
        historyTS_acc = []

        self.train_ds = tf.data.Dataset.from_tensor_slices((dataX, dataY)).shuffle(len(dataX)).batch(self.mb)
        self.test_ds = tf.data.Dataset.from_tensor_slices((testX, testY)).batch(self.mb)

        for i in range(epochs):
            start = time.time()
            self.train()
            self.test()

            if verbose > 0:
                print("epoch: " + str(i + 1) + " TRAIN LOSS: " + str(self.train_loss.result()) + \
                      " TEST LOSS: " + str(self.test_loss.result()) + " TRAIN ACC: " + \
                      str(self.train_accuracy.result()) + " TEST ACC: " + str(self.test_accuracy.result()) + \
                      "Time: " + str(round(time.time() - start, 2)))

            historyTR_loss.append(float(self.train_loss.result()))
            historyTS_loss.append(float(self.test_loss.result()))
            historyTR_acc.append(float(self.train_accuracy.result()))
            historyTS_acc.append(float(self.test_accuracy.result()))

            self.train_loss.reset_states()
            self.test_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_accuracy.reset_states()
        return historyTR_loss, historyTR_acc, historyTS_loss, historyTS_acc
