import tensorflow as tf

class Covid_pred_model(tf.keras.Model):

  def __init__(self,shape):

    super(Covid_pred_model, self).__init__()


    self.LSTM_layer1 = tf.keras.layers.LSTM(1024, 
                                            return_sequences=True,
                                            input_shape = shape)
    
    self.LSTM_layer2 = tf.keras.layers.LSTM(256)
    
    self.dense_layer1 = tf.keras.layers.Dense(512, activation='relu')
    self.dense_layer2 = tf.keras.layers.Dense(128, activation='relu')
    self.dense_layer3 = tf.keras.layers.Dense(64, activation='relu')
    self.dense_layer4 = tf.keras.layers.Dense(32, activation='relu')

    self.dropout_layer1 = tf.keras.layers.Dropout(0.5)

    self.output_layer = tf.keras.layers.Dense(1)


  def call(self, inputs):

    x = self.LSTM_layer1(inputs)
    x = self.dense_layer1(x)

    x = self.LSTM_layer2(x)
    x = self.dense_layer2(x)

    x = self.dense_layer3(x)
    x = self.dense_layer4(x)

    x = self.dropout_layer1(x)

    output = self.output_layer(x)

    return output

