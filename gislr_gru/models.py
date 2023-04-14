import tensorflow as tf

def get_gru_baseline(in_shape, learning_rate):
    # Define model architecture
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.GRU(units=512, input_shape=in_shape, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.GRU(units=512))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(units=512))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(units=250, activation='softmax'))

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy'])
    model.build((None, *in_shape))
    model.summary()
    return model