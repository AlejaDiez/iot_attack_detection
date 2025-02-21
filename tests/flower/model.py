import tensorflow as tf


def keras_model():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(
                24,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            ),
            tf.keras.layers.Dense(
                10,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            ),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )
    return model
