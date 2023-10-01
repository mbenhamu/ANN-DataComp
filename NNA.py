# The following code details the neural network architecture used in the final model.

analysis = tf.keras.Sequential([
  layers.Lambda(lambda x : x / 255.),
  layers.Conv2D(filters=16, kernel_size=3, strides=1, activation=tfc.GDN), layers.MaxPooling2D(),
  layers.Conv2D(filters=32, kernel_size=3, strides=1, activation=tfc.GDN), layers.MaxPooling2D(),
  layers.Conv2D(filters=64, kernel_size=3, strides=1, activation=tfc.GDN), layers.MaxPooling2D(),
  layers.Conv2D(filters=128, kernel_size=3, strides=1, activation=tfc.GDN), layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dropout(0.5),
  layers.Dense(2048),
  layers.LeakyReLU()
])
synthesis = tf.keras.Sequential([
  layers.Reshape((4,4,128), input_shape=(2048,)),
  layers.Dropout(0.5),
  layers.UpSampling2D(),
  layers.Conv2DTranspose(filters=128, kernel_size=3, strides=1, activation=tfc.GDN(inverse), layers.UpSampling2D(),
  layers.Conv2DTranspose(filters=64, kernel_size=3, strides=1, activation=tfc.GDN(inverse)), layers.UpSampling2D(),
  layers.Conv2DTranspose(filters=32, kernel_size=3, strides=1, activation=tfc.GDN(inverse)), layers.UpSampling2D(),
  layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, activation=tfc.GDN(inverse)), layers.Reshape((64, 64, 1)),
  layers.Lambda(lambda x : x * 255.)
])
