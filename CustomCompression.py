class CustomCompression(Model):
"""Custom Compression class inheriting from tf.keras.Model. Implements Nonlinear
Transform Coding using neural network analysis and synthesis transforms. alpha : lambda parameter used for lagrangian RD traversal
analysis : analysis transform; tensorflow.keras.Sequential model synthesis : synthesis transform; tensorflow.keras.Sequential model"""
def __init__(self , alpha, analysis, synthesis): 
  super(CustomCompression, self).__init__() 
  self.analysis = analysis
  self.synthesis = synthesis
  self.alpha = alpha
  self.var = [] # Class attributes to hold latent variance and mean self.mu = []
  
def call(self, inputs, training=True): # Overwritten call function, usage: model(data) 
  x = self.analysis(inputs)
  if training: # Use additive uniform noise when training
    x = self.add_noise(x)
  else:  # Quantize during testing
    x = self.quantize(x) 
return self.synthesis(x)

def gaussian(self, latent): # Gaussian entropy model
  result = 1/2*tf.experimental.numpy.log2(2 * 3.1415926535 * self.var) +
          tf.math.pow((latent-tf.cast(self.mu, dtype=tf.float32)), 2)/(2 * self.var)*
          np.log2(2.718) return result
  
def add_noise(self, latent): # Add uniform noise to each dimension 
  return latent + tf.random.uniform(tf.shape(latent), -1/2, 1/2)
def quantize(self, latent): # Quantize to the nearest integer
  return tf.math.floor(latent + 1/2)
def compression_loss(self, input, output): # Compute loss on a training batch 
  distortion = tf.reduce_mean(tf.square(input - output))
  latent = self.analysis(input)
  rate = tf.reduce_mean(self.gaussian(self.add_noise(latent)))
return rate, distortion

def train_step(self, x): # Compute loss of the training data x
  with tf.GradientTape() as tape:
    y = self(x, training=True)
    rate, distortion = self.compression_loss(x, y)
    loss = rate + self.alpha * distortion
# Compute and apply gradients to the trainable parameters
  gradients = tape.gradient(loss, self.trainable_variables) 
  self.optimizer.apply_gradients(zip(gradients, self.trainable_variables)) 
  return
  
def train(self, train_data, num_epochs, batch_size=32, lr=0.001): 
  optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
  for epoch in range(num_epochs):
    latent = self.analysis(train_data)
    self.mu = tf.math.reduce_mean(latent, 0) 
    self.var = tf.math.reduce_variance(latent, 0) 
    for i in range(0, len(train_data), batch_size):
      train_batch = train_data[i:i+batch_size] 
      self.train_step(train_batch)
  return 




