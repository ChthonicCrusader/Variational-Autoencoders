import tensorflow as tf 
import sonnet as snt
import matplotlib.pyplot as plt
from mlp.data_providers import MNISTDataProvider

class Encoder(snt.AbstractModule):
	def __init__(self, n_z, nonlinearity, name="encoder"):
		super(Encoder, self).__init__(name=name)
		# n_z = 2  no of latent variables
		self._nonlinearity = nonlinearity
		self._n_z = n_z

	def _build(self, input_):
		h_enc = self._nonlinearity(snt.Linear(256)(input_))
		mu = snt.Linear(self._n_z)(h_enc)
		std = snt.Linear(self._n_z)(h_enc)

		eps = tf.distributions.Normal(mu, std)
		return eps

class Decoder(snt.AbstractModule):
	def __init__(self, output_dim, nonlinearity, prior, name="decoder"):
		super(Decoder, self).__init__(name=name)
		self._nonlinearity = nonlinearity
		self._output_dim = output_dim
		self._prior = prior

	def _build(self, input_):
		z = input_
		h_enc = self._nonlinearity(snt.Linear(256)(z))
		output = snt.Linear(self._output_dim)(h_enc)
		p_x = tf.distributions.Bernoulli(output)
		return p_x

class VAE(snt.AbstractModule):
	def __init__(self, encoder, decoder, name="vae"):
		super(VAE, self).__init__(name=name)
		self._encoder = encoder
		self._decoder = decoder

	def reconstruct(self, input_, mean=True):
		q = self._encoder(input_)
		z = q.sample()
		p = self._decoder(z)
		if mean:
			return p.mean()
		else:
			return p.sample()

	def _build(self, input_):
		return self.reconstruct(input_)

	def log_prob_error(self, input_):
		q = self._encoder(input_)
		
		kl = tf.reduce_sum(tf.distributions  \
			   .kl_divergence(q, self._decoder._prior),axis=1)

		z = q.sample()
		p = self._decoder(z)
		log_prob =  tf.reduce_sum((p.log_prob(input_)), axis=1)
		
		return kl - log_prob

train_data = MNISTDataProvider('train',batch_size = 10)

inputs = tf.placeholder(tf.float32,[None,train_data.inputs.shape[1]], 'inputs')

enc = Encoder(64, tf.nn.relu)
dec = Decoder(784,tf.nn.relu, tf.distributions.Normal(0.,1.))

vae = VAE(enc,dec)

error = tf.reduce_mean(vae.log_prob_error(inputs))

tf.summary.scalar('error',error)

global_step = tf.train.get_or_create_global_step()

overall_summary = tf.summary.merge_all()

train_step = tf.train.AdamOptimizer(learning_rate=0.001)  \
                     .minimize(error, global_step=global_step)


summary_hook = tf.train.SummarySaverHook(
    save_steps=1,
    output_dir="./logs/",
    summary_op=overall_summary)

iterations = 2000
x_recon = vae.reconstruct(inputs)

print("Start model...")
with tf.train.MonitoredSession(hooks=[summary_hook]) as sess:
  for step in range(iterations):
    input_, _ = train_data.next()
    train_error, _ = sess.run([error, train_step], feed_dict={inputs:input_})
    if step % 100 == 0:
    	print(train_error)
  
  x_sample, _ = train_data.next()  
  x_reconstruct = sess.run(x_recon,feed_dict={inputs:x_sample})


plt.figure(figsize=(8, 12))
path='./out/'
for i in range(3):

    # plt.subplot(5, 2, 2*i + 1)
    plt.imshow(x_sample[i].reshape(28, 28), cmap='gray')
    plt.title("Test input")
    plt.savefig(path+str(i)+'_original.png')
    plt.close()
    # plt.subplot(5, 2, 2*i + 2)
    plt.imshow(x_reconstruct[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstruction")
    plt.savefig(path+str(i)+'_reconstruct.png')
    plt.close()
    #plt.colorbar()
