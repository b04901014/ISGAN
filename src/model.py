import tensorflow as tf
from collections import namedtuple
from utils import *
from module import *
import os
from scipy.io.wavfile import write
from ops import *
import traceback
from tqdm import tqdm
import sys
import itertools
from tensorflow.python import debug as tf_debug
from hparams import args
from sklearn.manifold import TSNE

class UVTGAN():
   def __init__(self, sess):
      self.sess = sess
      self.time_step = args.time_step
      self.sample_rate = args.sample_rate
      self.batch_size = args.batch_size
      print ("Time Span for audio : %.3f sec" %(float(self.time_step) * 0.005))
      self.vector_length = args.vector_length
      self.channel_size = args.channel_size
      self.Lambda = args.Lambda
      self.fp16_scale = args.fp16_scale if args.use_fp16 else 1.0
      self.use_fp16 = args.use_fp16
      if args.is_training:
         self.data_generator = multiproc_reader()
         self.procs = self.data_generator.start_enqueue()
      self.sample_num = args.sample_num
      self.sample_data_generator = sample_data_generator()
      self.test_data_generator = test_data_generator()
      self.gen_options = make_options("generator",
                                       "g",
                                       args.gen_conv_filter_size,
                                       args.gen_deconv_filter_size, 
                                       args.gen_conv_channel_size,
                                       args.gen_deconv_channel_size, 
                                       args.gen_stride, 
                                       args.gen_stridec, 
                                       args.gen_time_filter_size,
                                       args.gen_use_batch_norm)
      self.gen_options2d = make_options("generator",
                                       "2d",
                                       args.gen_conv_filter_size2d,
                                       args.gen_deconv_filter_size2d, 
                                       args.gen_conv_channel_size2d,
                                       args.gen_deconv_channel_size2d, 
                                       args.gen_stride2d, 
                                       args.gen_stride2dc, 
                                       args.gen_time_filter_size2d,
                                       args.gen_use_batch_norm)
      self.dis_options = make_options("discriminator",
                                       "d",
                                       args.dis_filter_size, 
                                       args.dis_channel_size, 
                                       args.dis_stride, 
                                       args.dis_use_batch_norm)
      self.dis_options2d = make_options("discriminator",
                                        "2d",
                                        args.dis_filter_size2d, 
                                        args.dis_channel_size2d, 
                                        args.dis_stride2d, 
                                        args.dis_use_batch_norm)
      try:
         self._build_model()
      except:
         for x in self.procs:
            x.terminate()
         traceback.print_exc()

   def _build_model(self):
      self.global_step_gen = tf.get_variable('global_step_generator', initializer=0, dtype=tf.int32, trainable=False)
      self.global_step_enc = tf.get_variable('global_step_encoder', initializer=0, dtype=tf.int32, trainable=False)
      self.global_step_dis = tf.get_variable('global_step_discriminator', initializer=0, dtype=tf.int32, trainable=False)
      self.realA = tf.placeholder(tf.float32, [None, self.channel_size, self.time_step, self.vector_length])
      self.realB = tf.placeholder(tf.float32, [None, self.channel_size, self.time_step, self.vector_length])
      self.sampleA = tf.placeholder(tf.float32, [None, self.channel_size, self.time_step, self.vector_length])
      self.seqlenA = tf.placeholder(tf.int32, [None])
      self.seqlenB = tf.placeholder(tf.int32, [None])
      self.mean_a, self.mean_b = tf.constant(mean_a, dtype=tf.float32), tf.constant(mean_b, dtype=tf.float32)
      self.std_a, self.std_b = tf.constant(std_a, dtype=tf.float32), tf.constant(std_b, dtype=tf.float32)
      self.placeholders = [self.realA, self.realB, self.seqlenA, self.seqlenB]
      if args.paired_training:
         self.realP = tf.placeholder(tf.float32, [None, self.channel_size, self.time_step, self.vector_length])
         self.seqlenP = tf.placeholder(tf.int32, [None])
         self.maskP = makemask(self.seqlenP, self.time_step) #(64, 128)
         self.placeholders = [self.realA, self.realB, self.realP, self.seqlenA, self.seqlenB, self.seqlenP]
      self.maskA = makemask(self.seqlenA, self.time_step) #(64, 128)
      self.maskB = makemask(self.seqlenB, self.time_step) #(64, 128)
      self.samplemaskA = makemask(self.seqlenA, tf.shape(self.sampleA)[2])
      self.fakeB, _ = unet_generator2d(self.realA,
                                  self.maskA,
                                  self.seqlenA,
                                  self.gen_options2d,
                                  train=True,
                                  scope='generator',
                                  reuse=False)
      self.fakeBB, _ = unet_generator2d(self.realB,
                                   self.maskB,
                                   self.seqlenB,
                                   self.gen_options2d,
                                   train=True,
                                   scope='generator',
                                   reuse=True)
      self.sample_B, self.sample_c = unet_generator2d(self.sampleA,
                                     self.samplemaskA,
                                     self.seqlenA,
                                     self.gen_options2d,
                                     train=False,
                                     scope='generator',
                                     reuse=True)

#      self.fakeBe = (tf.reduce_sum((self.fakeB + args.clip_to_value) ** 2, axis=2, keepdims=True) / tf.cast(tf.reduce_sum(self.maskA), tf.float32)) ** 0.5
#      self.realAe = (tf.reduce_sum((self.realA + args.clip_to_value) ** 2, axis=2, keepdims=True) / tf.cast(tf.reduce_sum(self.maskA), tf.float32)) ** 0.5
#      self.fakeB = (self.fakeB + args.clip_to_value) * (self.realAe / (self.fakeBe + 1e-5)) - args.clip_to_value
#      self.fakeB = tf.clip_by_value(self.fakeB, -args.clip_to_value, args.clip_to_value)
#      self.sampleAe = (tf.reduce_sum((self.sampleA + args.clip_to_value) ** 2, axis=2, keepdims=True) / tf.cast(tf.reduce_sum(self.maskA), tf.float32)) ** 0.5
#      self.sampleBe = (tf.reduce_sum((self.sample_B + args.clip_to_value) ** 2, axis=2, keepdims=True) / tf.cast(tf.reduce_sum(self.maskA), tf.float32)) ** 0.5
#      self.sample_B = (self.sample_B + args.clip_to_value) * (self.sampleAe / (self.sampleBe + 1e-5)) - args.clip_to_value
#      self.sample_B = tf.clip_by_value(self.sample_B, -args.clip_to_value, args.clip_to_value)
      self.sample_B = tf.transpose(self.sample_B, perm=[0, 3, 2, 1])
      #enfocing consistency
#      self.realAclip = self.realA_reshaped[:, int(self.time_step / 2): -int(self.time_step / 2)]
#      self.realA_backinput = tf.reshape(self.realAclip, [b, s - 1, 1, t, f])
#      self.fakeB_all_shift = full_generator2d(self.realA_backinput,
#                                        self.seqlenA - 1,
#                                        self.gen_options2d,
#                                        train=True,
#                                        scope='generator',
#                                        reuse=True)
#      self.fakeB_all_clip = self.fakeB_all[:, int(self.time_step / 2): -int(self.time_step / 2)]
#      self.constriant_loss = tf.reduce_mean(tf.abs(self.fakeB_all_clip - self.fakeB_all_shift))
      #discriminator part
      self.DfakeB, self.flfB = discriminator2d(self.fakeB, self.maskA, self.dis_options2d, scope='discriminator1', reuse=False)
      _, self.flfBB = discriminator2d(self.fakeBB, self.maskB, self.dis_options2d, scope='discriminator1', reuse=True)
      self.DrealB, self.flrB = discriminator2d(self.realB, self.maskB, self.dis_options2d, scope='discriminator1', reuse=True)
      _, self.flrA = discriminator2d(self.realA, self.maskA, self.dis_options2d, scope='discriminator1', reuse=True)
#      self.flfBB = discriminator(self.fakeBB, self.maskB, self.dis_options, only_first=True, scope='discriminator1', reuse=True)
#      num_activated = tf.cast(tf.reduce_sum(self.maskA), tf.float32) * \
#                      tf.cast(tf.shape(self.flrA)[1], tf.float32) * \
#                      tf.cast(tf.shape(self.flrA)[3], tf.float32)
#      num_activatedB = tf.cast(tf.reduce_sum(self.maskB), tf.float32) * \
#                       tf.cast(tf.shape(self.flrA)[1], tf.float32) * \
#                       tf.cast(tf.shape(self.flrA)[3], tf.float32)
#      self.DrealA = discriminator(self.realA, self.dis_options, 1.0, scope='discriminator1', reuse=True)
      #generator loss
#      self.fakeB1, self.fakeB2 = self.fakeB[: self.batch_size // 2], self.fakeB[-self.batch_size // 2: ]
#      self.diff_loss = tf.reduce_sum(tf.abs(self.fakeB1 - self.fakeB2) * self.maskA) / (num_activated + 1e-8)
#      self.diff_loss = tf.maximum(-self.diff_loss, -2.0)
      
      epsilon = tf.random_uniform([], 0.0, 1.0)
      self.fakeB_hat = epsilon * self.realB + (1 - epsilon) * self.fakeB
      self.DfakeB_hat, _ = discriminator2d(self.fakeB_hat, self.maskB * self.maskA, self.dis_options2d, scope='discriminator1', reuse=True)
      ddx = tf.gradients(self.DfakeB_hat, self.fakeB_hat)[0]
      ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=[1, 2, 3]))
      self.ddx = tf.reduce_mean(tf.square(ddx - 1.0) * 1.)

      self.logit_loss = tf.reduce_mean(self.DfakeB) if not args.use_lsgan else l2_metric(self.DfakeB, 1.0)
      
#      self.mean_loss = tf.reduce_mean((tf.reduce_mean(self.fakeB, axis=2) - tf.reduce_mean(self.realA, axis=2)) ** 2)
#      self.rec_loss =  tf.reduce_sum([(tf.reduce_mean(tf.abs(standardize(a) - standardize(b)) / 2 ** (2 * i))) \
#                          for i, (a, b) in enumerate(zip(self.flrB, self.flfBB))])
#      self.rec_loss = tf.reduce_mean(tf.abs(self.fakeBB - self.realB))
      self.gen_loss = self.logit_loss# + self.rec_loss
#      self.deltafB, self.deltarA = timediff(self.fakeB), timediff(self.realA)
#      self.ddeltafB, self.ddeltarA = timediff(self.deltafB), timediff(self.deltarA)
#      self.enc_loss = #tf.reduce_sum((self.fakeB - self.realA) ** 2) / (num_activated + 1e-8) + \
#      self.enc_loss = (tf.reduce_sum((self.deltarA - self.deltafB) ** 2) / (num_activated + 1e-8)) ** 0.5 + \
      self.enc_loss = tf.reduce_sum([(tf.reduce_mean(tf.abs(a - b) / 2 ** (2 * i))) \
                          for i, (a, b) in enumerate(zip(self.flrA, self.flfB))])
      if args.paired_training:
         _, self.flrP = discriminator2d(self.realP, self.maskP, self.dis_options2d, scope='discriminator1', reuse=True)
         self.enc_loss += tf.reduce_sum([(tf.reduce_mean(tf.abs(a - b) / 2 ** (2 * i))) \
                          for i, (a, b) in enumerate(zip(self.flrP, self.flfB))])
#      self.mfcc_a = tf.transpose(self.realA, perm=[0, 3, 2, 1]) * self.std_a + self.mean_a
#      self.mfcc_a = tf.squeeze(self.mfcc_a, axis=1)
#      self.mfcc_a = tf.contrib.signal.mfccs_from_log_mel_spectrograms(self.mfcc_a)[:, :, : 13]
#      ma, va = tf.nn.moments(self.mfcc_a, axes=[1], keep_dims=True)
#      self.mfcc_a = (self.mfcc_a - ma) / (va + 1e-8) ** 0.5
#      self.mfcc_b = tf.transpose(self.fakeB, perm=[0, 3, 2, 1]) * self.std_b + self.mean_b
#      self.mfcc_b = tf.squeeze(self.mfcc_b, axis=1)
#      self.mfcc_b = tf.contrib.signal.mfccs_from_log_mel_spectrograms(self.mfcc_b)[:, :, : 13]
#      mb, vb = tf.nn.moments(self.mfcc_b, axes=[1], keep_dims=True)
#      self.mfcc_b = (self.mfcc_b - mb) / (vb + 1e-8) ** 0.5
#      self.enc_loss = tf.reduce_mean(tf.abs(self.mfcc_a - self.mfcc_b))
#      self.enc_loss *= 10
#                       (tf.reduce_mean(tf.abs(self.realA - self.fakeB)))
#                      tf.reduce_sum((self.ddeltafB - self.ddeltarA) ** 2) / (num_activated + 1e-8)
#      self.enc_loss = l2_metric(self.DfakeB, self.DrealA)
#      self.enc_loss = tf.reduce_mean(tf.abs(self.fakeB - self.realA))
#      self.enc_loss *= 5
      #discriminator loss
      self.D_logit_loss_T = tf.reduce_mean(self.DrealB) if not args.use_lsgan else l2_metric(self.DrealB, 1.0)
      self.D_logit_loss_F = -tf.reduce_mean(self.DfakeB) if not args.use_lsgan else l2_metric(self.DfakeB, 0.0)
      self.dis_loss = self.D_logit_loss_F + self.D_logit_loss_T# + self.ddx

      t_vars = tf.trainable_variables()
      self.gen_vars = [var for var in t_vars if 'decoder' in var.name]
      self.dis_vars = [var for var in t_vars if 'discriminator' in var.name]
      self.enc_vars = [var for var in t_vars if 'encoder' in var.name]
      print ([v.name for v in self.gen_vars])
      print ([v.name for v in self.dis_vars])
      print ([v.name for v in self.enc_vars])
#      self.enc_loss = tf.Print(self.enc_loss, [self.enc_loss])

      #for samples
      self.numpara_gen, self.numpara_dis, self.numpara_enc = 0, 0, 0
      for var in self.gen_vars:
         varshape = var.get_shape().as_list()
         self.numpara_gen += np.prod(varshape)
      for var in self.dis_vars:
         varshape = var.get_shape().as_list()
         self.numpara_dis += np.prod(varshape)
      for var in self.enc_vars:
         varshape = var.get_shape().as_list()
         self.numpara_enc += np.prod(varshape)
      print ("Total number of parameters in generator: %r" %(self.numpara_gen))
      print ("Total number of parameters in attacker: %r" %(self.numpara_enc))
      print ("Total number of parameters in discriminator: %r" %(self.numpara_dis))

   def train(self):
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      adam_ep = 1e-4 if self.use_fp16 else 1e-8
      self.gen_optimizer = tf.train.AdamOptimizer(learning_rate=args.gen_lr, beta1=0.5, beta2=0.9, epsilon=adam_ep)
      self.dis_optimizer = tf.train.AdamOptimizer(learning_rate=args.dis_lr, beta1=0.5, beta2=0.9, epsilon=adam_ep)
      self.enc_optimizer = tf.train.AdamOptimizer(learning_rate=args.enc_lr, beta1=0.5, beta2=0.9, epsilon=adam_ep)
      with tf.control_dependencies(update_ops):
         with tf.control_dependencies([tf.check_numerics(self.gen_loss, 'nan')]):
            self.gen_loss = tf.identity(self.gen_loss)
         with tf.control_dependencies([tf.check_numerics(self.dis_loss, 'nan')]):
            self.dis_loss = tf.identity(self.dis_loss)
         with tf.control_dependencies([tf.check_numerics(self.enc_loss, 'nan')]):
            self.enc_loss = tf.identity(self.enc_loss)
         self.gen_grad = self.gen_optimizer.compute_gradients(self.gen_loss * self.fp16_scale, var_list=self.gen_vars)
         self.dis_grad = self.dis_optimizer.compute_gradients(self.dis_loss * self.fp16_scale, var_list=self.dis_vars)
         self.enc_grad = self.enc_optimizer.compute_gradients(self.enc_loss * self.fp16_scale, var_list=self.enc_vars)
#         with tf.control_dependencies([tf.check_numerics(x[0], 'nan') for x in self.gen_grad]):
#            self.gen_grad[0] = (tf.identity(self.gen_grad[0][0]), self.gen_grad[0][1])
#         with tf.control_dependencies([tf.check_numerics(x[0], 'nan') for x in self.dis_grad]):
#            self.dis_grad[0] = (tf.identity(self.dis_grad[0][0]), self.dis_grad[0][1])
#         with tf.control_dependencies([tf.check_numerics(x[0], 'nan') for x in self.enc_grad]):
#            self.enc_grad[0] = (tf.identity(self.enc_grad[0][0]), self.enc_grad[0][1])
#         self.gen_grad = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.gen_grad if grad is not None]
#         self.dis_grad = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.dis_grad if grad is not None]
#         self.end_grad = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.enc_grad if grad is not None]
         self.gen_grad_norm = tf.global_norm([grad for grad, var in self.gen_grad]) / self.numpara_gen * args.gen_lr
         self.enc_grad_norm = tf.global_norm([grad for grad, var in self.enc_grad]) / self.numpara_enc * args.enc_lr
         self.dis_grad_norm = tf.global_norm([grad for grad, var in self.dis_grad]) / self.numpara_dis * args.dis_lr
#         self.stand_grad_norm = tf.maximum(self.dis_grad_norm, 1e-13)
#         self.norm_ratio = self.stand_grad_norm / (self.dis_grad_norm + 1e-80) * 1
#         self.dis_grad = [(self.norm_ratio * grad, var) for grad, var in self.dis_grad if grad is not None]
#         self.norm_ratio = self.dis_grad_norm / (self.gen_grad_norm + 1e-80) * 1
#         self.gen_grad = [(self.norm_ratio * grad, var) for grad, var in self.gen_grad if grad is not None]
#         self.norm_ratio = self.dis_grad_norm / (self.enc_grad_norm + 1e-80) * 1
#         self.enc_grad = [(self.norm_ratio * grad, var) for grad, var in self.enc_grad if grad is not None]
#         self.gen_grad_norm = tf.global_norm([grad for grad, var in self.gen_grad]) / self.numpara_gen * args.gen_lr
#         self.enc_grad_norm = tf.global_norm([grad for grad, var in self.enc_grad]) / self.numpara_enc * args.enc_lr
#         self.dis_grad_norm = tf.global_norm([grad for grad, var in self.dis_grad]) / self.numpara_dis * args.dis_lr
         self.gen_grad = [(grad / self.fp16_scale, var) for grad, var in self.gen_grad if grad is not None]
         self.gen_grad = [(grad / self.fp16_scale, var) for grad, var in self.gen_grad if grad is not None]
         self.dis_grad = [(grad / self.fp16_scale, var) for grad, var in self.dis_grad if grad is not None]
         self.enc_grad = [(grad / self.fp16_scale, var) for grad, var in self.enc_grad if grad is not None]
         self.gen_op = self.gen_optimizer.apply_gradients(self.gen_grad, global_step=self.global_step_gen)
         self.dis_op = self.dis_optimizer.apply_gradients(self.dis_grad, global_step=self.global_step_dis)
         self.enc_op = self.enc_optimizer.apply_gradients(self.enc_grad, global_step=self.global_step_enc)
      varset = list(set(tf.global_variables()) | set(tf.local_variables()))
      self.saver = tf.train.Saver(var_list=varset, max_to_keep=8)
      num_batch = int(args.total_examples / self.batch_size)
      do_initialzie = True
      if args.loading_path:
         if self.load(args.loading_path):
            start_epoch = int(self.global_step_gen.eval() / num_batch)
            do_initialzie = False
      if do_initialzie:
         init_op = tf.global_variables_initializer()
         start_epoch = 0
         self.sess.run(init_op)
      self.writer = tf.summary.FileWriter(args.summary_dir, None)
      with tf.name_scope("summaries"):
         self.s_gen_loss = tf.summary.scalar('generator_loss', self.gen_loss)
         self.s_enc_loss = tf.summary.scalar('encoder_loss', self.enc_loss)
         self.s_dis_loss = tf.summary.scalar('discriminator_loss', self.dis_loss)
         self.s_gen_grad = tf.summary.scalar('generator_grad', self.gen_grad_norm)
         self.s_enc_grad = tf.summary.scalar('encoder_grad', self.enc_grad_norm)
         self.s_dis_grad = tf.summary.scalar('discriminator_grad', self.dis_grad_norm)
         self.gen_merged = tf.summary.merge([self.s_gen_loss, self.s_gen_grad])
         self.enc_merged = tf.summary.merge([self.s_enc_loss, self.s_enc_grad])
         self.dis_merged = tf.summary.merge([self.s_dis_loss, self.s_dis_grad])

      self.sample(args.sampling_path, start_epoch, self.sample_num)
      try:
         for epoch in range(start_epoch, args.epoch):
            loss_names = ["Generator Loss",
                          "DisLoss True",
                          "DisLoss False",
                          "Distance Loss"]
            buffers = buff(loss_names)
            for batch in tqdm(range(num_batch)):
               for i in range(args.num_train_gen):
                  input_data = self.data_generator.dequeue()
                  feed_dict = {a: b for a, b in zip(self.placeholders, input_data)}
                  _, gen_loss, gen_sum, gen_step = self.sess.run([self.gen_op,
                                                   self.gen_loss,
                                                   self.gen_merged,
                                                   self.global_step_gen],
                                                   feed_dict=feed_dict)
                  self.gate_add_summary(gen_sum, gen_step)
                  buffers.put([gen_loss], [0])
               for i in range(args.num_train_dis):
                  input_data = self.data_generator.dequeue()
                  feed_dict = {a: b for a, b in zip(self.placeholders, input_data)}
                  _, dis_loss_t, dis_loss_f, dis_sum, dis_step = self.sess.run([self.dis_op,
                                                   self.D_logit_loss_T,
                                                   self.D_logit_loss_F,
                                                   self.dis_merged,
                                                   self.global_step_dis],
                                                   feed_dict=feed_dict)
                  self.gate_add_summary(dis_sum, dis_step)
                  buffers.put([dis_loss_t, dis_loss_f], [1, 2])
               for i in range(args.num_train_enc):
                  input_data = self.data_generator.dequeue()
                  feed_dict = {a: b for a, b in zip(self.placeholders, input_data)}
                  _, dl2, enc_sum, enc_step = self.sess.run([self.enc_op,
                                          self.enc_loss,
                                          self.enc_merged,
                                          self.global_step_enc],
                                          feed_dict=feed_dict)
                  self.gate_add_summary(enc_sum, enc_step)
                  buffers.put([dl2], [3])
               if (batch + 1) % args.display_step == 0:
                  buffers.printout([epoch + 1, batch + 1, num_batch])
                  fd = True
            if (epoch + 1) % args.saving_epoch == 0 and args.saving_path:
               try :
                  self.save(args.saving_path, epoch + 1)
               except:
                  print ("Failed saving model, maybe no space left...")
            if (epoch + 1) % args.sample_epoch == 0 and args.sampling_path:
               self.sample(args.sampling_path, epoch + 1, self.sample_num)
      except KeyboardInterrupt:
         print ("KeyboardInterrupt")
      finally:
         for x in self.procs:
            x.terminate()

   def save(self, save_path, epoch):
      name = 'Model_Epoch_' + str(epoch)
      save_path = os.path.join(save_path, name)
      print("Saving Model to %r" %save_path)
      step = self.sess.run(self.global_step_gen)
      self.saver.save(self.sess, save_path, global_step=step)

   def load(self, load_path):
      ckpt = tf.train.get_checkpoint_state(load_path)
      if ckpt and ckpt.model_checkpoint_path:
         ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
         print ("Loading Model From %r" %os.path.join(load_path, ckpt_name))
         self.saver.restore(self.sess, os.path.join(load_path, ckpt_name))
         return True
      print ("Error Loading Model! Training From Initial State if in training phase...")
      return False

   def sample(self, save_path, epoch, sample_num):
#################
#     ipp = self.data_generator.dequeue()
#     feed_dict = {a: b for a, b in zip(self.placeholders, ipp)}
#     opp = self.sess.run(self.realB, feed_dict=feed_dict)
#     outputs = np.squeeze(opp, axis=3)
#     outputs = np.transpose(outputs, axes=[0, 2, 1])
#     outputdata = [x[: l] for x, l in zip(outputs, ipp[3])]
#     for idx, op in enumerate(outputdata):
#        try:
#           sub_path = "./tmp"
#           print ('Sampling tmp to %r ... %r' %(sub_path, idx))
#           fig, ax = plt.subplots()
#           msp2audio(os.path.join(sub_path, '%r_OR.wav' %idx),
#                     os.path.join(sub_path, 'Recog_OR.txt'),
#                     op,
#                     211,
#                     "Test",
#                     'b')
#           fig.savefig(os.path.join(sub_path, 'MelSpec%r.png' %(idx)))
#           plt.close(fig)
#        except:
#           print ("Failed sampling, maybe no space left...")
#           traceback.print_exc()
            
#################
      sample_data, sample_len = next(self.sample_data_generator)
      b, f, t, _ = sample_data.shape
      sample_data_in = np.transpose(sample_data, axes=[0, 2, 1, 3])
      sample_data_in = np.reshape(sample_data_in, [b * (t // args.time_step), args.time_step, args.channel_size, 1])
      sample_data_in = np.transpose(sample_data_in, axes=[0, 2, 1, 3])
#      sample_len_in = np.concatenate([len2list(x, t) for x in sample_len], axis=0)
      sample_len_in = np.zeros([sample_data_in.shape[0]]) + args.time_step
      assert sample_len_in.shape[0] == b * (t // args.time_step), (sample_len, sample_len_in, b, sample_len_in.shape[0])
      feed_dict = {self.sampleA: sample_data_in, self.seqlenA: sample_len_in}
      outputs = self.sess.run(self.sample_B, feed_dict=feed_dict)
      outputs = np.squeeze(outputs, axis=1)
      outputs = np.reshape(outputs, [b, t, f])
      outputdata = [x[: l] for x, l in zip(outputs, sample_len)]
      originaldata = np.transpose(sample_data, axes=[0, 3, 2, 1])
      originaldata = np.squeeze(originaldata, axis=1)
      originaldata = [x[: l] for x, l in zip(originaldata, sample_len)]
      for idx, (original, convert) in enumerate(zip(originaldata, outputdata)):
         try:
            name = 'Sample_Epoch_%r_%r' %(epoch, idx)
            sub_path = os.path.join(save_path, name)
            if not os.path.exists(sub_path):
               os.mkdir(sub_path)
            print ('Sampling to %r ...' %sub_path)
            fig, ax = plt.subplots()
            msp2audio(os.path.join(sub_path, '%r_OR.wav' %idx),
                      os.path.join(sub_path, 'Recog_OR.txt'),
                      original,
                      211,
                      "Original",
                      'a')
            msp2audio(os.path.join(sub_path, '%r_Gen.wav' %idx),
                      os.path.join(sub_path, 'Recog_Gen.txt'),
                      convert,
                      212,
                      "Transformed",
                      'b')
            fig.savefig(os.path.join(sub_path, 'MelSpec.png'))
            plt.close(fig)
         except:
            print ("Failed sampling, maybe no space left...")
            traceback.print_exc()
         
   def test(self):
      varset = list(set(tf.global_variables()) | set(tf.local_variables()))
      self.saver = tf.train.Saver(var_list=varset, max_to_keep=8)
      assert self.load(args.loading_path), "It's testing phase, thus the model path have to be correctedly specified."
      total_sum_c, length_c = 0, 0
      total_sum_o, length_o = 0, 0
      total_conditions = []
      for nb, (sample_data, sample_len) in enumerate(self.test_data_generator):
         b, f, t, _ = sample_data.shape
         sample_data_in = np.transpose(sample_data, axes=[0, 2, 1, 3])
         sample_data_in = np.reshape(sample_data_in, [b * (t // args.time_step), args.time_step, args.channel_size, 1])
         sample_data_in = np.transpose(sample_data_in, axes=[0, 2, 1, 3])
         sample_len_in = np.concatenate([len2list(x, t) for x in sample_len], axis=0)
         assert sample_len_in.shape[0] == b * (t // args.time_step), (sample_len, sample_len_in, b, sample_len_in.shape[0])
         feed_dict = {self.sampleA: sample_data_in, self.seqlenA: sample_len_in}
         outputs, conditions = self.sess.run([self.sample_B, self.sample_c], feed_dict=feed_dict)
         outputs = np.squeeze(outputs, axis=1)
         outputs = np.reshape(outputs, [b, t, f])
         outputdata = [x[: l] for x, l in zip(outputs, sample_len)]
         originaldata = np.transpose(sample_data, axes=[0, 3, 2, 1])
         originaldata = np.squeeze(originaldata, axis=1)
         originaldata = [x[: l] for x, l in zip(originaldata, sample_len)]
         total_conditions.append(conditions)
         for idx, (original, convert) in enumerate(zip(originaldata, outputdata)):
            try:
               numid = nb * args.batch_size + idx + 1
               name = 'Sample_%r' %(numid)
               sub_path = os.path.join(args.testing_path, name)
               if not os.path.exists(sub_path):
                  os.mkdir(sub_path)
               print ('Sampling to %r ...' %sub_path)
               fig, ax = plt.subplots()
               msp2audio(os.path.join(sub_path, '%r_OR.wav' %numid),
                         os.path.join(sub_path, 'Recog_OR.txt'),
                         original,
                         211,
                         "Original",
                         'a')
               msp2audio(os.path.join(sub_path, '%r_Gen.wav' %numid),
                         os.path.join(sub_path, 'Recog_Gen.txt'),
                         convert,
                         212,
                         "Transformed",
                         'b')
               fig.savefig(os.path.join(sub_path, 'MelSpec.png'))
               plt.close(fig)
               np.save(os.path.join(sub_path, 'msp_c.npy'), convert)
               np.save(os.path.join(sub_path, 'msp_o.npy'), original)
            except:
               print ("Failed sampling, maybe no space left...")
               traceback.print_exc()
      for fold in os.listdir(args.testing_path):
         if not os.path.isdir(os.path.join(args.testing_path, fold)):
            continue
         n = os.path.join(args.testing_path, fold, 'msp_c.npy')
         convert = np.load(n)
         total_sum_c += np.sum(convert, axis=0)
         length_c += convert.shape[0]
         n = os.path.join(args.testing_path, fold, 'msp_o.npy')
         original = np.load(n)
         total_sum_o += np.sum(original, axis=0)
         length_o += original.shape[0]
      mean_c = total_sum_c / length_c
      mean_o = total_sum_o / length_o
      total_sum_c, total_sum_o = 0, 0
      for fold in os.listdir(args.testing_path):
         if not os.path.isdir(os.path.join(args.testing_path, fold)):
            continue
         n = os.path.join(args.testing_path, fold, 'msp_c.npy')
         convert = np.load(n)
         total_sum_c += np.sum((convert - mean_c) ** 2, axis=0)
         n = os.path.join(args.testing_path, fold, 'msp_o.npy')
         original = np.load(n)
         total_sum_o += np.sum((original - mean_o) ** 2, axis=0)
      std_c = (total_sum_c / length_c) ** 0.5 * std_b
      std_o = (total_sum_o / length_o) ** 0.5 * std_a
      np.save(os.path.join(args.testing_path, 'var_c'), std_c ** 2)
      np.save(os.path.join(args.testing_path, 'var_o'), std_o ** 2)
      fig, ax = plt.subplots()
      plt.plot(np.arange(std_c.shape[0]), std_c ** 2)
      plt.plot(np.arange(std_o.shape[0]), std_a ** 2)
      plt.plot(np.arange(std_b.shape[0]), std_b ** 2)
      plt.legend(['Generated', 'Oral Impaired', 'Normal'], loc='upper left')
      fig.savefig(os.path.join(args.testing_path, 'GV.png'))
      plt.close(fig)
      print ("Running TSNE")
      tsne = TSNE(n_components=2, random_state=0)
      total_conditions = np.concatenate(total_conditions, axis=0)
      print (total_conditions.shape)
      n = total_conditions.shape[0]
      s = total_conditions.shape[1] * total_conditions.shape[2] * total_conditions.shape[3]
      total_conditions = np.reshape(total_conditions, [n, s])
      total_conditions = tsne.fit_transform(total_conditions)
      tsne_path = os.path.join(args.tsne_path, os.path.basename(args.testing_path) + '.npy')
      np.save(tsne_path, total_conditions)
      print ("Done testing!")
 
   def gate_add_summary(self, summary, step):
      try:     
         self.writer.add_summary(summary, step)
      except:
         print ("Failed adding summary, maybe no space left...")
