import tensorflow as tf
import numpy as np
from collections import namedtuple
from hparams import args
#var_initializer = tf.contrib.layers.xavier_initializer()
var_initializer = tf.random_normal_initializer(stddev=0.02)
use_fp16 = args.use_fp16

def get_var(name,
            shape,
            dtype=tf.float32,
            initializer=None,
            trainable=True,
            spec_norm=False,
            use_fp16=False):
   ret = tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=initializer, trainable=trainable)
   if spec_norm:
      ret = spectral_norm(ret)
   if use_fp16:
      assert dtype == tf.float32, "Dtype should be float32. Get %s" %(dtype)
      ret = tf.cast(ret, tf.float16)
   return ret

def MAE(x, y):
   return tf.reduce_mean(tf.abs(x - y))

def MSE(x, y):
   return tf.reduce_mean((x - y) ** 2)

def standardize(x, epsilon=1e-5):
   m, v = tf.nn.moments(x, axes=[2, 3], keep_dims=True)
   ret = (x - m) * tf.rsqrt(v + epsilon)
#   return tf.clip_by_value(ret, -1., 1.)
   return ret

def leaky_relu(inputs, alpha=.2):
   outputs = tf.maximum(alpha * inputs, inputs)
   return outputs

def spectral_norm(w, niters=1):
   wr = tf.reshape(w, [-1, w.shape[-1]])
   uv = get_var("u", [wr.shape[0], 1], initializer=var_initializer, dtype=tf.float32, trainable=False)
   u = uv
   for _ in range(niters):
      v = tf.nn.l2_normalize(tf.matmul(tf.transpose(wr), u), axis=None)
      u = tf.nn.l2_normalize(tf.matmul(wr, v), axis=None)
   with tf.control_dependencies([tf.assign(uv, u)]):
      u = tf.identity(u)
   norm = tf.matmul(tf.transpose(u), wr)
   norm = tf.matmul(norm, v)
   return w / norm

def instance_norm(inputs, epsilon=1e-5, scope=None, reuse=None):
   b, d, h, w = inputs.get_shape().as_list()
   inputs = tf.cast(inputs, tf.float32)
   with tf.variable_scope(scope or 'inst_norm', reuse=reuse):
      scale = get_var("scale", [1, d, 1, 1], initializer=tf.random_normal_initializer(1., .02))
      offset = get_var("offset", [1, d, 1, 1], initializer=tf.zeros_initializer())
      m, var = tf.nn.moments(inputs, axes=[2, 3], keep_dims=True)
      normalized = (inputs - m) * tf.rsqrt(var + epsilon)
      ret = scale * normalized + offset
      if use_fp16:
         ret = tf.cast(ret, tf.float16)
      return ret

def instance_norm_with_mask(inputs, mask, epsilon=1e-5, scope=None, reuse=None):
   b, d, h, w = inputs.get_shape().as_list()
   inputs = tf.cast(inputs, tf.float32)
   mask = tf.cast(mask, tf.float32)
   numactivated = tf.reduce_sum(mask, axis=[2, 3], keep_dims=True) * w
   dtype = tf.float16 if use_fp16 else tf.float32
   with tf.variable_scope(scope or 'inst_norm', reuse=reuse):
      scale = get_var("scale", [1, d, 1, 1], initializer=tf.random_normal_initializer(1., .02))
      offset = get_var("offset", [1, d, 1, 1], initializer=tf.zeros_initializer())
      m = tf.reduce_sum(inputs, axis=[2, 3], keep_dims=True) / (numactivated + 1e-4)
      var = ((inputs - m) ** 2) * mask
      var = tf.reduce_sum(var, axis=[2, 3], keep_dims=True) / (numactivated + 1e-4)
      normalized = (inputs - m) * tf.rsqrt(var + epsilon)
      ret = scale * normalized + offset
      ret = ret * mask
      if use_fp16:
         ret = tf.cast(ret, tf.float16)
      return ret

def batch_norm(inputs, epsilon=1e-5, momentum=0.997, train=True, scope=None, reuse=None):
   return instance_norm(inputs=inputs, epsilon=epsilon, scope=scope, reuse=reuse)
#   b, c, h, w = inputs.get_shape().as_list()
#   b = tf.shape(inputs)[0]
#   inputs = tf.transpose(inputs, perm=[1, 0, 2, 3])
#   indscale = tf.range(start=1, limit=c + 1)
#   indscale = tf.reshape(indscale, [c, 1, 1, 1])
#   inputs = inputs * tf.cast(indscale, tf.float32)
#   inputs = tf.transpose(inputs, perm=[1, 0, 2, 3])
   return tf.layers.batch_normalization(inputs,
                                        axis=1,
                                        momentum=momentum,
                                        epsilon=epsilon,
                                        center=True,
                                        scale=True,
                                        fused=True,
                                        training=train,
                                        name=scope,
                                        reuse=reuse)
#   return tf.reshape(inputs, [b, c, h, w])
#   return tf.contrib.layers.batch_norm(inputs,
#                                       decay=momentum,
#                                       updates_collections=None,
#                                       epsilon=epsilon,
#                                       scale=True,
#                                       center=True,
#                                       is_training=train,
#                                       scope=scope,
#                                       reuse=reuse)


def mydense(inputs, n_hidden, spec_norm=False, addbias=True, scope=None, reuse=None):
   with tf.variable_scope(scope or 'dense', reuse=reuse):
      input_dim = inputs.get_shape().as_list()[-1]
      weights = get_var("weights", shape=[input_dim, n_hidden], initializer=var_initializer, spec_norm=spec_norm, use_fp16=use_fp16)
      if not addbias:
         return tf.matmul(inputs, weights)
      biases = get_var("biases", shape=[n_hidden], initializer=tf.zeros_initializer(), use_fp16=use_fp16)
      return tf.nn.bias_add(tf.matmul(inputs, weights), biases)

def mydeconv2d(inputs,
               output_channel,
               filter_size=[4, 4],
               strides=[1, 1, 2, 2],
               spec_norm=False,
               scope=None,
               reuse=None):
   if type(filter_size) == int:
      filter_size = [filter_size, filter_size]
   if len(strides) == 2:
      strides = [1, 1] + strides
   #inputs : inputs (batch_size, channel_num, spatial_dim1, spatial_dim2)
   #         output_channel int tensor or int
   #         filter_size and strides
   #output : (batch_size, output_channel, filter_size1, filter_size2)
   with tf.variable_scope(scope or 'deconv', reuse=reuse):
      input_shape = inputs.get_shape().as_list()
      input_shape[2] = tf.shape(inputs)[2]
      batch_size = tf.shape(inputs)[0]
      input_channel = input_shape[1]
      #(filter_size1, filter_size2, output_channel, input_channel)
      #(batch_size, filter_size1, filter_size2, output_channel)
      weight_shape = [filter_size[0], filter_size[1], output_channel, input_channel]
      output_shape = [batch_size, output_channel, input_shape[2], input_shape[3]]
      output_shape = [a * b if a is not None else None for a, b in zip(output_shape, strides)]
      weights = get_var("weights", shape=weight_shape, initializer=var_initializer, spec_norm=spec_norm, use_fp16=use_fp16)
      biases = get_var("biases", shape=[output_channel], initializer=tf.zeros_initializer(), use_fp16=use_fp16)
      if use_fp16 and inputs.dtype == tf.float32:
         inputs = tf.cast(inputs, tf.float16)
      deconv = tf.nn.conv2d_transpose(inputs, weights, output_shape=output_shape, strides=strides, data_format="NCHW")
      return tf.nn.bias_add(deconv, biases, data_format="NCHW")

def mydeconv1d(inputs,
               output_channel,
               filter_size=5,
               stride=2,
               spec_norm=False,
               scope=None,
               reuse=None):
   #example input : (batch_size, ch_size, width, 1)
   assert inputs.get_shape().as_list()[3] == 1
   filter_size = [filter_size, 1]
   strides = [1, 1, stride, 1]
   return mydeconv2d(inputs, output_channel, filter_size, strides, spec_norm, scope, reuse)

def myconv2d(inputs,
             output_channel,
             filter_size=[4, 4],
             strides=[1, 1, 2, 2],
             padding='SAME',
             spec_norm=False,
             scope=None,
             reuse=None):
   if type(filter_size) == int:
      filter_size = [filter_size, filter_size]
   if len(strides) == 2:
      strides = [1, 1] + strides
   #inputs : inputs (batch_size, channel_num, spatial_dim1, spatial_dim2)
   #         filter_size and strides
   #output : (batch_size, output_channel, filter_size1, filter_size2)
   with tf.variable_scope(scope or 'conv2d', reuse=reuse):
      input_channel = inputs.get_shape().as_list()[1]
      #(filter_size1, filter_size2, input_channel, output_channel)
      weight_shape = [filter_size[0], filter_size[1], input_channel, output_channel]
      weights = get_var("weights", shape=weight_shape, initializer=var_initializer, spec_norm=spec_norm, use_fp16=use_fp16)
      biases = get_var("biases", shape=[output_channel], initializer=tf.zeros_initializer(), use_fp16=use_fp16)
      if use_fp16 and inputs.dtype == tf.float32:
         inputs = tf.cast(inputs, tf.float16)
      conv = tf.nn.conv2d(inputs, weights, strides=strides, padding=padding, data_format="NCHW")
      return tf.nn.bias_add(conv, biases, data_format="NCHW")

def myconv1d(inputs,
             output_channel,
             filter_size=4,
             stride=2,
             padding='SAME',
             spec_norm=False,
             scope=None,
             reuse=None):
   #example input : (batch_size, ch_size, width, 1)
   assert inputs.get_shape().as_list()[3] == 1
   filter_size = [filter_size, 1]
   strides = [1, 1, stride, 1]
   return myconv2d(inputs, output_channel, filter_size, strides, padding, spec_norm, scope, reuse)

def dynamic_LSTM(inputs, n_hidden, scope=None, reuse=None):
   b, c, h, w = inputs.get_shape().as_list()
   assert w == 1
   b = tf.shape(inputs)[0]
   inputs = tf.transpose(inputs, perm=[2, 0, 1, 3])
   inputs = tf.reshape(inputs, [h, b, c])
   with tf.variable_scope(scope or "dynamic_LSTM", reuse=reuse):
      lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden)
      outputs, states = tf.nn.static_rnn(lstm_cell, tf.unstack(inputs), dtype=tf.float32)
   outputs = tf.transpose(tf.stack(outputs), [1, 2, 0])
   return tf.reshape(outputs, [b, n_hidden, h, w])

def dynamic_GRU_2D(inputs, spec_norm, scope=None, reuse=None):
   b, c, t, f = inputs.get_shape().as_list()
   b = tf.shape(inputs)[0]
#   inputs_reshaped = tf.transpose(inputs, perm=[2, 0, 3, 1])
   channel_weights = tf.reduce_mean(inputs, axis=[2, 3])
#   ret = tf.reduce_mean(inputs_reshaped, axis=3)
   with tf.variable_scope(scope or "dynamic_GRU_2D", reuse=reuse):
#      gru_cell = tf.contrib.rnn.GRUCell(f, name="grucell")
#      ret, _ = tf.nn.static_rnn(gru_cell, tf.unstack(ret), dtype=tf.float32)
      channel_weights = mydense(channel_weights, c, spec_norm=spec_norm, scope="dense", reuse=reuse)
      channel_weights = tf.nn.softmax(channel_weights)
      channel_weights = tf.reshape(channel_weights, [b, c, 1, 1])
      return channel_weights * inputs
#      ret = inputs_reshaped * tf.expand_dims(ret, 3)
#      return tf.transpose(ret, perm=[1, 3, 0, 2]) * channel_weights

def residual_1d(inputs, mask, filter_size=4, padding='SAME', spec_norm=False, scope=None, reuse=None):
   with tf.variable_scope(scope or 'res', reuse=reuse):
      b, d, w, h = inputs.get_shape().as_list()
      out = tf.nn.elu(
         instance_norm_with_mask(
            myconv1d(inputs, d * 2, filter_size, 1, padding, spec_norm, 'conv1d_0', reuse),
            mask=mask, scope='bn1', reuse=reuse))
      out = instance_norm_with_mask(
         myconv1d(out, d, filter_size, 1, padding, spec_norm, 'conv1d_1', reuse),
         mask=mask, scope='bn2', reuse=reuse)
      return inputs + out

def residual_2d(inputs, filter_size=4, padding='SAME', spec_norm=False, train=True, scope=None, reuse=None):
   with tf.variable_scope(scope or 'res', reuse=reuse):
      b, d, w, h = inputs.get_shape().as_list()
      out = tf.nn.relu(batch_norm(myconv2d(inputs,
                                           d * 2,
                                           filter_size,
                                           [1, 1, 1, 1],
                                           padding,
                                           spec_norm, 
                                           'conv1d_0',
                                           reuse), train=train, scope='bn1', reuse=reuse))
      out = batch_norm(myconv2d(out,
                                d,
                                filter_size,
                                [1, 1, 1, 1],
                                padding,
                                spec_norm,
                                'conv1d_1',
                                reuse), train=train, scope='bn2', reuse=reuse)
      return inputs + out

def flatten(inputs):
   b, c, h, w = inputs.get_shape().as_list()
   ret = tf.reshape(inputs, [-1, c * h * w])
   return ret

def make_options(tp, num, *args):
   if tp == "generator":
      optn = "gen_options" + str(num)
      optstrl = ["gen_conv_filter_size", "gen_deconv_filter_size", "gen_conv_channel_size",
                 "gen_deconv_channel_size", "gen_stride", "gen_stridec", "gen_time_filter_size", "gen_use_batch_norm"]
      optstr = ""
      for s in optstrl:
         optstr += (s + ' ')
      optstr = optstr[: -1]
      ret = namedtuple(optn, optstr)
      return ret._make(args)
   elif tp == "discriminator":
      optn = "dis_options" + str(num)
      optstrl = ["dis_filter_size", "dis_channel_size", "dis_stride", "dis_use_batch_norm"]
      optstr = ""
      for s in optstrl:
         optstr += (s + ' ')
      optstr = optstr[: -1]
      ret = namedtuple(optn, optstr)
      return ret._make(args)
   else :
      raise ValueError("Type not allowed!")

def multimyconv(inputs,
                output_channel,
                filter_size=None,
                stride=2,
                padding='SAME',
                spec_norm=False,
                scope=None,
                reuse=None):
   b, c, h, w = inputs.get_shape().as_list()
   b = tf.shape(inputs)[0]
   output_channel = int(output_channel / 4)
   filter_size1 = [1, 2, 8, 16]
   filter_size2 = [4, 8, 16, 32]
   with tf.variable_scope(scope or 'multiconv', reuse=reuse):
      ret = []
      ret2 = []
      inputs = tf.transpose(inputs, [0, 2, 1, 3])
      for i, filter_size in enumerate(filter_size2):
         ret2.append(myconv1d(inputs, h, filter_size, int(filter_size / 2), 'VALID', spec_norm, "conv2_" + str(i), reuse))
      ret2 = tf.concat(ret2, axis=2)
      ret2 = tf.transpose(ret2, [0, 2, 1, 3])
      for i, filter_size in enumerate(filter_size1):
         ret.append(myconv1d(ret2, output_channel, filter_size, stride, padding, spec_norm, "conv1_" + str(i), reuse))
      ret = tf.concat(ret, axis=1)
      return ret

def multimydeconv(inputs,
                output_channel,
                filter_size=None,
                stride=2,
                spec_norm=False,
                scope=None,
                reuse=None):
   b, c, h, w = inputs.get_shape().as_list()
   b = tf.shape(inputs)[0]
   filter_size1 = filter_size
   filter_size2 = 16
   with tf.variable_scope(scope or 'multiconv', reuse=reuse):
      ret = mydeconv1d(inputs, int(output_channel / 4), filter_size, stride, spec_norm, "conv1", reuse)
      nh = ret.get_shape().as_list()[2]
      ret = tf.transpose(ret, [0, 2, 1, 3])
      ret = mydeconv1d(ret, nh, filter_size, 4, spec_norm, "conv2", reuse)
      ret = tf.transpose(ret, [0, 2, 1, 3])
      print (ret)
      return ret

def cyclic_shift(inputs):
   #inputs : numpy array
   b, c, h, w = inputs.shape
   ret = np.transpose(inputs, axes=[0, 2, 1, 3])
   ret = np.reshape(ret, [b * h, c, w])
   ret = ret[int(h / 2): -int(h / 2)]
   pad = np.zeros([h, c, w])
   ret = np.concatenate([ret, pad], axis=0)
   ret = np.transpose(ret, axes=[1, 0, 2])
   ret = np.reshape(ret, [b, c, h, w])
   return ret

def cyclic_back(inputs):
   #inputs : numpy array
   b, w, h, c = inputs.shape
   assert b >= 2
   assert w == 1
   b2 = int(b / 2)
   psize = int(h / 2)
   p1 = inputs[: b2]
   p2 = inputs[b2:]
   p1 = np.reshape(p1, [b2 * h, c])
   p2 = np.reshape(p2, [b2 * h, c])[: -h]
   pad1, pad2, middle = p1[: psize], p1[-psize:], p1[psize: -psize]
   middle = (middle + p2) / 2
   ret = np.concatenate([pad1, middle, pad2], axis=0)
   ret = np.reshape(ret, [b2, w, h, c])
   return ret

def appenddiff(inputs):
   diff = (inputs[:, 1:] - inputs[:, :-1]) / 2
   return tf.concat([inputs, diff], axis=1)
   
def appenddiff2(inputs):
   diff = (inputs[:, 1:] - inputs[:, :-1]) / 2
   diff2 = (diff[:, 1:] - diff[:, :-1]) / 2
   tdiff = (inputs[:, :, 1:] - inputs[:, :, :-1]) / 2
   tdiff = tf.concat([inputs[:, :, :1], tdiff], axis=2)
   return tf.concat([inputs, diff, diff2], axis=1)

def appenddiff2d(inputs):
   print (inputs)
   b, c, t, f = inputs.get_shape().as_list()
   b = tf.shape(inputs)[0]
   pad = tf.random_uniform([b, c, t, 1], -1., 1.)
   pad2 = tf.random_uniform([b, c, t, 1], -1, 1.)
   diff = tf.concat([pad, (inputs[:, :, :, 1:] - inputs[:, :, :, :-1]) / 2], axis=3)
   diff2 = tf.concat([pad2, (diff[:, :, :, 1:] - diff[:, :, :, :-1]) / 2], axis=3)
   ret = tf.concat([inputs, diff], axis=1)
   return ret

def stackraw(inputs):
   ret = appenddiff2d(inputs)
   return ret

def gettfshape(inputs):
   tmpshape = inputs.get_shape().as_list()
   tmpshape[0] = tf.shape(inputs)[0]
   return tmpshape

def timediff(inputs):
   r1 = inputs[:, :, 2:] - inputs[:, :, :-2]
   r2 = inputs[:, :, 4:] - inputs[:, :, :-4]
#   r3 = inputs[:, :, 6:] - inputs[:, :, :-6]
#   ret = r1[:, :, 2: -2] + r2[:, :, 1: -1] * 2 + r3 * 3
   ret = r1[:, :, 1: -1] + r2
#   ret = ret / (2 * (1 ** 2 + 2 ** 2 + 3 ** 2))
#   ret = ret / (2 * (1 ** 2 + 2 ** 2))
   ret = ret / 4
   return ret

def appenddelta(inputs):
   _, f, t, v = inputs.get_shape().as_list()
   b = tf.shape(inputs)[0]
#   pad = tf.zeros([b, f, 3, v], dtype=tf.float32)
   pad = tf.zeros([b, f, 2, v], dtype=tf.float32)
   delta = timediff(inputs)
   delta = tf.concat([pad, delta, pad], axis=2)
   ddelta = timediff(delta)
   ddelta = tf.concat([pad, ddelta, pad], axis=2)
   return tf.concat([inputs, delta, ddelta], axis=1)

def appenddelta2d(inputs):
   inputs = tf.transpose(inputs, perm=[0, 3, 2, 1])
   _, f, t, v = inputs.get_shape().as_list()
   b = tf.shape(inputs)[0]
#   pad = tf.zeros([b, f, 3, v], dtype=tf.float32)
   pad = tf.zeros([b, f, 2, v], dtype=tf.float32)
   delta = timediff(inputs)
   delta = tf.concat([pad, delta, pad], axis=2)
   ddelta = timediff(delta)
   ddelta = tf.concat([pad, ddelta, pad], axis=2)
   return tf.transpose(tf.concat([inputs, delta, ddelta], axis=3), perm=[0, 3, 2, 1])


def appenddelta2dwithfreq(inputs):
   inputs = tf.transpose(inputs, perm=[0, 3, 2, 1])
   _, f, t, v = inputs.get_shape().as_list()
   b = tf.shape(inputs)[0]
#   pad = tf.zeros([b, f, 3, v], dtype=tf.float32)
   pad = tf.zeros([b, f, 2, v], dtype=tf.float32)
   delta = timediff(inputs)
   delta = tf.concat([pad, delta, pad], axis=2)
   ddelta = timediff(delta)
   ddelta = tf.concat([pad, ddelta, pad], axis=2)
   finputs = tf.transpose(inputs, perm=[0, 2, 1, 3])
   fdelta = timediff(finputs)
   pad = tf.zeros([b, t, 2, v], dtype=tf.float32)
   fdelta = tf.concat([pad, fdelta, pad], axis=2)
   ffdelta = timediff(fdelta)
   ffdelta = tf.concat([pad, ffdelta, pad], axis=2)
   fdelta = tf.transpose(fdelta, perm=[0, 2, 1, 3])
   ffdelta = tf.transpose(ffdelta, perm=[0, 2, 1, 3])
   return tf.transpose(tf.concat([inputs, delta, ddelta, fdelta, ffdelta], axis=3), perm=[0, 3, 2, 1])

def makemask(seqlen, time_step):
   dtype = tf.float16 if use_fp16 else tf.float32
   mask = tf.sequence_mask(seqlen, time_step) #(64, 128)
   mask = tf.expand_dims(tf.cast(mask, dtype), axis=2)
   mask = tf.expand_dims(mask, axis=1) #(64, 1, 128, 1)
   return mask

class SNLSTMCell(tf.contrib.rnn.LSTMCell):
   def __init__(self, *args, **kwargs):
      super(SNLSTMCell, self).__init__(*args, **kwargs)

   def build(self, inputs_shape):
      super(SNLSTMCell, self).build(inputs_shape)
      self._kernel = spectral_norm(self._kernel)
