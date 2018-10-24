import tensorflow as tf
from ops import *
from hparams import args

dprate = args.dprate
dprated = args.dprated

def encoder(inputs, options, lat=False, train=True, scope=None, reuse=None):
   opc_f = options.gen_conv_filter_size
   opc_c = options.gen_conv_channel_size
   op_s = options.gen_stride
   op_bn = options.gen_use_batch_norm
   with tf.variable_scope(scope or "encoder", reuse=reuse):
      ret = inputs
      for x in range(len(op_s)):
         name = 'cv' + str(x)
         ret = multimyconv(ret, opc_c[x], opc_f[x], op_s[x], 'SAME', False, tf.nn.relu, train, name, reuse)
         if op_bn:
            ret = batch_norm(ret, train=train, scope='batch_norm2' + name, reuse=reuse)
         ret = tf.nn.relu(ret)
         if train:
            ret = tf.nn.dropout(ret, dprate)
#         ret = dynamic_LSTM(ret, opc_c[x], 'LSTM' + name, reuse)
#         ret = tf.nn.relu(ret)
      if lat:
         latent = multimyconv(ret, 512, 4, 1, 'SAME', False, tf.nn.relu, train, 'preconv22', reuse)
         latent = tf.nn.softmax(latent, axis=1)
         tmpshape = latent.get_shape().as_list()
         tmpshape[0] = tf.shape(latent)[0]
         idxs = tf.argmax(latent, axis=1)
         onehot = tf.one_hot(idxs, depth=tmpshape[1], axis=1)
         ret = tf.transpose(latent, perm=[0, 2, 1, 3])
         ret = tf.reshape(ret, [-1, tmpshape[1]])
         ret = mydense(ret, opc_c[-1], False, 'codebook', reuse)
         ret = tf.reshape(ret, [tmpshape[0], tmpshape[2], opc_c[-1], tmpshape[3]])
         ret = tf.transpose(ret, perm=[0, 2, 1, 3])
         ret2 = tf.transpose(onehot, perm=[0, 2, 1, 3])
         ret2 = tf.reshape(ret2, [-1, tmpshape[1]])
         ret2 = mydense(ret2, opc_c[-1], False, 'codebook', True)
         ret2 = tf.reshape(ret2, [tmpshape[0], tmpshape[2], opc_c[-1], tmpshape[3]])
         ret2 = tf.transpose(ret2, perm=[0, 2, 1, 3])
         ret = ret + tf.stop_gradient(ret2 - ret)
         return ret, latent
      return ret

def decoder(inputs, options, aug=True, train=True, scope=None, reuse=None):
   opd_f = options.gen_deconv_filter_size
   opd_c = options.gen_deconv_channel_size
   op_s = options.gen_stride
   op_bn = options.gen_use_batch_norm
   with tf.variable_scope(scope or "decoder", reuse=reuse):
      ret = inputs
      if aug:
         c = ret.get_shape().as_list()[1]
         ret1 = multimyconv(ret, c, 4, 1, 'SAME', False, tf.nn.relu, train, 'preconv1', reuse)
         ret2 = multimyconv(ret, c, 4, 1, 'SAME', False, tf.nn.relu, train, 'preconv2', reuse)
         ret3 = multimyconv(ret, c, 4, 1, 'SAME', False, tf.nn.relu, train, 'preconv3', reuse)
         ret = tf.concat([ret1, ret2, ret3], axis=2)
         ret = batch_norm(ret, train=train, scope='batch_norm1', reuse=reuse)
         ret = tf.nn.relu(ret)
      for x in range(len(op_s) - 1):
         name = 'dcv' + str(len(op_s) + x)
         ret = mydeconv1d(ret, opd_c[x], opd_f[x], op_s[-(x + 1)], name, reuse)
         if op_bn:
            ret = batch_norm(ret, train=train, scope='batch_norm2' + name, reuse=reuse)
         ret = tf.nn.relu(ret)
         if train:
            ret = tf.nn.dropout(ret, dprate)
      ret = mydeconv1d(ret, opd_c[-1], opd_f[-1], op_s[0], 'out_deconv', reuse)
      return tf.tanh(ret)

def full_generator(inputs, options, aug=True, split=False, train=True, scope=None, reuse=None):
   opd_f = options.gen_deconv_filter_size
   opd_c = options.gen_deconv_channel_size
   op_s = options.gen_stride
   op_bn = options.gen_use_batch_norm
   with tf.variable_scope(scope or "generator", reuse=reuse):
      ret = encoder(inputs, options, False, train, "encoder", reuse)
      ret = shifter(ret, options, train, "shifter", reuse)
      if aug:
         ret1 = decoder(ret, options, False, train, "decoder", reuse)
         ret2 = decoder(ret, options, False, train, "decoder", True)
         ret3 = decoder(ret, options, False, train, "decoder", True)
         ret = tf.concat([ret1, ret2, ret3], axis=2)
      if split:
         return tf.split(ret, 3, axis=2)
      return ret

def unet_generator(inputs, mask, seqlen, options, train=True, scope=None, reuse=None):
   opc_f = options.gen_conv_filter_size
   opc_c = options.gen_conv_channel_size
   op_s = options.gen_stride
   op_bn = options.gen_use_batch_norm
   opd_f = options.gen_deconv_filter_size
   opd_c = options.gen_deconv_channel_size
   op_t = options.gen_time_filter_size
   layers = []
   reslayers = []
   prelayer = None
   masks = [mask]
   with tf.variable_scope(scope or "generator", reuse=reuse):
      with tf.variable_scope("encoder", reuse=reuse):
         ret = inputs
         for x in range(len(op_s)):
            name = 'cv' + str(x)
            ret = myconv1d(ret, opc_c[x], opc_f[x], op_s[x], 'SAME', True, name, reuse)
            if op_s[x] == 2:
               mask = mask[:, :, ::2]
            ret = ret * mask
            if op_bn:
               ret = instance_norm_with_mask(ret, mask, scope='batch_norm2' + name, reuse=reuse)
            ret = tf.nn.elu(ret)
            if x != len(op_s) - 1:
               layers.append(ret * mask)
            masks.append(mask)
      prelayer = None
      with tf.variable_scope("decoder", reuse=reuse):
         ret = ret * masks[-1]
         for x in range(len(opd_c) - 1):
            name = 'dcv' + str(len(op_s) + x)
            ns = op_s[-(x + 1)]
            cmask = masks[-(x + 2)]
            ret = mydeconv1d(ret, opd_c[x], opd_f[x], ns, True, name, reuse)
            ret = ret * cmask
            if op_bn:
               ret = instance_norm_with_mask(ret, cmask, scope='batch_norm2' + name, reuse=reuse)
            ret = tf.nn.elu(ret)
            ret = tf.nn.dropout(ret, dprate)
         ret = mydeconv1d(ret, opd_c[-1], opd_f[-1], op_s[0], True, 'out_deconv', reuse)
         ret = tf.tanh(ret) * args.clip_to_value
      return tf.cast(ret * masks[0], tf.float32)

def unet_generator2d(inputs, mask, seqlen, options, train=True, scope=None, reuse=None):
   opc_f = options.gen_conv_filter_size
   opc_c = options.gen_conv_channel_size
   op_s = options.gen_stride
   op_sc = options.gen_stridec
   op_bn = options.gen_use_batch_norm
   opd_f = options.gen_deconv_filter_size
   opd_c = options.gen_deconv_channel_size
   op_t = options.gen_time_filter_size
#   layers = []
#   reslayers = []
   inputs = tf.transpose(inputs, perm=[0, 3, 2, 1])
   with tf.variable_scope(scope or "generator", reuse=reuse):
      with tf.variable_scope("encoder", reuse=reuse):
         ret = inputs
         for x in range(len(op_sc)):
            name = 'cv' + str(x)
            ret = myconv2d(ret, opc_c[x], opc_f[x], op_sc[x], 'SAME', True, name, reuse)
            print (np.prod(np.array(op_sc)[: (x + 1), 2]))
            ret = ret * mask[:, :, ::np.prod(np.array(op_sc)[: (x + 1), 2])]
            if op_bn:
               ret = instance_norm_with_mask(ret, mask[:, :, ::np.prod(np.array(op_sc)[: (x + 1), 2])], scope='batch_norm2' + name, reuse=reuse)
            ret = tf.nn.elu(ret)
            ret = tf.nn.dropout(ret, dprate)
#            if x != len(op_s) - 1:
#               layers.append(ret * mask)
            print (ret)
         ret /= tf.maximum((tf.reduce_sum(ret ** 2, axis=1, keepdims=True) + 1e-5) ** 0.5, 1.)
         ret = ret * mask[:, :, ::np.prod(np.array(op_sc)[:, 2])]
         condition = ret
         print (ret)
      with tf.variable_scope("decoder", reuse=reuse):
         for x in range(len(opd_c) - 1):
            name = 'dcv' + str(len(op_s) + x)
            ns = op_s[-(x + 1)]
            ret = mydeconv2d(ret, opd_c[x], opd_f[x], ns, True, name, reuse)
            ret = ret * mask[:, :, ::np.prod(np.array(op_s)[:-(x + 1) , 2])]
            print (np.prod(np.array(op_s)[:-(x + 1), 2]))
            if op_bn:
               ret = instance_norm_with_mask(ret, mask[:, :, ::np.prod(np.array(op_s)[:-(x + 1), 2])], scope='batch_norm2' + name, reuse=reuse)
            ret = tf.nn.elu(ret)
#            if x in [1, 2]:
#               ret = tf.concat([ret, layers[-(x + 1)]], axis=1)
            print (ret)
#            ret = tf.nn.dropout(ret, dprate)
         ret = mydeconv2d(ret, opd_c[-1], opd_f[-1], op_s[0], True, 'out_deconv', reuse)
#         ret = ret * masks[0]
#         if op_bn:
#            ret = instance_norm_with_mask(ret, masks[0], scope='batch_norm2_out', reuse=reuse)
#         ret = tf.nn.elu(ret)
#         ret, _ = tf.nn.dynamic_rnn(SNLSTMCell(args.channel_size), tf.squeeze(ret, axis=1), dtype=tf.float32)
         ret = tf.tanh(ret) * args.clip_to_value
#         ret = tf.expand_dims(ret, axis=1)
         ret = tf.transpose(ret * mask, perm=[0, 3, 2, 1])
      return tf.cast(ret, tf.float32), condition

def discriminator(inputs, mask, options, only_first=False, scope=None, reuse=None):
#   inputs = appenddelta(inputs)
   op_s = options.dis_stride
   op_c = options.dis_channel_size
   op_f = options.dis_filter_size
   op_bn = options.dis_use_batch_norm
   p_layers = [inputs]
   with tf.variable_scope(scope or "discriminator", reuse=reuse):
      c_layer = inputs
#######################
      c_layer = tf.transpose(c_layer, perm=[0, 3, 2, 1])
      c_layer = myconv2d(c_layer, 8, [5, 5], [1, 1], 'SAME', True,'preconvv', reuse)
      if op_bn:
         c_layer = instance_norm_with_mask(c_layer, mask, scope='batc_norm_pre', reuse=reuse)
      c_layer = tf.nn.relu(c_layer)
      c_layer = tf.transpose(c_layer, perm=[0, 3, 2, 1])
      c_layer = c_layer * mask
      c_layer = tf.nn.dropout(c_layer, dprated)
      c_layer = tf.reduce_mean(c_layer, axis=3, keepdims=True)
#      c_layer = tf.transpose(c_layer, perm=[0, 3, 2, 1])
#      c_layer = myconv2d(c_layer, 1, [5, 5], [1, 1], 'SAME', True,'preconvv3', reuse)
#      if op_bn:
#         c_layer = instance_norm_with_mask(c_layer, mask, scope='batc_norm_pre3', reuse=reuse)
#      c_layer = tf.nn.relu(c_layer)
#      c_layer = tf.transpose(c_layer, perm=[0, 3, 2, 1])
#      c_layer = c_layer * mask
#      p_layers.append(c_layer)
#######################

#######################
      res = inputs
      tostridet = 1
#      inputsf = inputs.get_shape().as_list()[1]
#######################
      for x in range(len(op_s)):
         name = 'cv' + str(x)
         padding = 'SAME'# if x != len(op_s) - 1 else 'VALID'
#######################
#         tostridef = int(inputsf / op_c[x]) if inputsf > op_c[x] else 1
#         if op_s[x] > 1 and tostridef > 1:
#            tostridet *= op_s[x]
#            res = tf.transpose(inputs, perm=[0, 3, 2, 1])
#            res = tf.layers.max_pooling2d(res,
#                                          pool_size=[tostridet * 2, tostridef * 2],
#                                          strides=[tostridet, tostridef],
#                                          padding='same',
#                                          data_format='channels_first')
#            res = tf.transpose(res, perm=[0, 3, 2, 1])
         if op_s[x] > 1:
            tostridet *= op_s[x]
            res = tf.layers.max_pooling1d(tf.squeeze(inputs, axis=-1),
                                           pool_size=tostridet,
                                           strides=tostridet,
                                           padding='same',
                                           data_format='channels_first')
            res = tf.expand_dims(res, axis=-1)
         print (res, tostridet, op_f[x])
#######################
         c_layer = myconv1d(c_layer, op_c[x], op_f[x], op_s[x], padding, True, name, reuse)
         if op_s[x] == 2:
            mask = mask[:, :, ::2]
         c_layer = c_layer * mask
         if op_bn:
            c_layer = instance_norm_with_mask(c_layer, mask, scope='batch_norm2' + name, reuse=reuse)
         c_layer = tf.nn.relu(c_layer)
         c_layer = c_layer * mask
         if x == 0:
            first_layer = c_layer
            if only_first:
               return tf.cast(first_layer, tf.float32)
         p_layers.append(c_layer)
         print (c_layer, padding)
#######################
         if c_layer.dtype.base_dtype == tf.float16 and res.dtype.base_dtype == tf.float32:
            res = tf.cast(res, tf.float16)
#         c_layer = tf.concat([c_layer, res], axis=1)
         if x != 0:
            c_layer = tf.nn.dropout(c_layer, dprated)
         with tf.control_dependencies([tf.check_numerics(c_layer, 'nan' + name)]):
            c_layer = tf.identity(c_layer)
#######################
      c_layer = flatten(c_layer)
      c_layer = mydense(c_layer, 1, True, True, 'output_linear', reuse)
      return tf.cast(c_layer, tf.float32), [tf.cast(first_layer, tf.float32) for first_layer in p_layers]

def discriminator2d(inputs, mask, options, scope=None, reuse=None):
   inputs = tf.transpose(inputs, perm=[0, 3, 2, 1])
#   inputs = appenddelta2d(inputs)
   inputs = appenddelta2dwithfreq(inputs)
   p_layers = [inputs]
   op_s = options.dis_stride
   op_c = options.dis_channel_size
   op_f = options.dis_filter_size
   op_bn = options.dis_use_batch_norm
#   p_layers = []
   with tf.variable_scope(scope or "discriminator", reuse=reuse):
      c_layer = inputs
#######################
      res = inputs
#######################
      for x in range(len(op_s)):
         name = 'cv' + str(x)
         padding = 'SAME'# if x != len(op_s) - 1 else 'VALID'
#######################
         res = tf.layers.max_pooling2d(res,
                                       pool_size=[2, 2],
                                       strides=op_s[x][2: ],
                                       padding='same',
                                       data_format='channels_first')
#######################
         c_layer = myconv2d(c_layer, op_c[x], op_f[x], op_s[x], padding, True, name, reuse)
         if op_s[x][2] == 2:
            mask = mask[:, :, ::2]
         if op_s[x][2] == 4:
            mask = mask[:, :, ::4]
         c_layer = c_layer * mask
#         if op_bn and x != 0:
         c_layer = instance_norm_with_mask(c_layer, mask, scope='batch_norm2' + name, reuse=reuse)
         c_layer = tf.nn.relu(c_layer)
#         c_layer = tf.clip_by_value(c_layer, 0., 1.)
         c_layer = tf.nn.dropout(c_layer, dprated)
         p_layers.append(c_layer)
#######################
         if c_layer.dtype.base_dtype == tf.float16 and res.dtype.base_dtype == tf.float32:
            res = tf.cast(res, tf.float16)
         padlen = op_c[x] // res.get_shape().as_list()[1]
         toconcat = [c_layer] + [res] * padlen
         c_layer = tf.concat(toconcat, axis=1)
#######################
         print (c_layer, padding)
      c_layer = flatten(c_layer)
      c_layer = mydense(c_layer, 1, True, True, 'output_linear', reuse)
      return tf.cast(c_layer, tf.float32), [tf.cast(first_layer, tf.float32) for first_layer in p_layers]

def vallina_metric(logits, labels):
   ret = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
   return tf.reduce_mean(ret)

def softmax_metric(logits, labels):
   logits = tf.squeeze(logits)
   labels = tf.squeeze(labels)
   labels = tf.cast(tf.reduce_mean(labels, -1), tf.int32)
   ret = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
   return tf.reduce_mean(ret)

def l1_metric(logits, labels):
   return tf.reduce_mean(tf.abs(logits - labels))

def l2_metric(logits, labels):
   return tf.reduce_mean((logits - labels) ** 2)

def vector_l2_metric(logits, labels):
   length = logits.get_shape().as_list()[1]
   if length == 1:
     return l2_metric(logits, labels)
   logits = tf.squeeze(logits)
   labels = tf.squeeze(labels)
   idx = tf.cast(tf.reduce_mean(labels), tf.int32)
   labels = tf.zeros_like(logits)
   labels += tf.one_hot(idx, length)
   return l2_metric(logits, labels)

def bound_low(loss, x=0.8):
   return tf.maximum(loss, x)

def bound_up(loss, x=0.8):
   return tf.minimum(loss, x)

def mask_tensor(inputs, keep_prob):
   if keep_prob == 1.0:
      return inputs
   noise_shape = inputs.get_shape().as_list()
   noise_shape[0] = tf.shape(inputs)[0]
   random_tensor = keep_prob
   random_tensor += tf.random_uniform(noise_shape, dtype=tf.float32)
   binary_tensor = tf.floor(random_tensor)
   return inputs * binary_tensor

def mask_tensor_hor(inputs, keep_prob):
   if keep_prob == 1.0:
      return inputs
   b, c, h, w = inputs.get_shape().as_list()
   random_tensor = keep_prob
   random_tensor += tf.random_uniform([b, 1, h, w], dtype=tf.float32)
   binary_tensor = tf.floor(random_tensor)
   return inputs * binary_tensor

def mask_tensor_back(inputs, remain_num):
   b, c, h, w = inputs.get_shape().as_list()
   b = tf.shape(inputs)[0]
   inputs = inputs[:, :, : remain_num]
   pad = tf.zeros([b, c, h - remain_num, w])
   return tf.concat([inputs, pad], axis=2)

def insert_tensor(inputs, gap=10):
   b, c, h, w = inputs.get_shape().as_list()
   b = tf.shape(inputs)[0]
   insert_idx = tf.random_uniform([], gap, h - gap * 2, dtype=tf.int32)
   insert_len = tf.random_uniform([], 0, gap, dtype=tf.int32)
   insert = tf.zeros([b, c, insert_len, w], dtype=tf.float32)
   inserted = tf.concat([inputs[:, :, : insert_idx], insert, inputs[:, :, insert_idx: (h - insert_len)]], axis=2)
   inserted = tf.reshape(inserted, [b, c, h, w])
   return inserted

def shorten_tensor(inputs, gap=10):
   b, c, h, w = inputs.get_shape().as_list()
   b = tf.shape(inputs)[0]
   insert_idx = tf.random_uniform([], gap, h - gap * 2, dtype=tf.int32)
   insert_len = tf.random_uniform([], 0, gap, dtype=tf.int32)
   insert = tf.zeros([b, c, insert_len, w], dtype=tf.float32)
   inserted = tf.concat([inputs[:, :, : insert_idx], inputs[:, :, (insert_idx + insert_len): ], insert], axis=2)
   inserted = tf.reshape(inserted, [b, c, h, w])
   return inserted

def proc_tensor(inputs, keep_prob, gap):
#   inputs = mask_tensor(inputs, keep_prob)
#   inputs = insert_tensor(inputs, gap)
#   inputs = shorten_tensor(inputs, gap)
   return inputs

def mask_l1_metric(logits, labels):
   ret = tf.abs(logits - labels)
   ret = tf.maximum(ret, 2.0) - 2.0
   return tf.reduce_mean(ret)

def varlen_dropout(inputs, keep_prob=1.0, gap=5):
   ret = insert_tensor(inputs, gap)
   ret = shorten_tensor(ret, gap)
   ret = tf.nn.dropout(ret, keep_prob)
   return ret

def make_pn_one(prob, shape=[]):
   random_tensor = prob
   random_tensor += tf.random_uniform(shape, dtype=tf.float32)
   binary_tensor = tf.floor(random_tensor)
   return 2 * binary_tensor - 1 

def prob_l2_metric(logits, labels, prob):
   shape = logits.get_shape().as_list()
   diff = (logits - labels) ** 2
   p_mat = make_pn_one(prob, shape)
   return tf.reduce_mean(p_mat * diff)

def hinge_metric(logits, labels):
   #have loss if logits > labels
   return tf.reduce_mean(tf.nn.relu(logits - labels))

def keep_or_mask(ret, inputs, keep_prob=1.0):
   if keep_prob == 1.0:
      return ret
   noise_shape = inputs.get_shape().as_list()
   random_tensor = keep_prob
   random_tensor += tf.random_uniform(noise_shape, dtype=tf.float32)
   binary_tensor = tf.floor(random_tensor)
   return ret * binary_tensor + inputs * (1 - binary_tensor)
   
def fgsm(x, loss, order, epsilon, big_or_small=True):
   if big_or_small:
      factor = make_pn_one(0.5)
   else:
      factor = 1.0
   grad = tf.gradients(loss, x)[0]
   if order == 'inf':
      normalized_grad = tf.sign(grad)
      normalized_grad = tf.stop_gradient(normalized_grad)
   elif order == 1:
      normalized_grad = grad / tf.reduce_sum(tf.abs(grad), axis=[1, 2, 3])
   elif order == 2:
      normalized_grad = grad / tf.sqrt(tf.reduce_sum(grad ** 2, axis=[1, 2, 3]))
   else:
      raise NotImplementedError("Only inf, 1, 2 norm is allowed.")
   return x + factor * epsilon * normalized_grad

def f0_act(inputs):
   mceps, lf0 = inputs[:, :, :, : -1], inputs[:, :, :, -1:]
#   lf0 = 4.0 * tf.sigmoid(lf0) + 2.0
#   lf0 = tf.where(lf0 > 4.0, lf0, tf.zeros_like(lf0))
   return tf.concat([mceps, lf0], axis=3)

class buff():
   def __init__(self, loss_names):
      self.loss_names = loss_names
      self.buffers = [0. for x in self.loss_names]
      self.count = [0 for x in self.loss_names]
      self.loss_string = "Epoch : %r Batch : %r / %r "
      for loss_name in loss_names:
         self.loss_string = self.loss_string + loss_name + " : %.6f "

   def put(self, x, index):
      assert len(x) == len(index)
      for y, idx in zip(x, index):
         self.buffers[idx] += y
         self.count[idx] += 1.

   def printout(self, prefix):
      losses = tuple(prefix + [x / c if c != 0 else 0 for x, c in zip(self.buffers, self.count)])
      self.buffers = [0. for x in self.buffers]
      self.count = [0 for x in self.buffers]
      print (self.loss_string %losses)

   def get(self, index):
      return self.buffers[index] / self.count[index] if self.count[index] != 0 else 0
