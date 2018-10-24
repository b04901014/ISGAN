import tensorflow as tf
tf.set_random_seed(42)
import os
import numpy as np

from hparams import args
from model import UVTGAN

def main(_):
   if args.saving_path:
      if not os.path.exists(args.saving_path):
         os.makedirs(args.saving_path)
   if args.sampling_path:
      if not os.path.exists(args.sampling_path):
         os.makedirs(args.sampling_path)
   gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
   tfconfig = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

   with tf.Session(config=tfconfig) as sess:
      model = UVTGAN(sess)
      if args.is_training:
         model.train()
      else:
         model.test()

if __name__ == '__main__':
   tf.app.run()
