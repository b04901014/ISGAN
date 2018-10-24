import numpy as np
import random
import os
from scipy.io.wavfile import read, write
import tensorflow as tf
import threading
import multiprocessing
import time
import librosa
import librosa.display
import scipy
from scipy.io.wavfile import read
import speech_recognition as sprecog
import matplotlib.pyplot as plt
from hparams import args
from datasets.procaudio import norm, wav2msp
plt.switch_backend('Agg')

mean_a = np.load('datasets/stats/mean_a.npy')
#mean_a = np.mean(mean_a)
mean_b = np.load('datasets/stats/mean_b.npy')
#mean_b = np.mean(mean_b)
std_a = np.load('datasets/stats/std_a.npy')
#std_a = np.mean(std_a ** 2) ** 0.5
std_b = np.load('datasets/stats/std_b.npy')
#std_b = np.mean(std_b ** 2) ** 0.5

def len2list(l, t):
   x = [args.time_step] * (l // args.time_step)
   r = l % args.time_step
   if r != 0:
      x.append(r)
   n = t // args.time_step
   if len(x) < n:
      x += [0] * (n - len(x))
   assert len(x) == n
   return x

def speech_recog(audio, sr):
   r = sprecog.Recognizer()
   audio = sprecog.AudioData(audio.tobytes(), sr, 2)
   try:
      text = r.recognize_sphinx(audio, language=args.sr_lan, show_all=False)
   except sprecog.UnknownValueError:
      text = ""
   return text

def padtomaxlen(inputs, maxlen):
   for i in range(len(inputs)):
      if inputs[i].shape[0] > maxlen:
         rn = random.randint(0, inputs[i].shape[0] - maxlen // 2)
         inputs[i] = inputs[i][rn: ]
      if inputs[i].shape[0] < maxlen:
         pad = np.zeros([maxlen - inputs[i].shape[0]] + list(inputs[i].shape[1:]))
         inputs[i] = np.concatenate([inputs[i], pad], axis=0)
      else:
         inputs[i] = inputs[i][: maxlen]
   return np.array(inputs) #(5, 5, 1, 64, 64)
         
def loadmsp(x):
   return np.load(x)

def load_np_wav(path, l):
   y = norm(path, False)
   y = wav2msp(y)
   y = (y - mean_a) / std_a
   ret_len = y.shape[0]
   if y.shape[0] < l:
      pad = np.zeros([l - len(y), y.shape[1]])
      y = np.concatenate([y, pad], 0)
   return np.clip(y, -args.clip_to_value, args.clip_to_value), ret_len #(64, 64)

def load_np_sam(path, l):
   y = loadmsp(path)
   y = (y - mean_a) / std_a
   ret_len = y.shape[0]
   if y.shape[0] < l:
      pad = np.zeros([l - len(y), y.shape[1]])
      y = np.concatenate([y, pad], 0)
   return np.clip(y, -args.clip_to_value, args.clip_to_value), ret_len #(64, 64)

def sample_data_generator():
   a_fnames = [os.path.join(args.a_data_dir, fname) for fname in os.listdir(args.a_data_dir) if fname[-3:] == "npy"]
   random.shuffle(a_fnames)
   ret_data, ret_len, cnt = [], [], 0
   while 1:
      for i in range(args.sample_num):
         a, b = load_np_sam(a_fnames[cnt], args.time_step)
         ret_data.append(a)
         ret_len.append(b)
         cnt += 1
         if cnt == len(a_fnames):
            cnt = 0
            random.shuffle(a_fnames)
      maxlen = max([data.shape[0] for data in ret_data])
      maxlen = (int(maxlen / args.time_step) + int(maxlen % args.time_step != 0)) * args.time_step
      ret_data = padtomaxlen(ret_data, maxlen)
      ret_len = np.array(ret_len, dtype=np.int32)
      ret_data = np.transpose(ret_data, axes=[0, 2, 1])
      yield np.expand_dims(ret_data, 3), ret_len
      ret_data, ret_len = [], []

def test_data_generator():
   if args.test_data_dir:
      a_fnames = [os.path.join(args.test_data_dir, fname) for fname in os.listdir(args.test_data_dir) if fname[-3:] == "wav"]
   else:
      a_fnames = [os.path.join(args.a_data_dir, fname) for fname in os.listdir(args.a_data_dir) if fname[-3:] == "npy"]
   ret_data, ret_len = [], []
   for i, cf in enumerate(a_fnames):
      if args.test_data_dir:
         a, b = load_np_wav(a_fnames[i], args.time_step)
      else:
         a, b = load_np_sam(a_fnames[i], args.time_step)
      ret_data.append(a)
      ret_len.append(b)
      if (i + 1) % args.batch_size == 0:
         maxlen = max([data.shape[0] for data in ret_data])
         maxlen = (int(maxlen / args.time_step) + int(maxlen % args.time_step != 0)) * args.time_step
         ret_data = padtomaxlen(ret_data, maxlen)
         ret_len = np.array(ret_len, dtype=np.int32)
         ret_data = np.transpose(ret_data, axes=[0, 2, 1])
         yield np.expand_dims(ret_data, 3), ret_len
         ret_data, ret_len = [], []
   if len(ret_data) != 0:
      maxlen = max([data.shape[0] for data in ret_data])
      maxlen = (int(maxlen / args.time_step) + int(maxlen % args.time_step != 0)) * args.time_step
      ret_data = padtomaxlen(ret_data, maxlen)
      ret_len = np.array(ret_len, dtype=np.int32)
      ret_data = np.transpose(ret_data, axes=[0, 2, 1])
      yield np.expand_dims(ret_data, 3), ret_len

def proc_y(y, l):
   y = loadmsp(y)
   seqlen = y.shape[0]
   if y.shape[0] < l:
      pad = np.zeros([l - len(y), y.shape[1]])
      y = np.concatenate([y, pad], 0)
   y = y[: l]
   return np.expand_dims(y, 0), seqlen

def do_rec(arr, acc_p, p, l, time_step, vec_len):
   ret = np.reshape(arr[acc_p: acc_p + p], [time_step * p, vec_len])
   if p == 1:
      ret = ret[: l]
   elif time_step * p != l:
      qqq = time_step * (p - 1)
      ret1 = ret[: qqq]
      ret2 = ret[-(l- qqq): ]
      ret = np.concatenate([ret1, ret2], axis=0)
   assert ret.shape[0] == l
   return ret

def do_rec_ar(arr, p, l):
   ret = arr
   if p == 1:
      ret = ret[: l]
   elif args.time_step * p != l:
      qqq = time_step * (p - 1)
      ret1 = ret[: qqq]
      ret2 = ret[-(l- qqq): ]
      ret = np.concatenate([ret1, ret2], axis=0)
   assert ret.shape[0] == l
   return ret

def msp2audio(path, outpath, x, plotid, title, style=None):
   if style == 'a':
      x = x * std_a + mean_a
   elif style == 'b':
      x = x * std_b + mean_b
   x = x.transpose()
   x = np.power(10, x / -20.)
   toplot = librosa.amplitude_to_db(x ** 1.5, ref=np.max)
   plt.subplot(plotid)
   plt.title(title)
   librosa.display.specshow(toplot, sr=args.sample_rate, y_axis='mel', x_axis='time')
   plt.colorbar(format="%+2.0f dB")
   x = np.maximum(1e-10, np.matmul(args.melbasisinv, x)) ** 1.5
   angles = np.exp(2j * np.pi * np.random.rand(*x.shape))
   x = x.astype(np.complex)
   ret = librosa.core.istft(x * angles, hop_length=args.hop_length, win_length=args.window_size)
   for i in range(80):
     angles = np.exp(1j * np.angle(librosa.core.stft(ret, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.window_size)))
     ret = librosa.core.istft(x * angles, hop_length=args.hop_length, win_length=args.window_size)
#   ret = scipy.signal.lfilter([1], [1, -0.97], ret)
   ret = np.clip(ret, -1.0, 1.0) * 32767 / max(0.01, np.max(np.abs(ret))) 
   ret = ret.astype(np.int16) # To 16-bit PCM
   with open(outpath, "w") as textfile:
     print(speech_recog(ret, args.sample_rate), file=textfile)
   write(path, args.sample_rate, ret)

class multiproc_reader():
   def __init__(self):
      self.a_data_dir = args.a_data_dir
      self.b_data_dir = args.b_data_dir
      self.p_data_dir = args.p_data_dir
      self.time_span = args.time_step
      print ("Training Inputs are constriant to %.4f seconds" \
             %(self.time_span * args.frame_shift / 1000.))
      self.batch_size = args.batch_size
      self.manager = multiprocessing.Manager()
      self.afnames = [os.path.join(self.a_data_dir, fname) for fname in os.listdir(self.a_data_dir) if fname[-3:] == "npy"]
      print ("Total number of A domain data: %r" %(len(self.afnames)))
      self.afnames = self.manager.list(self.afnames)
      random.shuffle(self.afnames)
      self.bfnames = [os.path.join(self.b_data_dir, fname) for fname in os.listdir(self.b_data_dir) if fname[-3:] == "npy"]
      print ("Total number of B domain data: %r" %(len(self.bfnames)))
      self.bfnames = self.manager.list(self.bfnames)
      random.shuffle(self.bfnames)
      if args.paired_training:
         self.pfnames = [os.path.join(self.p_data_dir, fname) for fname in os.listdir(self.p_data_dir) if fname[-3:] == "npy"]
         print ("Total number of Paired domain data: %r" %(len(self.pfnames)))
         self.pfnames = self.manager.list(self.pfnames)
         random.shuffle(self.pfnames)
         assert len(self.pfnames) == len(self.afnames)
      self.queue = multiprocessing.Queue(500)
      self.lock = multiprocessing.Lock()
      self.cnte, self.cntc = multiprocessing.Value('i', 0), multiprocessing.Value('i', 0)

   def dequeue(self):
      return self.queue.get()

   def main_proc(self, cntc, cnte):
      stop = False
      while not stop:
         yc, ye, yp = [], [], []
         for _ in range(self.batch_size):
            self.lock.acquire()
            refc = cntc.value
            refe = cnte.value
            cntc.value += 1
            cnte.value += 1
            if cntc.value >= len(self.afnames):
               cntc.value = 0 
               index = np.arange(len(self.afnames))
               random.shuffle(index)
               self.afnames = self.manager.list([self.afnames[i] for i in index])
               if args.paired_training:
                  self.pfnames = self.manager.list([self.pfnames[i] for i in index])
            if cnte.value >= len(self.bfnames):
               cnte.value = 0
               random.shuffle(self.bfnames)
            self.lock.release()
            afname = self.afnames[refc]
            bfname = self.bfnames[refe]
            pa = np.clip((loadmsp(afname) - mean_a) / std_a, -args.clip_to_value, args.clip_to_value)
            pb = np.clip((loadmsp(bfname) - mean_b) / std_b, -args.clip_to_value, args.clip_to_value)
            yc.append(pa)
            ye.append(pb)
            if args.paired_training:
               pfname = self.pfnames[refc]
               pp = np.clip((loadmsp(pfname) - mean_b) / std_b, -args.clip_to_value, args.clip_to_value)
               yp.append(pp)
         seqlenc = [x.shape[0] for x in yc]
         seqlene = [x.shape[0] for x in ye]
         retyc = np.expand_dims(padtomaxlen(yc, self.time_span), 1) #(64, 1, 512, 64)
         retye = np.expand_dims(padtomaxlen(ye, self.time_span), 1) #(64, 1, 512, 64)
         retyc = np.transpose(retyc, axes=[0, 3, 2, 1]) #(64, 64, 512, 1)
         retye = np.transpose(retye, axes=[0, 3, 2, 1]) #(64, 64, 512, 1)
         if args.paired_training:
            seqlenp = [x.shape[0] for x in yp]
            retyp = np.expand_dims(padtomaxlen(yp, self.time_span), 1) #(64, 1, 512, 64)
            retyp = np.transpose(retyp, axes=[0, 3, 2, 1]) #(64, 64, 512, 1)
         if not args.paired_training:
            self.queue.put((retyc.astype(np.float32), retye.astype(np.float32), np.array(seqlenc, dtype=np.int32), np.array(seqlene, dtype=np.int32)))
         else:
            self.queue.put((retyc.astype(np.float32), retye.astype(np.float32), retyp.astype(np.float32), np.array(seqlenc, dtype=np.int32), np.array(seqlene, dtype=np.int32), np.array(seqlenp, dtype=np.int32)))

   def start_enqueue(self, num_proc=multiprocessing.cpu_count()):
#      num_proc = 4 #locked
      procs = []
      for _ in range(num_proc):
         p = multiprocessing.Process(target=self.main_proc, args=(self.cntc, self.cnte))
         p.start()
         procs.append(p)
      return procs

   def printqsize(self):
      print ("Queue Size : ", self.queue.qsize())
