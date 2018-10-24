import numpy as np
import librosa
import os
import sys
import subprocess
import scipy
import traceback
from scipy.io.wavfile import read, write
from pydub import AudioSegment
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from hparams import args

#################################################################
# If Hparam adjustion is needed, please change it in hparams.py #
#################################################################

def norm(ifn, trim_inner_scilence=False):
   sound = AudioSegment.from_file(ifn, "wav")
   if sound.channels != 1:
      sound = sound.split_to_mono()[0]
   normalized_sound = match_target_amplitude(sound, -20.0)
   y = np.array(normalized_sound.get_array_of_samples(), dtype=np.float32)
   y = y / 2 ** 15
   if trim_inner_scilence:
      yss = librosa.effects.split(y, top_db=10, frame_length=2048, hop_length=512)
      y = [y[z[0]: z[1] + 10] for z in yss]
      y = np.concatenate(y, axis=0)
#      write("./test/" + os.path.basename(op_path) + '.wav', 16000, y)
   else:
      y, index = librosa.effects.trim(y, top_db=10, frame_length=2048, hop_length=512)
   return y

def wav2msp(x):
#   ret = scipy.signal.lfilter([1, -0.97], [1], x)
   ret = librosa.stft(x, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.window_size)
   ret = np.abs(ret)
   ret = np.matmul(args.melbasis, ret)
   ret = -20 * np.log10(np.maximum(ret, 1e-8))
   ret = np.clip(ret, -150, 150)
   return ret.transpose()

def match_target_amplitude(sound, target_dBFS):
  change_in_dBFS = target_dBFS - sound.dBFS
  return sound.apply_gain(change_in_dBFS)

def save_audio2arr(y, op_path):
   tos = wav2msp(y)
   np.save(op_path, tos)
   return np.sum(tos, axis=0), tos.shape[0]

def norm_and_save(ifn, ofn, trim_inner_scilence=False):
   y = norm(ifn, trim_inner_scilence)
#   write("./test/" + os.path.basename(ifn), 16000, ar)
   return save_audio2arr(y, ofn)

def process_one_dir(root, op_dir, trim_inner_scilence=False):
   print ("Cleaning Output Directory %s" %op_dir)
   for f in os.listdir(op_dir):
      f = os.path.join(op_dir, f)
      if os.path.isfile(f):
         os.remove(f)
   mean_a = np.zeros([args.channel_size])
   la = 0
   i = 0
   for path, subdirs, files in os.walk(root):
      for name in sorted(files):
         if name[-3:] != "wav":
            continue
         ifn = os.path.join(path, name)
         ofn = os.path.join(op_dir, str(i))
         try:
            ma, _la = norm_and_save(ifn, ofn, trim_inner_scilence)
            mean_a += ma
            la += _la
            i += 1
            print ("Writing %r to %r" %(ifn, ofn))
         except:
            traceback.print_exc()
            continue
   return mean_a, la

def process_one_dir_std(mean_a, op_dir):
   std_a = np.zeros([args.channel_size])
   for f in os.listdir(op_dir):
      if f[-3:] == 'npy':
         ofn = os.path.join(op_dir, f)
         try:
            std_a += np.sum((np.load(ofn) - mean_a) ** 2, axis=0)
         except:
            traceback.print_exc()
            continue
   return std_a
   
def dtw_pair(a, b, c):
   fa, fb = sorted(os.listdir(a)), sorted(os.listdir(b))
   for _x, _y in zip(fa, fb):
      x, y = os.path.join(a, _x), os.path.join(b, _y)
      x, y = np.load(x), np.load(y)
      D, ind = librosa.core.dtw(x.T, y.T)
      ind = [ind[0]] + [ind[i] for i in range(1, len(ind)) if ind[i][0] != ind[i - 1][0]]
      ind = np.array([x[1] for x in ind])
      np.save(os.path.join(c, _y), y[ind])

def run():
#   mean_a, la = process_one_dir(args.a_wav_dir, args.a_data_dir, True)
#   mean_a /= la
#   std_a = process_one_dir_std(mean_a, args.a_data_dir)
#   std_a = (std_a / la) ** 0.5
#   np.save("./datasets/stats/mean_a", mean_a)
#   np.save("./datasets/stats/std_a", std_a)

#  mean_b, lb = process_one_dir(args.b_wav_dir, args.b_data_dir)
#  mean_b /= lb
#  std_b = process_one_dir_std(mean_b, args.b_data_dir)
#  std_b = (std_b / lb) ** 0.5
#  np.save("./datasets/stats/mean_b", mean_b)
#  np.save("./datasets/stats/std_b", std_b)
#  print (la, lb)
#   if args.p_wav_dir is not None and args.paired_training:
#      process_one_dir(args.p_wav_dir, args.p_data_dir, True)
    dtw_pair(args.a_data_dir, '/data/lichen/VCTK_TASK_2/Paired/', '/data/lichen/VCTK_TASK_2/Aligned/')
   

if __name__ =='__main__':
   run()
