import argparse
import librosa
import numpy as np

parser = argparse.ArgumentParser(description='GAN for Voice Transformation')
parser.add_argument('--time_step', dest='time_step', type=int, default=64, help='Time Step of Inputs')
parser.add_argument('--sample_rate', dest='sample_rate', type=int, default=16000, help='Sample Rate of Input Audios')
parser.add_argument('--vec_len', dest='vector_length', type=int, default=1, help='Vector Length of Inputs')
parser.add_argument('--ch_size', dest='channel_size', type=int, default=128, help='Channel Size of Inputs')
parser.add_argument('--lambda', dest='Lambda', type=float, default=1., help='Constant Factor of Reconstruction Loss')
parser.add_argument('--a_wav_dir', dest='a_wav_dir', default='/data/lichen/new_data/Subject3', help='Path to A Audio Utterances')
parser.add_argument('--b_wav_dir', dest='b_wav_dir', default='/data/lichen/TTS_data/male', help='Path to B Audio Utterances')
parser.add_argument('--a_data_dir', dest='a_data_dir', default='/data/lichen/VCTK_TASK_2/Native/', help='Path to A Speakers\' Data(Numpy array)')
parser.add_argument('--b_data_dir', dest='b_data_dir', default='/data/lichen/VCTK_TASK_2/Chinese/', help='Path to B Speakers\' Data(Numpy array)')
parser.add_argument('--test_data_dir', dest='test_data_dir', default=None, help='Path to Testing wavs')
parser.add_argument('--p_data_dir', dest='p_data_dir', default=None, help='Path to Paired Speakers\' Data(Numpy array)')
parser.add_argument('--p_wav_dir', dest='p_wav_dir', default='/data/lichen/Normal', help='Path to Paired wavs')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64, help='Batch Size')
parser.add_argument('--sample_num', dest='sample_num', type=int, default=5, help='Number of Audios per Sample')

parser.add_argument('--gen_lr', dest='gen_lr', type=float, default=2e-4, help='Generator Initial Learning Rate')
parser.add_argument('--enc_lr', dest='enc_lr', type=float, default=2e-4, help='Encoder Initial Learning Rate')
parser.add_argument('--dis_lr', dest='dis_lr', type=float, default=1e-4, help='Discriminator Initial Learning Rate')
parser.add_argument('--beta1', dest='beta1', type=float, default=.5, help='Beta1 for Adam Optimizer')
parser.add_argument('--total_examples', dest='total_examples', type=int, default=1000 * 64, help='Number of Examples per Epoch')
parser.add_argument('--epoch', dest='epoch', type=int, default=800, help='Number of Epochs')
parser.add_argument('--display_step', dest='display_step', type=int, default=100, help='Batch to Output Training Details')
parser.add_argument('--saving_epoch', dest='saving_epoch', type=int, default=5, help='Epoch to Save Model')
parser.add_argument('--sample_epoch', dest='sample_epoch', type=int, default=1, help='Epoch to Sample')

parser.add_argument('--gen_use_batch_norm', dest='gen_use_batch_norm', type=bool, default=True, help='Generator Using Batch Normalization or not')
parser.add_argument('--dis_use_batch_norm', dest='dis_use_batch_norm', type=bool, default=True, help='Discriminator Using Batch Normalization or not')
parser.add_argument('--use_lsgan', dest='use_lsgan', default=False, action='store_true', help='Using LSGAN loss or not, default is WGAN')

parser.add_argument('--num_train_gen', dest='num_train_gen', type=int, default=1, help='Iteration to train generator within one episode')
parser.add_argument('--num_train_dis', dest='num_train_dis', type=int, default=1, help='Iteration to train discriminator within one episode')
parser.add_argument('--num_train_enc', dest='num_train_enc', type=int, default=1, help='Iteration to train encoder within one episode')

parser.add_argument('--saving_path', dest='saving_path', default='../model2', help='Path to save model if specified')
parser.add_argument('--loading_path', dest='loading_path', default='../model2', help='Path to load model if specified')
parser.add_argument('--sampling_path', dest='sampling_path', default='../samples2', help='Path to save samples if specified')
parser.add_argument('--testing_path', dest='testing_path', default='../full_model_samples', help='Path to save testing samples if specified')
parser.add_argument('--tsne_path', dest='tsne_path', default='../tsne', help='Path to save testing tsne results')
parser.add_argument('--summary_dir', dest='summary_dir', default='../summary', help='Path to save summaries')

parser.add_argument('--gpu_fraction', dest='gpu_fraction', type=float, default=0.95, help='Fraction of GPU Memory to use')

parser.add_argument('--is_training', dest='is_training', default=True, action='store_false', help='True if training phase, while false if testing phase')
parser.add_argument('--use_fp16', dest='use_fp16', default=False, action='store_false', help='True if use float16 for tensorcore acceleration')
parser.add_argument('--fp16_scale', dest='fp16_scale', type=float, default=128, help='Scaling factor for fp16 computation')

parser.add_argument('--clip_to_value', dest='clip_to_value', type=float, default=3.0, help='Scaling factor for tanh')
parser.add_argument('--sr_lan', dest='sr_lan', default="en-US", help='Language for speech recognition')
parser.add_argument('--dprate', dest='dprate', type=float, default=0.9, help='Dropout rate for generator')
parser.add_argument('--dprated', dest='dprated', type=float, default=0.8, help='Dropout rate for discriminator')

## Audio Processing Hparams
parser.add_argument('--num_freq', dest='num_freq', type=int, default=513, help='Number of frequency bins for STFT')
parser.add_argument('--frame_shift', dest='frame_shift', type=float, default=12.5, help='Time length of frame shift in miliseconds for STFT')
parser.add_argument('--frame_length', dest='frame_length', type=int, default=50, help='Frame length for STFT')

##The model Structure Hyperparams, NOT CHANGEABLE from command line##
args = parser.parse_args()

args.gen_conv_filter_size = [5, 5, 5, 5]
args.gen_deconv_filter_size = [5, 5, 5, 5]
args.gen_conv_channel_size = [128, 256, 512, 1024]
args.gen_deconv_channel_size = [512, 256, 128, args.channel_size]
args.gen_stride = [2, 2, 2, 2]
args.gen_stridec = [2, 2, 2, 2]
args.gen_time_filter_size = [4] * 0

args.gen_conv_filter_size2d = [[5, 5]] * 5
args.gen_deconv_filter_size2d = [[5, 5]] * 5
args.gen_conv_channel_size2d = [64, 128, 256, 512, 1024]
args.gen_deconv_channel_size2d = [1024, 512, 256, 128, 1]
args.gen_stride2d = [[1, 1, 2, 4], #(64, 64)
                     [1, 1, 2, 2], #(32, 32)
                     [1, 1, 2, 2], #(16, 16)
                     [1, 1, 2, 2], #(8, 8)
                     [1, 1, 2, 2]] #(4, 4)
args.gen_stride2dc = [[1, 1, 2, 4], #(64, 64)
                     [1, 1, 2, 2], #(4, 4)
                     [1, 1, 2, 2], #(4, 4)
                     [1, 1, 2, 2], #(4, 4)
                     [1, 1, 2, 2]] #(4, 4)
args.gen_time_filter_size2d = [4] * 0

args.dis_filter_size = [5, 5, 5, 5]
args.dis_channel_size = [64, 32, 16, 8]
args.dis_stride = [2, 2, 2, 2] 

args.dis_filter_size2d = [[5, 5]] * 3 + [[3, 3]] * 1
args.dis_channel_size2d = np.array([64, 128, 256, 512])
args.dis_stride2d = [[1, 1, 2, 4],
                     [1, 1, 2, 4],
                     [1, 1, 2, 2],
                     [1, 1, 2, 2]]

args.n_fft = (args.num_freq - 1) * 2
args.hop_length = int(args.frame_shift / 1000. * args.sample_rate)
args.window_size = int(args.frame_length / 1000. * args.sample_rate)
args.melbasis = librosa.filters.mel(args.sample_rate, args.n_fft, n_mels=args.channel_size, fmin=55, fmax=7600)
args.melbasisinv = np.linalg.pinv(args.melbasis)
args.paired_training = args.p_data_dir is not None
