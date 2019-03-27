import matplotlib.pyplot as plt
import numpy as np
import os

model = 'GRU'
opt = 'SGD_LR_SCHEDULE'
dir = 'Probleme_4_2/'
dir1 = 'Probleme_4_3/'

tf_adam = dir+'TRANSFORMER/TRANSFORMER_ADAM_model=TRANSFORMER_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=.9_0'
tf_SGD = dir+'TRANSFORMER/TRANSFORMER_SGD_model=TRANSFORMER_optimizer=SGD_initial_lr=20_batch_size=128_seq_len=35_hidden_size=512_num_layers=6_dp_keep_prob=.9_0'
tf_SGD_LR = 'Probleme_4_1/TRANSFORMER'

rnn_adam = 'Probleme_4_1/RNN'
rnn_SGD = dir + 'RNN/RNN_SGD_model=RNN_optimizer=SGD_initial_lr=0.0001_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35'
rnn_SGD_LR = dir+'RNN/RNN_SGD_LR_SCHEDULE_model=RNN_optimizer=SGD_LR_SCHEDULE_initial_lr=1_batch_size=20_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.35'

gru_adam = dir + 'GRU/GRU_ADAM_model=GRU_optimizer=ADAM_initial_lr=0.0001_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_0'
gru_SGD = dir + 'GRU/GRU_SGD_model=GRU_optimizer=SGD_initial_lr=10_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_0'
gru_SGD_LR = 'Probleme_4_1/GRU'


if model == 'TRANSFORMER':
    dirs = [tf_adam,tf_SGD,tf_SGD_LR]
if model == 'RNN':
    dirs = [rnn_adam,rnn_SGD,rnn_SGD_LR]
if model == 'GRU':
    dirs = [gru_adam,gru_SGD,gru_SGD_LR]
'''
if opt == 'ADAM':
    dirs = [rnn_adam,gru_adam,tf_adam]
if opt == 'SGD':
    dirs = [rnn_SGD,gru_SGD,tf_SGD]
if opt == 'SGD_LR_SCHEDULE':
    dirs = [rnn_SGD_LR,gru_SGD_LR,tf_SGD_LR]'''

labels = ['RNN','GRU','TRANSFORMER']
colors = ['C'+str(i) for i in range(len(dirs))]

train_ppls_all, val_ppls_all, epochs_all, times_all = [], [], [], []
for r in dirs:
    lc_path = os.path.join(r, 'learning_curves.npy')
    a = np.load(lc_path)[()]
    train_ppls, val_ppls, times= a['train_ppls'], a['val_ppls'],a['times']
    epochs = np.arange(40)+1
    times = np.cumsum(times)
    train_ppls_all.append(train_ppls)
    val_ppls_all.append(val_ppls)
    epochs_all.append(epochs)
    times_all.append(times)

plt.figure(figsize=(12,10))
plt.subplot(2,1,1)
for i in range(len(dirs)):
    #plt.plot(epochs_all[i], train_ppls_all[i], color=colors[i], label=labels[i] + " Train PPL")
    plt.plot(epochs_all[i], val_ppls_all[i], color=colors[i], label=labels[i] + " Val PPL")
    plt.legend()
    plt.ylim([0, 1000])
    plt.title("PPL with respect to Epochs")
    plt.ylabel("PPL")
    plt.xlabel("Epochs")


plt.subplot(212)
for i in range(len(dirs)):
    #plt.plot(times_all[i], train_ppls_all[i], color=colors[i], label=labels[i] + " Train PPL")
    plt.plot(times_all[i], val_ppls_all[i], color=colors[i], label=labels[i] + " Val PPL")
    plt.legend()
    plt.ylim([0, 1000])
    plt.title("PPL with respect to Wall clock time")
    plt.ylabel("PPL")
    plt.xlabel("Wall clock time (seconds)")

plt.suptitle('Model comparison with '+ opt + ' optimizer')
plt.savefig(opt+'_comparison_opt.png')
