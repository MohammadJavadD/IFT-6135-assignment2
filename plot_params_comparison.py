import matplotlib.pyplot as plt
import numpy as np
import os
import re
path = "Probleme_4_2/GRU/GRU_ADAM_model=GRU_optimizer=ADAM_initial_lr=0.0001_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_0"
model = 'GRU/'
p_dir = 'Probleme_4_3/'

labels = []

'''for i,dir in enumerate(os.listdir(p_dir+model)):
    model = re.search('model=(.*)_optimizer', p_dir+model+dir, re.IGNORECASE).group(1)
    opt = re.search('optimizer=(.*)_initial_lr', p_dir+model+dir, re.IGNORECASE).group(1)
    initial_lr = re.search('initial_lr=(.*)_batch_size', p_dir+model+dir, re.IGNORECASE).group(1)
    batch_size = re.search('batch_size=(.*)_seq_len', p_dir+model+dir, re.IGNORECASE).group(1)
    seq_len = re.search('seq_len=(.*)_hidden_size', p_dir+model+dir, re.IGNORECASE).group(1)
    hidden_size = re.search('hidden_size=(.*)_num_layers', p_dir+model+dir, re.IGNORECASE).group(1)
    nums_layer = re.search('num_layers=(.*)_dp_keep_prob', p_dir+model+dir, re.IGNORECASE).group(1)
    dp = re.search('dp_keep_prob=(.*)_', p_dir+model+dir, re.IGNORECASE).group(1)

    labels.append("%s opt=%s, lr=%s, batch_size=%s, seq=%s, h_size=%s, layers=%s, dp=%s" % \
                      (model, opt, initial_lr, batch_size, seq_len, hidden_size, nums_layer, dp))'''


train_ppls_all, val_ppls_all, epochs_all, times_all = [], [], [], []
#for r in os.listdir(p_dir+model):
lc_path = os.path.join(path, 'learning_curves.npy')
a = np.load(lc_path)[()]
train_ppls, val_ppls, times= a['train_ppls'], a['val_ppls'],a['times']
epochs = np.arange(40)+1
times = np.cumsum(times)
train_ppls_all.append(train_ppls)
val_ppls_all.append(val_ppls)
epochs_all.append(epochs)
times_all.append(times)


#for i in range(len(labels)):
plt.figure(figsize=(12,10))
plt.subplot(2, 1, 1)
plt.plot(epochs,val_ppls, '--', label="GRU opt=ADAM, lr=0.0001, batch_size=20, seq=35, h_size=1500, layers=2, dp=0.35" + " Val PPL")
plt.plot(epochs, train_ppls, label="GRU opt=ADAM, lr=0.0001, batch_size=20, seq=35, h_size=1500, layers=2, dp=0.35" + " Train PPL")
plt.legend(prop={'size':6})
plt.ylim([0, 1000])
plt.ylabel("PPL")
plt.xlabel("Epochs")
plt.title("PPL with respect to Epochs")
plt.subplot(2, 1, 2)
plt.plot(times,val_ppls, '--', label="GRU opt=ADAM, lr=0.0001, batch_size=20, seq=35, h_size=1500, layers=2, dp=0.35" + " Val PPL")
plt.plot(times, train_ppls, label="GRU opt=ADAM, lr=0.0001, batch_size=20, seq=35, h_size=1500, layers=2, dp=0.35" + " Train PPL")
plt.legend(prop={'size':6})
plt.ylim([0, 1000])
plt.ylabel("PPL")
plt.xlabel("Wall clock time (seconds)")
plt.title("PPL with respect to wall clock time")
plt.savefig('exp4.png')
plt.close()

