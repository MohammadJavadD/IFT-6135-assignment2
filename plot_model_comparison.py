import matplotlib.pyplot as plt
import numpy as np
import os

rnn = 'Probleme_4_1/RNN'
gru = 'Probleme_4_1/GRU'
transformer = 'Probleme_4_1/TRANSFORMER'

dirs = [rnn,gru,transformer]
labels = ['RNN','GRU','Transformer']
colors = ['C'+str(i) for i in range(len(dirs))]

# Read learning_curves
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
    plt.plot(epochs_all[i], train_ppls_all[i],color=colors[i], label=labels[i]+" Train PPL")
    plt.plot(epochs_all[i], val_ppls_all[i], '--', color=colors[i], label=labels[i]+" Val PPL")
    plt.legend()
    plt.ylim([0, 1000])
    plt.title("PPL with respect to Epochs")
    plt.ylabel("PPL")
    plt.xlabel("Epochs")

plt.subplot(212)
for i in range(len(dirs)):
    plt.plot(times_all[i], train_ppls_all[i],color=colors[i], label=labels[i]+" Train PPL")
    plt.plot(times_all[i], val_ppls_all[i], '--',color=colors[i], label=labels[i]+" Val PPL")
    plt.legend()
    plt.ylim([0, 1000])
    plt.title("PPL with respect to Wall clock time")
    plt.ylabel("PPL")
    plt.xlabel("Wall clock time (seconds)")

plt.savefig('model_comparison.png')
plt.clf()
plt.close()
