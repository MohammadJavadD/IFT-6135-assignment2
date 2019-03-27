
import collections
import numpy as np
import os
import time
import torch
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt
from utils import load_model, init_device, ptb_raw_data, ModelInfo, Batch, ptb_iterator

def compute_loss(model, model_info, device, data, loss_fn):

    model.eval()
    all_losses = np.empty((0, 35))

    # LOOP THROUGH MINIBATCHES
    for step, (x, y) in tqdm.tqdm(enumerate(ptb_iterator(data, model.batch_size, model.seq_len)),
                                  total=(len(data)//model.batch_size - 1)//model.seq_len):
        if model_info.model == 'TRANSFORMER':
            batch = Batch(torch.from_numpy(x).long().to(device))
            model.zero_grad()
            outputs = model.forward(batch.data, batch.mask).transpose(1,0)
        else:
            inputs = torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous().to(device)#.cuda()
            model.zero_grad()
            hidden = model.init_hidden().to(device)
            outputs, hidden = model(inputs, hidden)

        # Target
        targets = torch.from_numpy(y.astype(np.int64)).transpose(0, 1).contiguous().to(device)

        # Loss computation
        outputs = outputs.contiguous()
        losses_in_batch = []
        for output_t, target_t in zip(outputs, targets):
            losses_in_batch.append(loss_fn(output_t, target_t).data.item())
        all_losses = np.vstack((all_losses, losses_in_batch))
    # Return
    return np.mean(all_losses, axis=0)


def plot_loss(model_infos, losses_by_model):
    # losses_by_model = np.load('losses_by_model.npy')
    results_folder = ''

    for model_info, losses in zip(model_infos, losses_by_model):
        plt.plot(np.arange(35) + 1, losses, label=model_info.model)
    plt.title("Validation Loss with respect to timesteps")
    plt.ylabel("Validation Loss")
    plt.xlabel("Timestep")
    plt.legend()
    file_name = 'loss_per_timestep.png'
    plt.savefig(os.path.join(results_folder, file_name), bbox_inches='tight', pad_inches=0.2)
    plt.clf()
    plt.close()


def compute_loss_by_model():
    device = init_device()
    train_data, valid_data, test_data, word_to_id, id_2_word = ptb_raw_data(data_path='data')
    vocab_size = len(word_to_id)
    loss_fn = nn.CrossEntropyLoss()
    # Models from 4_1
    models = [ModelInfo('RNN', 'ADAM', 0.0001, 20, 35, 1500, 2, 0.35),
                   ModelInfo('GRU','SGD_LR_SCHEDULE', 10, 20, 35, 1500, 2, 0.35),
                    ModelInfo('TRANSFORMER', 'ADAM', 20, 128, 35, 512, 6, 0.9)]

    losses_by_model = []
    for model_info in models:
        model = load_model(model_info, device, vocab_size)
        loss_per_step = compute_loss(model, model_info, device, valid_data, loss_fn)
        losses_by_model.append(loss_per_step)

    plot_loss(models, losses_by_model)


if __name__ == '__main__':
    compute_loss_by_model()