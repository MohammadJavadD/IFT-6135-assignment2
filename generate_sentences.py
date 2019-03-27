import torch
import numpy as np
from utils import ptb_raw_data, load_model, init_device, ModelInfo,repackage_hidden
from itertools import product

def generate_sentences(model_info, device, seq_len, batch_size=10, start_word=None):

    train_data, valid_data, test_data, word_to_id, id_2_word = ptb_raw_data(data_path='data')
    vocab_size = len(word_to_id)
    start = [word_to_id["the"]]
    start.append(word_to_id["a"])
    start.append(word_to_id["an"])
    start.append(word_to_id["he"])
    start.append(word_to_id["she"])
    start.append(word_to_id["it"])
    start.append(word_to_id["they"])
    start.append(word_to_id["why"])
    start.append(word_to_id["how"])
    start.append(word_to_id["to"])

    model = load_model(model_info, device, vocab_size=vocab_size, load_on_device=False)

    hidden = torch.zeros(model.num_layers, batch_size, model.hidden_size)
    hidden = repackage_hidden(hidden)
    samples = model.generate(start, hidden, seq_len)
    generated_sentences = []

    for i,sample in enumerate(samples.to("cpu"),1):
        sentence = " ".join([id_2_word[int(word)] for word in sample])
        generated_sentences.append(sentence)

    with open('%s_%s_generated_samples.txt'%(model_info.model,str(seq_len)), 'w') as f:
        for sentence in generated_sentences:
            f.write("%s\n\n" % sentence)


if __name__ == "__main__":
    seq_lens = [35, 70]
    generations_per_seq_len = 10
    starting_word = "<eos>"
    models = [ModelInfo('RNN', 'ADAM', 0.0001, 20, 35, 1500, 2, 0.35),
              ModelInfo('GRU', 'SGD_LR_SCHEDULE', 10, 20, 35, 1500, 2, 0.35)]

    device = init_device()
    for m, s in product(models, seq_lens):
        generate_sentences(m, device, s, generations_per_seq_len, starting_word)