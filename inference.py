import argparse
import os
import os.path

import torch
from tqdm import tqdm
from scipy.io.wavfile import write

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-i', '--input', required=True)
    args = parser.parse_args()

    hps = utils.get_hparams_from_file(args.config)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cpu()
    _ = net_g.eval()
    _ = utils.load_checkpoint(args.model, net_g, None)

    os.makedirs(args.output, exist_ok=True)

    with open(args.input) as f:
        for i, text in enumerate(tqdm(f)):
            if text.strip() == '': continue
            stn_tst = get_text(text, hps)
            with torch.no_grad():
                x_tst = stn_tst.cpu().unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()
                sid = torch.LongTensor([4]).cpu()
                audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
            write(os.path.join(args.output, str(i) + '.wav'), hps.data.sampling_rate, audio)


if __name__ == '__main__':
    main()
