"""
Run to get the baseline results for unaltered speech.
Recieves the path to a folder containing the audio files and relevant dataframes (speakers.csv)

Example:
    python get_baselines.py --path ./data/unaltered/
                            --model w2v2_960
"""

import pandas as pd
import argparse
import torch
import os
from tqdm import tqdm
import numpy as np

import textgrids

from utils.mp3_to_transcript import add_audio_data, obtain_transcripts_baseline
from utils.evaluate import eval_df_slices

def parse_option():
    parser = argparse.ArgumentParser("Obtain transcripts from .mp3 files")

    parser.add_argument("--path", type=str, default='./data/unaltered/', help="path to the folder containing the audio and speakers")
    parser.add_argument("--test", type=str, default='trains', help="Text spoken in the segment to remix")
    parser.add_argument("--detailed", type=bool, default=False, help="Set to true to add total number of mistranscripted words and male-female breakdown to the output")
    parser.add_argument("--noise", type=float, default=0, help="Value between 0 and 1. How much noise to be added")
    parser.add_argument(
        "--model",
        type=str,
        default="w2v2_960",
        choices=[
            "w2v2_960",
            "voxpp_es",
            "voxpp_sk",
            "voxpp_it",
            "whisper",
            "whisper.en",
            "w2v2_xlsr",
            "w2v2_large"
        ],
        help="choose model to use for the transcriptions",
    )


    args = parser.parse_args()

    if args.test == 'trains':
        args.text = 'THE TRAIN STATION'
        args.test = 'trains_'
        args.word_idx = 17
    elif args.test == 'stella':
        args.text = 'PLEASE CALL STELLA'
        args.test = 'stella_'
        args.word_idx = 0
    elif args.test == 'bob':
        args.text = 'HER BROTHER BOB'
        args.test = 'bob_'
        args.word_idx = 8
    elif args.test == 'need':
        args.text = 'WE ALSO NEED A'
        args.test = 'need_'
        args.word_idx = 9 
    #
    # Extra baselines for the surprisal tests
    #
    elif args.test == 'ask':
        args.text = 'ASK HER TO'
        args.test = 'ask_'
        args.word_idx = 1
    elif args.test == 'store':
        args.text = 'FROM THE STORE'
        args.test = 'store_'
        args.word_idx = 3
    elif args.test == 'spoons':
        args.text = 'SIX SPOONS OF'
        args.test = 'spoons_'
        args.word_idx = 4
    elif args.test == 'cheese':
        args.text = 'OF BLUE CHEESE'
        args.test = 'cheese_'
        args.word_idx = 6
    elif args.test == 'snake':
        args.text = 'SMALL PLASTIC SNAKE'
        args.test = 'snake_'
        args.word_idx = 10 
    elif args.test == 'kids':
        args.text = 'FOR THE KIDS'
        args.test = 'kids_'
        args.word_idx = 12 
    elif args.test == 'scoop':
        args.text = 'SHE CAN SCOOP'
        args.test = 'scoop_'
        args.word_idx = 13 
    elif args.test == 'bags':
        args.text = 'THREE RED BAGS'
        args.test = 'bags_'
        args.word_idx = 15

    args.remix = 'unaltered'

    if args.noise > 0:
        args.remix = args.remix + '_noise'+str(args.noise)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args
    
def main():
    args = parse_option()

    random_seeds = [0, 10, 20, 30, 42]
    noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
                    1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2,
                    2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3
            ]
    models = [
            # "w2v2_960",
            # "w2v2_large",
            # "w2v2_xlsr",       
            # "whisper",
            "whisper.en",
        ]

    columns = ['noise_level', 'eng-wer', 'eng-wer_std', 'eng-cer', 'eng-cer_std', 'spa-wer', 'spa-wer_std', 'spa-cer', 'spa-cer_std']

    for model in models:
        args.model = model
        out_df = pd.DataFrame(columns=columns)
        for noise in noise_levels:
            args.noise = noise
            wer_eng = []
            cer_eng = []
            wer_spa = []
            cer_spa = []

            for seed in random_seeds:    
                df = add_audio_data(args.path, args.noise, save_file = False, seed = seed)

                # Create transcripts
                test_tp = []
                audio_len = []
                for idx in range(len(df)):
                    grid = textgrids.TextGrid(args.path + df.loc[idx, 'file_name'] + '.TextGrid')
                    audio_len.append(grid.xmax)
                    test_tp.append([grid['3WORDS'][args.word_idx].xmin, grid['3WORDS'][args.word_idx].xmax])

                df = df.assign(test_timepoints = test_tp)
                df = df.assign(audio_length = audio_len)

                df = obtain_transcripts_baseline(df, args)

                prompt = "PLEASE CALL STELLA ASK HER TO BRING THESE THINGS WITH HER FROM THE STORE SIX SPOONS OF FRESH SNOW PEAS FIVE THICK SLABS OF BLUE CHEESE AND MAYBE A SNACK FOR HER BROTHER BOB WE ALSO NEED A SMALL PLASTIC SNAKE AND A BIG TOY FROG FOR THE KIDS SHE CAN SCOOP THESE THINGS INTO THREE RED BAGS AND WE WILL GO MEET HER WEDNESDAY AT THE TRAIN STATION"

                ev_df = eval_df_slices(prompt.upper(), df, args)

                # Evaluate english speakers
                eng_eval = ev_df.loc[ev_df['language'] == 'eng']
                wer_eng.append(eng_eval.loc[:, 'wer'].mean())
                cer_eng.append(eng_eval.loc[:, 'cer'].mean())

                # Evaluate spanish speakers
                spa_eval = ev_df.loc[ev_df['language'] == 'spa']
                wer_spa.append(spa_eval.loc[:, 'wer'].mean())
                cer_spa.append(spa_eval.loc[:, 'cer'].mean())
            
            print('Done. Noise factor ', args.noise)
            print('Model ', args.model)
            print('English')
            print('mean wer', np.mean(wer_eng))
            print('std wer', np.std(wer_eng))
            print('mean cer', np.mean(cer_eng))
            print('std cer', np.std(cer_eng))
            print('\n Spanish')
            print('mean wer', np.mean(wer_spa))
            print('std wer', np.std(wer_spa))
            print('mean cer', np.mean(cer_spa))
            print('std cer', np.std(cer_spa))

            out_df.loc[len(out_df.index)] = [
                        noise,
                        np.mean(wer_eng), np.std(wer_eng), np.mean(cer_eng), np.std(cer_eng),
                        np.mean(wer_spa), np.std(wer_spa), np.mean(cer_spa), np.std(cer_spa)
                    ]
        out_df.to_json('noisy_baseline_'+ args.model +'.json')

    return 0

main()