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

import textgrids

from utils.mp3_to_transcript import add_audio_data, obtain_transcripts_baseline
from utils.evaluate import evaluate_unaltered

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
    if args.noise != 0:
        print('Noise factor', args.noise)
    
    if os.path.exists(args.path+ args.model + '/' + args.test + args.remix + '_transcripts.json'):
        df = pd.read_json(args.path+ args.model + '/' + args.test + args.remix + '_transcripts.json')
    else:
        if args.noise == 0:
            # if os.path.exists(args.path + 'df_audio.json'):
            #     df = pd.read_json(args.path + 'df_audio.json')
            # else:
            #     print('No audio data found')
            df = add_audio_data(args.path, args.noise)
        else:
            name = 'df_audio'+ '_noise'+str(args.noise) +'.json'
            if os.path.exists(args.path + name):
                df = pd.read_json(args.path + name)
            else:
                print('No audio data found')
                df = add_audio_data(args.path, args.noise)

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

    # if args.model == 'whisper':
    #     prompt = "PLEASE CALL STELLA ASK HER TO BRING THESE THINGS WITH HER FROM THE STORE 6 SPOONS OF FRESH SNOW PEAS 5 THICK SLABS OF BLUE CHEESE AND MAYBE A SNACK FOR HER BROTHER BOB WE ALSO NEED A SMALL PLASTIC SNAKE AND A BIG TOY FROG FOR THE KIDS SHE CAN SCOOP THESE THINGS INTO 3 RED BAGS AND WE WILL GO MEET HER WEDNESDAY AT THE TRAIN STATION"
    # else:
    prompt = "PLEASE CALL STELLA ASK HER TO BRING THESE THINGS WITH HER FROM THE STORE SIX SPOONS OF FRESH SNOW PEAS FIVE THICK SLABS OF BLUE CHEESE AND MAYBE A SNACK FOR HER BROTHER BOB WE ALSO NEED A SMALL PLASTIC SNAKE AND A BIG TOY FROG FOR THE KIDS SHE CAN SCOOP THESE THINGS INTO THREE RED BAGS AND WE WILL GO MEET HER WEDNESDAY AT THE TRAIN STATION"

    evaluate_unaltered(df, prompt, args)
    return 0

main()