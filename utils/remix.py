"""
Functions to build audio files such that different sections are spoken by different speakers.
"""
from typing import List, Set, Dict, Tuple

import pandas as pd
import pydub
from pydub import AudioSegment
import textgrids

from tqdm import tqdm

from utils.mp3_to_transcript import convert_to_audio_data


def get_segments_indices(grid_file_name:str, word_idx:int) -> List[float]:
    """
    Recieves the textgrid and first word index and returns the time indices at which each segment starts.
    The segments it returns are:
        1. Please call... ...from the store:
        2. Six spoons... ...her brother Bob.
        3. We also need... ...for the kids.
        4. She can scoop... ...meet her Wednesday
        5. at the train station

    Inputs:
        grid_file_name (str) : path to the TextGrid from which to get the timestamps
        first_word_idx (int) : index under the tier "words" of the first word of segment 5

    Output:
        indices (list[float]) : list of time indices where each segment starts in the audio file
    """
    grid = textgrids.TextGrid(grid_file_name)

    indices = []
    if word_idx == 0:
        indices.append(1000 * grid['3WORDS'][word_idx].xmax)
    else:
        indices.append(0)

    indices.append(1000 * grid['QUARTERS'][0].xmax)
    indices.append(1000 * grid['QUARTERS'][1].xmax)
    indices.append(1000 * grid['QUARTERS'][2].xmax)
    if word_idx == 0:
        indices.append(1000 * grid['QUARTERS'][3].xmax)
    else:
        indices.append(1000 * grid['3WORDS'][word_idx].xmin)

    return indices

def sentence_indices(grid_file_name, test):
    grid = textgrids.TextGrid(grid_file_name)
    aux = []

    aux.append(0)
    if test == 'stella_':
        aux.append(1000 * grid['3WORDS'][0].xmax)
    aux.append(1000 * grid['SENTENCES'][0].xmax)
    aux.append(1000 * grid['SENTENCES'][1].xmax)
    aux.append(1000 * grid['SENTENCES'][2].xmax)
    aux.append(1000 * grid['SENTENCES'][3].xmax)
    if test == 'bob_':
        aux.append(1000 * grid['3WORDS'][8].xmin)
    aux.append(1000 * grid['SENTENCES'][4].xmax)
    if test == 'need_':
        aux.append(1000 * grid['3WORDS'][9].xmax)
    aux.append(1000 * grid['SENTENCES'][5].xmax)
    aux.append(1000 * grid['SENTENCES'][6].xmax)
    if test == 'trains_':
        aux.append(1000 * grid['3WORDS'][17].xmin)
    aux.append(1000 * grid['SENTENCES'][7].xmax)

    return aux

def remix_to_three_varying(args):
    columns = ['test', 'remix', 'accent1_segments', 'gender', 'test_timepoints', 'audio_length', 'audio_data', 'transcript', 'transcript_test']
    df = pd.DataFrame(columns = columns)

    speakers = pd.read_csv(args.unaltered_path + 'speakers.csv')

    for gender in ['M', 'F']:
        g_speakers = speakers.loc[speakers['gender'] == gender]

        for test_spkr in tqdm(g_speakers.loc[g_speakers['language'] == args.lang_list[2]]['file_name'].to_list(), desc='Creating the mixed audios for gender ' + gender + '...'):
            aux = AudioSegment.from_file(args.unaltered_path + test_spkr + '.wav')
            grid = textgrids.TextGrid(args.unaltered_path + test_spkr + '.TextGrid')

            start_point = 1000 * grid['3WORDS'][args.word_idx].xmin
            end_point = 1000 * grid['3WORDS'][args.word_idx].xmax
            test = aux[start_point:end_point]

            for spkr2 in g_speakers.loc[g_speakers['language'] == args.lang_list[1]]['file_name'].to_list():
                if spkr2 == test_spkr:
                    continue
                audio2 = AudioSegment.from_file(args.unaltered_path + spkr2 + '.wav')
                spkr2_segments = get_segments_indices(args.unaltered_path + spkr2 + '.TextGrid', args.word_idx)

                for spkr1 in g_speakers.loc[g_speakers['language'] == args.lang_list[0]]['file_name'].to_list():
                    if spkr1 == test_spkr or spkr1 == spkr2:
                        continue
                    audio1 = AudioSegment.from_file(args.unaltered_path + spkr1 + '.wav')
                    spkr1_segments = get_segments_indices(args.unaltered_path + spkr1 + '.TextGrid', args.word_idx)

                    for i in [0, 1, 2, 3, 4]:
                        tp = [0, 0]
                        middle_audio = audio1[spkr1_segments[0]:spkr1_segments[i]] + audio2[spkr2_segments[i]:spkr2_segments[len(spkr2_segments)-1]]
                        if args.text == 'THE TRAIN STATION':
                            audio = middle_audio + test
                            audio_length = len(audio)
                            tp[0] = len(middle_audio)
                            tp[1] = audio_length
                        elif args.text == 'PLEASE CALL STELLA':
                            audio = test + middle_audio
                            tp[0] = 0
                            tp[1] = len(test)
                            audio_length = len(audio)

                        audio_data = convert_to_audio_data(audio)

                        df.loc[len(df.index)] = [args.text, args.remix, i, gender, tp, audio_length, audio_data, '', ''] 

    df.to_json(args.path + args.test + args.remix + '_audio.json')

    return df


def remix_to_three_middle(args):
    columns = ['test', 'remix', 'accent1_segments', 'gender', 'test_timepoints', 'audio_length', 'audio_data', 'transcript', 'transcript_test']
    df = pd.DataFrame(columns = columns)

    speakers = pd.read_csv(args.unaltered_path + 'speakers.csv')

    for gender in ['M', 'F']:
        g_speakers = speakers.loc[speakers['gender'] == gender]

        for test_spkr in tqdm(g_speakers.loc[g_speakers['language'] == args.lang_list[2]]['file_name'].to_list(), desc='Creating the mixed audios for gender ' + gender + '...'):
            aux = AudioSegment.from_file(args.unaltered_path + test_spkr + '.wav')
            grid = textgrids.TextGrid(args.unaltered_path + test_spkr + '.TextGrid')

            start_point = 1000 * grid['3WORDS'][args.word_idx].xmin
            end_point = 1000 * grid['3WORDS'][args.word_idx].xmax
            test = aux[start_point:end_point]

            for spkr2 in g_speakers.loc[g_speakers['language'] == args.lang_list[1]]['file_name'].to_list():
                if spkr2 == test_spkr:
                    continue
                audio2 = AudioSegment.from_file(args.unaltered_path + spkr2 + '.wav')
                spkr2_segments = sentence_indices(args.unaltered_path + spkr2 + '.TextGrid', args.test)

                for spkr1 in g_speakers.loc[g_speakers['language'] == args.lang_list[0]]['file_name'].to_list():
                    if spkr1 == test_spkr or spkr1 == spkr2:
                        continue
                    audio1 = AudioSegment.from_file(args.unaltered_path + spkr1 + '.wav')
                    spkr1_segments = sentence_indices(args.unaltered_path + spkr1 + '.TextGrid', args.test)

                    tp = [0, 0]
                    beg = audio1[spkr1_segments[0]:spkr1_segments[5]]
                    end = audio2[spkr2_segments[6]:spkr2_segments[len(spkr2_segments)-1]]
                    audio = beg + test + end
                    audio_length = len(audio)
                    tp[0] = len(beg)
                    tp[1] = len(beg) + len(test)

                    audio_data = convert_to_audio_data(audio)

                    df.loc[len(df.index)] = [args.text, args.remix, 2, gender, tp, audio_length, audio_data, '', ''] 

    df.to_json(args.path + args.test + args.remix + '_audio.json')

    return df