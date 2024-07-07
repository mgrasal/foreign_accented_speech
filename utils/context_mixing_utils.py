import pandas as pd
import matplotlib.pyplot as plt

import textgrids
from pydub import AudioSegment

from utils.mp3_to_transcript import convert_to_audio_data

def sentence_indices(args, test):
    indices = []
    i=0
    for speaker in args.speakers:
        grid = textgrids.TextGrid(args.unaltered_path + speaker + '.TextGrid')
        aux = []

        aux.append(0)
        if test == 'stella':
            aux.append(1000 * grid['3WORDS'][0].xmax)
        aux.append(1000 * grid['SENTENCES'][0].xmax)
        aux.append(1000 * grid['SENTENCES'][1].xmax)
        aux.append(1000 * grid['SENTENCES'][2].xmax)
        aux.append(1000 * grid['SENTENCES'][3].xmax)
        if test == 'bob':
            aux.append(1000 * grid['3WORDS'][8].xmin)
        aux.append(1000 * grid['SENTENCES'][4].xmax)
        if test == 'need':
            aux.append(1000 * grid['3WORDS'][9].xmax)
        aux.append(1000 * grid['SENTENCES'][5].xmax)
        aux.append(1000 * grid['SENTENCES'][6].xmax)
        if test == 'trains':
            aux.append(1000 * grid['3WORDS'][17].xmin)
        aux.append(1000 * grid['SENTENCES'][7].xmax)

        indices.append(aux)

    return indices

def add_remix_arrays(df, args):
    eng1_audio = AudioSegment.from_file(args.unaltered_path + args.speakers[0] + '.wav')    
    eng2_audio = AudioSegment.from_file(args.unaltered_path + args.speakers[1] + '.wav')    
    spa1_audio = AudioSegment.from_file(args.unaltered_path + args.speakers[2] + '.wav')
    spa2_audio = AudioSegment.from_file(args.unaltered_path + args.speakers[3] + '.wav')
    test_audio = AudioSegment.from_file(args.unaltered_path + args.speakers[4] + '.wav')

    # stella
    timepoints = sentence_indices(args, 'stella')
    test = test_audio[timepoints[4][0]:timepoints[4][1]]
    # The speaker change (when applicable) will occur between sentences 4 and 5 (0-indexed)
    # This will correspond to timepoints[0][6] if test == 'stella'
    eng1 = eng1_audio[timepoints[0][1]:timepoints[0][6]]
    eng2 = eng2_audio[timepoints[1][6]:timepoints[1][9]]
    spa1 = spa1_audio[timepoints[2][1]:timepoints[2][6]]
    spa2 = spa2_audio[timepoints[3][6]:timepoints[3][9]]

    df.loc[0].audio_array = convert_to_audio_data(test+eng1+eng2)
    df.loc[1].audio_array = convert_to_audio_data(test+eng1+spa2)
    df.loc[2].audio_array = convert_to_audio_data(test+spa1+eng2)
    df.loc[3].audio_array = convert_to_audio_data(test+spa1+spa2)

    # trains
    timepoints = sentence_indices(args, 'trains')
    test = test_audio[timepoints[4][8]:timepoints[4][9]]
    # The speaker change (when applicable) will occur between sentences 4 and 5 (0-indexed)
    # This will correspond to timepoints[0][5] if test == 'trains'
    eng1 = eng1_audio[timepoints[0][0]:timepoints[0][5]]
    eng2 = eng2_audio[timepoints[1][5]:timepoints[1][8]]
    spa1 = spa1_audio[timepoints[2][0]:timepoints[2][5]]
    spa2 = spa2_audio[timepoints[3][5]:timepoints[3][8]]

    df.loc[4].audio_array = convert_to_audio_data(eng1+eng2+test)
    df.loc[5].audio_array = convert_to_audio_data(eng1+spa2+test)
    df.loc[6].audio_array = convert_to_audio_data(spa1+eng2+test)
    df.loc[7].audio_array = convert_to_audio_data(spa1+spa2+test)

    # bob
    timepoints = sentence_indices(args, 'bob')
    test = test_audio[timepoints[4][5]:timepoints[4][6]]
    
    eng1 = eng1_audio[timepoints[0][0]:timepoints[0][5]]
    eng2 = eng2_audio[timepoints[1][6]:timepoints[1][9]]
    spa1 = spa1_audio[timepoints[2][0]:timepoints[2][5]]
    spa2 = spa2_audio[timepoints[3][6]:timepoints[3][9]]

    df.loc[8].audio_array = convert_to_audio_data(eng1+test+eng2)
    df.loc[9].audio_array = convert_to_audio_data(eng1+test+spa2)
    df.loc[10].audio_array = convert_to_audio_data(spa1+test+eng2)
    df.loc[11].audio_array = convert_to_audio_data(spa1+test+spa2)

    # need
    timepoints = sentence_indices(args, 'need')
    test = test_audio[timepoints[4][5]:timepoints[4][6]]
    
    eng1 = eng1_audio[timepoints[0][0]:timepoints[0][5]]
    eng2 = eng2_audio[timepoints[1][6]:timepoints[1][9]]
    spa1 = spa1_audio[timepoints[2][0]:timepoints[2][5]]
    spa2 = spa2_audio[timepoints[3][6]:timepoints[3][9]]

    df.loc[12].audio_array = convert_to_audio_data(eng1+test+eng2)
    df.loc[13].audio_array = convert_to_audio_data(eng1+test+spa2)
    df.loc[14].audio_array = convert_to_audio_data(spa1+test+eng2)
    df.loc[15].audio_array = convert_to_audio_data(spa1+test+spa2)

    # test.export("test.wav", format="wav")
    # eng1.export("eng1.wav", format="wav")
    # eng2.export("eng2.wav", format="wav")
    # spa1.export("spa1.wav", format="wav")
    # spa2.export("spa2.wav", format="wav")

    return df

def create_remix_df(args):
    speakers = pd.read_csv(args.unaltered_path + 'speakers.csv')

    assert speakers.loc[speakers['file_name'] == args.eng_speaker]['language'].item() == 'eng', "Speaker indicated as english-accented is not english accented!"
    assert speakers.loc[speakers['file_name'] == args.spa_speaker]['language'].item() == 'spa', "Speaker indicated as spanish-accented is not spanish accented!"

    gender = speakers.loc[speakers['file_name'] == args.eng_speaker]['gender'].item()
    if speakers.loc[speakers['file_name'] == args.spa_speaker]['gender'].item() != gender:
        print('Warning: not all of the three speakers selected are of the same gender.\n To minimize the effect of different voices, it is recommended to choose speakers of similar voices, and of the same gender.')
        print('Discrepancy detected between '+args.eng_speaker+' and '+args.spa_speaker)
    elif speakers.loc[speakers['file_name'] == args.test_speaker]['gender'].item() != gender:
        print('Warning: not all of the three speakers selected are of the same gender.\n To minimize the effect of different voices, it is recommended to choose speakers of similar voices, and of the same gender.')
        print('Discrepancy detected between '+args.eng_speaker+' and '+args.test_speaker)

    columns = ['test', 'remix', 'timepoints', 'audio_array']
    df = pd.DataFrame(columns = columns)
    
    for test in ['stella', 'trains', 'bob', 'need']:
        for remix in ['eng_eng', 'eng_spa', 'spa_eng', 'spa_spa']:
            df.loc[len(df.index)] = [test, remix, [], []] 

    df = add_remix_arrays(df, args)

    # Add information for all cases
    tp = sentence_indices(args, 'stella')
    # The speaker change (when applicable) will occur between sentences 4 and 5 (0-indexed)
    # This will correspond to timepoints[0][6] if test == 'stella'
    # Note that in the case of stella indices 1 and 2 are the same, since the test segment is a whole sentence (3 words long)
    df.loc[0].timepoints = [
        tp[4][0], tp[4][1],
        tp[4][1]-tp[0][1]+tp[0][3], tp[4][1]-tp[0][1]+tp[0][4], tp[4][1]-tp[0][1]+tp[0][5], tp[4][1]-tp[0][1]+tp[0][6], 
        tp[4][1]-tp[0][1]+tp[0][5]-tp[1][5]+tp[1][7], tp[4][1]-tp[0][1]+tp[0][5]-tp[1][5]+tp[1][8], tp[4][1]-tp[0][1]+tp[0][5]-tp[1][5]+tp[1][9]
    ]
    df.loc[1].timepoints = [
        tp[4][0], tp[4][1],
        tp[4][1]-tp[0][1]+tp[0][3], tp[4][1]-tp[0][1]+tp[0][4], tp[4][1]-tp[0][1]+tp[0][5], tp[4][1]-tp[0][1]+tp[0][6], 
        tp[4][1]-tp[0][1]+tp[0][5]-tp[3][5]+tp[3][7], tp[4][1]-tp[0][1]+tp[0][5]-tp[3][5]+tp[3][8], tp[4][1]-tp[0][1]+tp[0][5]-tp[3][5]+tp[3][9]
    ]
    df.loc[2].timepoints = [
        tp[4][0], tp[4][1],
        tp[4][1]-tp[2][1]+tp[2][3], tp[4][1]-tp[2][1]+tp[2][4], tp[4][1]-tp[2][1]+tp[2][5], tp[4][1]-tp[2][1]+tp[2][6], 
        tp[4][1]-tp[2][1]+tp[2][5]-tp[1][5]+tp[1][7], tp[4][1]-tp[2][1]+tp[2][5]-tp[1][5]+tp[1][8], tp[4][1]-tp[2][1]+tp[2][5]-tp[1][5]+tp[1][9]
    ]
    df.loc[3].timepoints = [
        tp[4][0], tp[4][1],
        tp[4][1]-tp[2][1]+tp[2][3], tp[4][1]-tp[2][1]+tp[2][4], tp[4][1]-tp[2][1]+tp[2][5], tp[4][1]-tp[2][1]+tp[2][6], 
        tp[4][1]-tp[2][1]+tp[2][5]-tp[3][5]+tp[3][7], tp[4][1]-tp[2][1]+tp[2][5]-tp[3][5]+tp[3][8], tp[4][1]-tp[2][1]+tp[2][5]-tp[3][5]+tp[3][9]
    ]

    tp = sentence_indices(args, 'trains')
    # The speaker change (when applicable) will occur between sentences 4 and 5 (0-indexed)
    # This will correspond to timepoints[0][5] if test == 'trains'
    df.loc[4].timepoints = [
        tp[0][0], tp[0][1], tp[0][2], tp[0][3], tp[0][4], tp[0][5],
        tp[0][5]-tp[1][5]+tp[1][6],tp[0][5]-tp[1][5]+tp[1][7],
        tp[0][5]-tp[1][5]+tp[1][8], tp[0][5]-tp[1][5]+tp[1][8]-tp[4][8]+tp[4][9]
    ]
    df.loc[5].timepoints = [
        tp[0][0], tp[0][1], tp[0][2], tp[0][3], tp[0][4], tp[0][5],
        tp[0][5]-tp[3][5]+tp[3][6],tp[0][5]-tp[3][5]+tp[3][7],
        tp[0][5]-tp[3][5]+tp[3][8], tp[0][5]-tp[3][5]+tp[3][8]-tp[4][8]+tp[4][9]
    ]
    df.loc[6].timepoints = [
        tp[2][0], tp[2][1], tp[2][2], tp[2][3], tp[2][4], tp[2][5],
        tp[2][5]-tp[1][5]+tp[1][6],tp[2][5]-tp[1][5]+tp[1][7],
        tp[2][5]-tp[1][5]+tp[1][8], tp[2][5]-tp[1][5]+tp[1][8]-tp[4][8]+tp[4][9]
    ]
    df.loc[7].timepoints = [
        tp[2][0], tp[2][1], tp[2][2], tp[2][3], tp[2][4], tp[2][5],
        tp[2][5]-tp[3][5]+tp[3][6],tp[2][5]-tp[3][5]+tp[3][7],
        tp[2][5]-tp[3][5]+tp[3][8], tp[2][5]-tp[3][5]+tp[3][8]-tp[4][8]+tp[4][9]
    ]

    tp = sentence_indices(args, 'bob')
    df.loc[8].timepoints = [
        tp[0][0], tp[0][1], tp[0][2], tp[0][3], tp[0][4], tp[0][5],
        tp[0][5]-tp[4][5]+tp[4][6],
        tp[0][5]-tp[4][5]+tp[4][6]-tp[1][6]+tp[1][7], tp[0][5]-tp[4][5]+tp[4][6]-tp[1][6]+tp[1][8], tp[0][5]-tp[4][5]+tp[4][6]-tp[1][6]+tp[1][9]
    ]
    df.loc[9].timepoints = [
        tp[0][0], tp[0][1], tp[0][2], tp[0][3], tp[0][4], tp[0][5],
        tp[0][5]-tp[4][5]+tp[4][6],
        tp[0][5]-tp[4][5]+tp[4][6]-tp[3][6]+tp[3][7], tp[0][5]-tp[4][5]+tp[4][6]-tp[3][6]+tp[3][8], tp[0][5]-tp[4][5]+tp[4][6]-tp[3][6]+tp[3][9]
    ]
    df.loc[10].timepoints = [
        tp[2][0], tp[2][1], tp[2][2], tp[2][3], tp[2][4], tp[2][5],
        tp[2][5]-tp[4][5]+tp[4][6],
        tp[2][5]-tp[4][5]+tp[4][6]-tp[1][6]+tp[1][7], tp[2][5]-tp[4][5]+tp[4][6]-tp[1][6]+tp[1][8], tp[2][5]-tp[4][5]+tp[4][6]-tp[1][6]+tp[1][9]
    ]
    df.loc[11].timepoints = [
        tp[2][0], tp[2][1], tp[2][2], tp[2][3], tp[2][4], tp[2][5],
        tp[2][5]-tp[4][5]+tp[4][6],
        tp[2][5]-tp[4][5]+tp[4][6]-tp[3][6]+tp[3][7], tp[2][5]-tp[4][5]+tp[4][6]-tp[3][6]+tp[3][8], tp[2][5]-tp[4][5]+tp[4][6]-tp[3][6]+tp[3][9]
    ]

    tp = sentence_indices(args, 'need')
    df.loc[12].timepoints = [
        tp[0][0], tp[0][1], tp[0][2], tp[0][3], tp[0][4], tp[0][5],
        tp[0][5]-tp[4][5]+tp[4][6],
        tp[0][5]-tp[4][5]+tp[4][6]-tp[1][6]+tp[1][7], tp[0][5]-tp[4][5]+tp[4][6]-tp[1][6]+tp[1][8], tp[0][5]-tp[4][5]+tp[4][6]-tp[1][6]+tp[1][9]
    ]
    df.loc[13].timepoints = [
        tp[0][0], tp[0][1], tp[0][2], tp[0][3], tp[0][4], tp[0][5],
        tp[0][5]-tp[4][5]+tp[4][6],
        tp[0][5]-tp[4][5]+tp[4][6]-tp[3][6]+tp[3][7], tp[0][5]-tp[4][5]+tp[4][6]-tp[3][6]+tp[3][8], tp[0][5]-tp[4][5]+tp[4][6]-tp[3][6]+tp[3][9]
    ]
    df.loc[14].timepoints = [
        tp[2][0], tp[2][1], tp[2][2], tp[2][3], tp[2][4], tp[2][5],
        tp[2][5]-tp[4][5]+tp[4][6],
        tp[2][5]-tp[4][5]+tp[4][6]-tp[1][6]+tp[1][7], tp[2][5]-tp[4][5]+tp[4][6]-tp[1][6]+tp[1][8], tp[2][5]-tp[4][5]+tp[4][6]-tp[1][6]+tp[1][9]
    ]
    df.loc[15].timepoints = [
        tp[2][0], tp[2][1], tp[2][2], tp[2][3], tp[2][4], tp[2][5],
        tp[2][5]-tp[4][5]+tp[4][6],
        tp[2][5]-tp[4][5]+tp[4][6]-tp[3][6]+tp[3][7], tp[2][5]-tp[4][5]+tp[4][6]-tp[3][6]+tp[3][8], tp[2][5]-tp[4][5]+tp[4][6]-tp[3][6]+tp[3][9]
    ]

    df.to_json(args.path + args.mix + '.json')
    return df

def set_plot_background(axs, i, df, title, max_len, args):

    colors_dict = {
        'english1' : 'blue',
        'english90' : 'cyan',
        'spanish26' : 'red',
        'spanish74' : 'orange',
        'spanish27' : 'green',
        'english35' : 'navy',
        'english54' : 'deepskyblue',
        'spanish22' : 'salmon',
        'spanish114' : 'gold',
        'spanish197' : 'green',
        'english95' : 'seagreen',
    }

    axs.set_xlabel('Frame')
    axs.set_ylabel('Attention weight')

    axs.set_title(title)

    frames = []
    for timepoint in df.loc[i].timepoints:
        frame = (timepoint/df.loc[i].timepoints[-1])*max_len
        axs.axvline(frame, color='k', linestyle='dashed')
        frames.append(frame)
        axs.set_ylim(bottom=0, auto = True)

    if df.loc[i].test == 'stella':
        axs.axvline(frames[0], color='k')
        axs.axvline(frames[1], color='k')
        axs.axvspan(frames[0], frames[1], alpha=0.15, color = colors_dict[args.test_speaker], label = args.test_speaker)
        axs.axvline(frames[5], color='k')
        if df.loc[i].remix == 'eng_eng':
            axs.axvspan(frames[1], frames[5], alpha=0.1, color = colors_dict[args.eng_speaker], label = args.eng_speaker)
            axs.axvspan(frames[5], frames[8], alpha=0.1, color = colors_dict[args.eng_speaker2], label = args.eng_speaker2)
        elif df.loc[i].remix == 'spa_eng':
            axs.axvspan(frames[1], frames[5], alpha=0.1, color = colors_dict[args.spa_speaker], label = args.spa_speaker)
            axs.axvspan(frames[5], frames[8], alpha=0.1, color = colors_dict[args.eng_speaker2], label = args.eng_speaker2)
        elif df.loc[i].remix == 'eng_spa':
            axs.axvspan(frames[1], frames[5], alpha=0.1, color = colors_dict[args.eng_speaker], label = args.eng_speaker)
            axs.axvspan(frames[5], frames[8], alpha=0.1, color = colors_dict[args.spa_speaker2], label = args.spa_speaker2)
        elif df.loc[i].remix == 'spa_spa':
            axs.axvspan(frames[1], frames[5], alpha=0.1, color = colors_dict[args.spa_speaker], label = args.spa_speaker)
            axs.axvspan(frames[5], frames[8], alpha=0.1, color = colors_dict[args.spa_speaker2], label = args.spa_speaker2)
    elif df.loc[i].test == 'trains':
        axs.axvline(frames[5], color='k')
        axs.axvline(frames[8], color='k')
        axs.axvline(frames[9], color='k')
        axs.axvspan(frames[8], frames[9], alpha=0.15, color = colors_dict[args.test_speaker], label = args.test_speaker)
        if df.loc[i].remix == 'eng_eng':
            axs.axvspan(frames[0], frames[5], alpha=0.1, color = colors_dict[args.eng_speaker], label = args.eng_speaker)
            axs.axvspan(frames[5], frames[8], alpha=0.1, color = colors_dict[args.eng_speaker2], label = args.eng_speaker2)
        elif df.loc[i].remix == 'spa_eng':
            axs.axvspan(frames[0], frames[5], alpha=0.1, color = colors_dict[args.spa_speaker], label = args.spa_speaker)
            axs.axvspan(frames[5], frames[8], alpha=0.1, color = colors_dict[args.eng_speaker2], label = args.eng_speaker2)
        elif df.loc[i].remix == 'eng_spa':
            axs.axvspan(frames[0], frames[5], alpha=0.1, color = colors_dict[args.eng_speaker], label = args.eng_speaker)
            axs.axvspan(frames[5], frames[8], alpha=0.1, color = colors_dict[args.spa_speaker2], label = args.spa_speaker2)
        elif df.loc[i].remix == 'spa_spa':
            axs.axvspan(frames[0], frames[5], alpha=0.1, color = colors_dict[args.spa_speaker], label = args.spa_speaker)
            axs.axvspan(frames[5], frames[8], alpha=0.1, color = colors_dict[args.spa_speaker2], label = args.spa_speaker2)
    elif (df.loc[i].test == 'bob') or (df.loc[i].test == 'need'):
        axs.axvline(frames[5], color='k')
        axs.axvline(frames[6], color='k')
        axs.axvline(frames[9], color='k')
        axs.axvspan(frames[5], frames[6], alpha=0.15, color = colors_dict[args.test_speaker], label = args.test_speaker)
        if df.loc[i].remix == 'eng_eng':
            axs.axvspan(frames[0], frames[5], alpha=0.1, color = colors_dict[args.eng_speaker], label = args.eng_speaker)
            axs.axvspan(frames[6], frames[9], alpha=0.1, color = colors_dict[args.eng_speaker2], label = args.eng_speaker2)
        elif df.loc[i].remix == 'spa_eng':
            axs.axvspan(frames[0], frames[5], alpha=0.1, color = colors_dict[args.spa_speaker], label = args.spa_speaker)
            axs.axvspan(frames[6], frames[9], alpha=0.1, color = colors_dict[args.eng_speaker2], label = args.eng_speaker2)
        elif df.loc[i].remix == 'eng_spa':
            axs.axvspan(frames[0], frames[5], alpha=0.1, color = colors_dict[args.eng_speaker], label = args.eng_speaker)
            axs.axvspan(frames[6], frames[9], alpha=0.1, color = colors_dict[args.spa_speaker2], label = args.spa_speaker2)
        elif df.loc[i].remix == 'spa_spa':
            axs.axvspan(frames[0], frames[5], alpha=0.1, color = colors_dict[args.spa_speaker], label = args.spa_speaker)
            axs.axvspan(frames[6], frames[9], alpha=0.1, color = colors_dict[args.spa_speaker2], label = args.spa_speaker2)

    return axs