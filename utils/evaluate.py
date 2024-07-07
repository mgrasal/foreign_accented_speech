"""
Functions to evaluate quality of transcriptions.
"""

import pandas as pd
from jiwer import wer, cer

def evaluate_unaltered(df:pd.DataFrame, prompt:str, args):
    """
    Prints average wer and cer for all audios, as well as male and female audios, for the unaltered files
    
    Input:
        df (pd.DataFrame) : pandas DataFrame containing wer, cer
        prompt (str) : Correct transcription of the text
    """
    ev_df = eval_df_slices(prompt.upper(), df, args)
    ev_df.to_json(args.path+ args.model + '/' + args.test + args.remix + '_eval.json')

    print('\n'+'*** Evaluating ' + args.model + ' on test segment "' + args.text + '" ***')
    # Evaluate english speakers
    eng_eval = ev_df.loc[ev_df['language'] == 'eng']
    print(' -- English --')
    print(eng_eval[['wer', 'cer', 'subs_wer', 'subs_cer']].describe())
    if args.detailed:
        f_wrong = 0
        m_wrong = 0
        for idx in eng_eval.index:
            if eng_eval.loc[idx, 'transcript'][-1 * len(args.text):] != 'THE TRAIN STATION':
                if eng_eval.loc[idx, 'gender'] == 'F':
                    f_wrong += 1
                else:
                    m_wrong += 1
            # print('\n', wrong)
        print('Total mistranscriptions:', m_wrong+f_wrong)

        print('\n -- Male English --')
        print(eng_eval.loc[eng_eval['gender'] == 'M', ['wer', 'cer', 'subs_wer', 'subs_cer']].describe())
        print('Mistranscriptions (Male):', m_wrong)

        print('\n -- Female English --')
        print(eng_eval.loc[eng_eval['gender'] == 'F', ['wer', 'cer', 'subs_wer', 'subs_cer']].describe())
        print('Mistranscriptions (Female):', f_wrong)

    # Evaluate spanish speakers
    spa_eval = ev_df.loc[ev_df['language'] == 'spa']
    print('\n -- Spanish --')
    print(spa_eval[['wer', 'cer', 'subs_wer', 'subs_cer']].describe())
    if args.detailed:
        f_wrong = 0
        m_wrong = 0
        for idx in spa_eval.index:
            if spa_eval.loc[idx, 'transcript'][-1 * len(args.text):] != 'THE TRAIN STATION':
                if spa_eval.loc[idx, 'gender'] == 'F':
                    f_wrong += 1
                else:
                    m_wrong += 1
        print('Total mistranscriptions:', m_wrong+f_wrong)

        print('\n -- Male Spanish --')
        print(spa_eval.loc[spa_eval['gender'] == 'M', ['wer', 'cer', 'subs_wer', 'subs_cer']].describe())
        print('Mistranscriptions (Male):', m_wrong)

        print('\n -- Female Spanish --')
        print(spa_eval.loc[spa_eval['gender'] == 'F', ['wer', 'cer', 'subs_wer', 'subs_cer']].describe())
        print('Mistranscriptions (Female):', f_wrong)

def evaluate_varying_context(ev_df:pd.DataFrame, partitions:int, args):
    """
    Prints average wer and cer for both the full text and the last three words "the train station"
    Input:
        ev_df (pd.DataFrame) : pandas DataFrame containing wer, cer, subs_wer, and subs_cer for each audio, which is also labeled with "accent1_segments"
        partitions (int) : number of segments the text was divided in (excluding "the train station")
    """
    print('Results for combination (speaker1_speaker2_speaker) = ' + args.remix)
    print('Testing on ' + args.text + ' using model ' + args.model)
    for i in range(partitions):
        print('---- Speaker 1 (' + args.lang_list[0] + ') is used for ' + str(i) + ' out of ' + str(partitions - 1) + ' parts ----')
        part_df = ev_df.loc[ev_df['accent1_segments'] == i]

        print(part_df[['wer', 'cer', 'subs_wer', 'subs_cer']].describe())

        print('\n')

def eval_df_slices(prompt:str, df:pd.DataFrame, args) -> pd.DataFrame:
    """
    Add columns for error rate on both word an character level.
    Analizes error rates both for the whole sentence and the last "length" caracters

    Input:
        prompt (str): Correct transcription
        df (pd.DataFrame): DataFrame containing a column called 'transcript' of transcripts
        length (int): Number of characters (starting from the back) in which to focus

    Output:
        df (pd.DataFrame): Updated DataFrame
    """
    if args.model == 'whisper':
        print('------------------------------------------------------------------------')
        print('| Warning: Transcriptions of test segments not implemented for whisper |')
        print('------------------------------------------------------------------------')
        print('|      Only implemented for test segments "stella" and "trains"        |')
        print('------------------------------------------------------------------------')

    for idx in range(len(df)):
        df.at[idx, 'wer'] = wer(prompt, df.loc[idx, 'transcript'])
        df.at[idx, 'cer'] = cer(prompt, df.loc[idx, 'transcript'])

        if args.model == 'whisper':
            if args.test == 'stella_' or args.test == 'trains_':
                df.at[idx, 'subs_wer'] = wer(args.text, df.loc[idx, 'transcript_test'])
                df.at[idx, 'subs_cer'] = cer(args.text, df.loc[idx, 'transcript_test'])
            else:
                df.at[idx, 'subs_wer'] = None
                df.at[idx, 'subs_cer'] = None
        else:
            df.at[idx, 'subs_wer'] = wer(args.text, df.loc[idx, 'transcript_test'])
            df.at[idx, 'subs_cer'] = cer(args.text, df.loc[idx, 'transcript_test'])

    return df