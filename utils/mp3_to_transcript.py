"""
Functions to add the arrays corresponding to .mp3 files and transcriptions to a dataframe containing the filenames.
"""
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

import textgrids

import pydub
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Config
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def add_audio_data(path, noise, save_file = True, seed = 0, sample_rate:int = 16000) -> pd.DataFrame:
    """
    Adds the arrays corresponding to the audio files to speakers.csv
    Saves the resulting pd.DataFrame at the same path as "df_audio.json"

    Input:
        path (str) : path to a folder which contains a "speakers.csv" file (pd.DataFrame of speaker information) and .wav audio files
        sample_rate (int) : Sample rate (default 16000 to which wav2vec2 is trained on)

    Output:
        df (pd.DataFrame) : resulting DataFrame, with the added arrays
    """
    df = pd.read_csv(path + 'speakers.csv')

    for idx in tqdm(range(len(df)), desc = 'Converting audio to arrays...'):
        audio_data, _ = librosa.load(path + df.loc[idx, 'file_name'] + '.wav', sr = sample_rate)
        df.at[idx, 'audio_data'] = audio_data

    if save_file:
        df.to_json(path + 'df_audio.json')

    return df

def convert_to_audio_data(audio:pydub.audio_segment.AudioSegment, sample_rate:int = 16000) -> np.array:
    """
    Convert a pydub.audio_segment.AudioSegment item to an audio_data array

    Input:
        audio (pydub.audio_segment.AudioSegment) : audio file to convert
        sample_rate (int) : sample rate (default 16000 to which wav2vec2 is trained on)

    Output:
        audio_data (np.array) : np.array containing the audio data
    """
    _ = audio.export("output.wav", format="wav")
    audio_data, _ = librosa.load("output.wav", sr = sample_rate)

    return audio_data

def transcript_tokens_to_string(token_list):

    token_list = [x if x != '' else ' ' for x in token_list]

    join_list = [' ' ]
    for token in token_list:
        if join_list[-1] != token:
            join_list.append(token)

    join_list = [i for i in join_list if i != '<pad>']

    transcript = ''.join(join_list)
    return transcript.strip()

def find_timepoints_textgrid(grid, args):
    test_segment = args.text.lower().split(' ')
    min_time = 0
    max_time = 0
    reading = False
    for word in grid['words']:
        if word.text == test_segment[0]:
            min_time = word.xmin
            reading = True
        elif word.text == test_segment[-1] and reading:
            max_time = word.xmax
            break
    return min_time, max_time

def obtain_transcripts_baseline(df:pd.DataFrame, args, sample_rate:int = 16000) -> pd.DataFrame:
    """
    
    """
    all_audio_data = df['audio_data'].apply(np.array)
    df['transcript'] = df['transcript'].apply(str)
    if args.model.startswith('whisper'):
        if args.model == 'whisper':
            processor = WhisperProcessor.from_pretrained("openai/whisper-small")
            model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(args.device)
        elif args.model == 'whisper.en':
            processor = WhisperProcessor.from_pretrained("openai/whisper-small.en")
            model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small.en").to(args.device)
        model.config.forced_decoder_ids = None

        model.eval()
        for i in tqdm(range(len(all_audio_data)), desc = 'Generating transcripts...'):
            # retrieve logits
            input_features = processor(all_audio_data[i], sampling_rate=sample_rate, return_tensors="pt").input_features.to(args.device)
            with torch.no_grad():
                predicted_ids = model.generate(input_features)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            # take argmax and decode
            transcription = transcription.replace('.', '')
            transcription = transcription.replace(',', '')
            transcription = transcription.replace(':', '')
            transcription = transcription.replace(';', '')
            transcription = transcription.replace('  ', ' ')
            transcription = transcription.replace('6', 'six')
            transcription = transcription.replace('5', 'five')
            transcription = transcription.replace('3', 'three')
            df.at[i, 'transcript'] = transcription.strip().upper()

            if args.model == 'whisper':
                aligned_grid = textgrids.TextGrid('./data/unaltered/aligned_whisper/' + df.loc[i].file_name + '.TextGrid')
            elif args.model == 'whisper.en':
                aligned_grid = textgrids.TextGrid('./data/unaltered/aligned_whisper.en/' + df.loc[i].file_name + '.TextGrid')
            
            baseline_grid = textgrids.TextGrid('./data/unaltered/aligned_baseline/' + df.loc[i].file_name + '.TextGrid')

            min_time, max_time = find_timepoints_textgrid(baseline_grid, args)

            transcript = ''
            for word in aligned_grid['words']:
                if word.xmin >= max_time:
                    break
                if word.xmin >= min_time:
                    transcript = transcript + word.text + ' '

            transcript = transcript.replace('  ', ' ')
            transcript = transcript.replace('6', 'six')
            transcript = transcript.replace('3', 'three')

            df.at[i, 'transcript_test'] = transcript.strip().upper()

    else:
        # Import model
        if args.model == 'w2v2_960':
            model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", output_hidden_states=True).to(args.device)
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        elif args.model.startswith("voxpp_"): 
            model = Wav2Vec2ForCTC.from_pretrained('../finetune_models/models/model_'+args.model+'/')
            model.to(args.device)
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        elif args.model == 'w2v2_xlsr':
            model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english", output_attentions=True).to(args.device)
            processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        elif args.model == 'w2v2_large':
            model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h", output_attentions=True).to(args.device)
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

        # Obtain and save transcriptions
        model.eval()
        # for i in tqdm(range(len(all_audio_data)), desc = 'Generating transcripts...'):
        print('Generating transcripts...')
        for i in range(len(all_audio_data)):
            # tokenize
            # Use batch size 1, arrays are of different sizes
            input_values = processor(all_audio_data[i], return_tensors="pt", padding="longest", sampling_rate = sample_rate).input_values.to(args.device)

            # retrieve logits
            with torch.no_grad():
                output = model(input_values)
                logits = output.logits

            # take argmax and decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcriptions = processor.batch_decode(predicted_ids)[0]
            df.at[i, 'transcript'] = transcriptions.upper()

            # Add transcripts for the test segment
            start_point = df.at[i, 'test_timepoints'][0]
            end_point = df.at[i, 'test_timepoints'][1]
            length = df.at[i, 'audio_length']
            start_frame = np.floor((start_point/length)*len(predicted_ids[0])).astype('int')
            end_frame = np.ceil((end_point/length)*len(predicted_ids[0])).astype('int')

            test_ids = predicted_ids[0][start_frame:end_frame]
            test_transcript_list = processor.batch_decode(test_ids)
            test_transcript = transcript_tokens_to_string(test_transcript_list)
            df.at[i, 'transcript_test'] = test_transcript.upper()
        
    df.drop(['audio_data'], axis = 1, inplace=True)

    if args.remix != 'unaltered':
        df.to_json(args.path+ args.model + '/' + args.test + args.remix + '_transcripts.json')
    
    return df

def obtain_transcripts_test(df:pd.DataFrame, args, sample_rate:int = 16000) -> pd.DataFrame:
    """
    
    """
    all_audio_data = df['audio_data'].apply(np.array)
    df['transcript'] = df['transcript'].apply(str)
    if args.model.startswith('whisper'):
        if args.model == 'whisper':
            processor = WhisperProcessor.from_pretrained("openai/whisper-small")
            model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(args.device)
        elif args.model == 'whisper.en':
            processor = WhisperProcessor.from_pretrained("openai/whisper-small.en")
            model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small.en").to(args.device)
        model.config.forced_decoder_ids = None

        model.eval()
        for i in tqdm(range(len(all_audio_data)), desc = 'Generating transcripts...'):
            # retrieve logits
            input_features = processor(all_audio_data[i], sampling_rate=sample_rate, return_tensors="pt").input_features.to(args.device)
            with torch.no_grad():
                predicted_ids = model.generate(input_features)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            # take argmax and decode
            df.at[i, 'transcript'] = transcription.strip().upper()

            # Add transcripts for the test segment
            # For now we will keep it at three words
            transcript_list  = transcription.split(' ')
            if args.test == 'trains_':
                test_transcript = transcript_list[-3] + ' ' + transcript_list[-2] + ' ' + transcript_list[-1]
            elif args.test == 'stella_':
                test_transcript = transcript_list[0] + ' ' + transcript_list[1] + ' ' + transcript_list[2]
            test_transcript = test_transcript.replace('.', '')
            test_transcript = test_transcript.replace(',', '')
            test_transcript = test_transcript.replace(':', '')
            test_transcript = test_transcript.replace(';', '')
            df.at[i, 'transcript_test'] = test_transcript.upper()
            print(test_transcript)
    else:
        # Import model
        if args.model == 'w2v2_960':
            model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", output_hidden_states=True).to(args.device)
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        elif args.model.startswith("voxpp_"): 
            model = Wav2Vec2ForCTC.from_pretrained('../finetune_models/models/model_'+args.model+'/')
            model.to(args.device)
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        elif args.model == 'w2v2_xlsr':
            model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english", output_attentions=True).to(args.device)
            processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        elif args.model == 'w2v2_large':
            model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h", output_attentions=True).to(args.device)
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
            
        # Obtain and save transcriptions
        model.eval()
        # for i in tqdm(range(len(all_audio_data)), desc = 'Generating transcripts...'):
        print('Generating transcripts...')
        for i in range(len(all_audio_data)):
            # tokenize
            # Use batch size 1, arrays are of different sizes
            input_values = processor(all_audio_data[i], return_tensors="pt", padding="longest", sampling_rate = sample_rate).input_values.to(args.device)

            # retrieve logits
            with torch.no_grad():
                output = model(input_values)
                logits = output.logits

            # take argmax and decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcriptions = processor.batch_decode(predicted_ids)[0]
            df.at[i, 'transcript'] = transcriptions.upper()

            # Add transcripts for the test segment
            start_point = df.at[i, 'test_timepoints'][0]
            end_point = df.at[i, 'test_timepoints'][1]
            length = df.at[i, 'audio_length']
            start_frame = np.floor((start_point/length)*len(predicted_ids[0])).astype('int')
            end_frame = np.ceil((end_point/length)*len(predicted_ids[0])).astype('int')

            test_ids = predicted_ids[0][start_frame:end_frame]
            test_transcript_list = processor.batch_decode(test_ids)
            test_transcript = transcript_tokens_to_string(test_transcript_list)
            # print(test_transcript)
            df.at[i, 'transcript_test'] = test_transcript.upper()
        
    df.drop(['audio_data'], axis = 1, inplace=True)

    if args.remix != 'unaltered':
        df.to_json(args.path+ args.model + '/' + args.test + args.remix + '_transcripts.json')
    
    return df