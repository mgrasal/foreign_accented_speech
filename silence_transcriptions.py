import pandas as pd
import numpy as np
import argparse
import torch
import os

import textgrids

import pydub
from pydub import AudioSegment
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Config
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from utils.mp3_to_transcript import add_audio_data, obtain_transcripts_baseline
from utils.evaluate import evaluate_unaltered

def parse_option():
    parser = argparse.ArgumentParser("Obtain transcripts from .mp3 files")

    parser.add_argument("--path", type=str, default='./data/unaltered/', help="path to the folder containing the audio and speakers")
    parser.add_argument("--test", type=str, default='trains', help="Text spoken in the segment to remix")
    parser.add_argument("--detailed", type=bool, default=False, help="Set to true to add total number of mistranscripted words and male-female breakdown to the output")
    parser.add_argument(
        "--model",
        type=str,
        default="w2v2_960",
        choices=[
            "w2v2_960",
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

    args.remix = 'S'

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args
    
def main():
    args = parse_option()
    sample_rate = 16000
    
    if os.path.exists(args.path+ args.model + '/' + args.test + args.remix + '_transcripts.json'):
        print('Reading transcriptions file')
        df = pd.read_json(args.path+ args.model + '/' + args.test + args.remix + '_transcripts.json')
    else:
        if os.path.exists(args.path + 'df_S_audio.json'):
            df = pd.read_json(args.path + 'df_S_audio.json')
        else:
            print('No audio data found')
            df = pd.read_csv(args.path + 'speakers.csv')
            for idx in range(len(df)):
                audio_data, _ = librosa.load(args.path + df.loc[idx, 'file_name'] + '.wav')
                print(df.loc[idx, 'file_name'])
                zeros_to_add = np.zeros(shape=(int(np.ceil(len(audio_data)/10)),))
                print(len(audio_data), len(audio_data) + len(zeros_to_add))
                df.at[idx, 'audio_data'] = np.append(zeros_to_add, audio_data)
            df.to_json(args.path + 'df_S_audio.json')

        # Create transcripts
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
            # for i in tqdm(range(len(all_audio_data)), desc = 'Generating transcripts...'):
            for i in range(len(all_audio_data)):
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
                df.at[i, 'transcript_test'] = ' '

        else:
            # Import model
            if args.model == 'w2v2_960':
                model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(args.device)
                processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            elif args.model == 'w2v2_xlsr':
                model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english").to(args.device)
                processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
            elif args.model == 'w2v2_large':
                model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to(args.device)
                processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

            # Obtain and save transcriptions
            model.eval()
            # for i in tqdm(range(len(all_audio_data)), desc = 'Generating transcripts...'):
            print('Generating transcripts...')
            for i in range(len(all_audio_data)):
                # tokenize
                # Use batch size 1, arrays are of different sizes
                input_values = processor(all_audio_data[i], return_tensors="pt", padding="longest", sampling_rate = sample_rate).input_values
                input_values = input_values.to(args.device)

                # retrieve logits
                with torch.no_grad():
                    output = model(input_values)
                    logits = output.logits

                # take argmax and decode
                predicted_ids = torch.argmax(logits, dim=-1)
                transcriptions = processor.batch_decode(predicted_ids)[0]
                df.at[i, 'transcript'] = transcriptions.upper()

                df.at[i, 'transcript_test'] = ' '
                df.at[i, 'audio_data'] = [0]
                del input_values
                torch.cuda.empty_cache()
            
        df.drop(['audio_data'], axis = 1, inplace=True)

        if args.remix != 'unaltered':
            df.to_json(args.path+ args.model + '/' + args.test + args.remix + '_transcripts.json')

    prompt = "PLEASE CALL STELLA ASK HER TO BRING THESE THINGS WITH HER FROM THE STORE SIX SPOONS OF FRESH SNOW PEAS FIVE THICK SLABS OF BLUE CHEESE AND MAYBE A SNACK FOR HER BROTHER BOB WE ALSO NEED A SMALL PLASTIC SNAKE AND A BIG TOY FROG FOR THE KIDS SHE CAN SCOOP THESE THINGS INTO THREE RED BAGS AND WE WILL GO MEET HER WEDNESDAY AT THE TRAIN STATION"

    evaluate_unaltered(df, prompt, args)
    return 0

main()