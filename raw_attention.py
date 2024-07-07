import pandas as pd
import numpy as np
import json
import argparse
import torch
import os
import matplotlib.pyplot as plt

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, AutoFeatureExtractor, AutoConfig
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperTokenizer

from utils.context_mixing_utils import create_remix_df, set_plot_background

def parse_option():
    parser = argparse.ArgumentParser(" ")

    parser.add_argument("--unaltered_path", type=str, default='./data/unaltered/', help="path to the folder containing the audio and speakers")
    parser.add_argument("--plot_from", type=int, default=8, help="Only plot from this layer onwards")
    parser.add_argument("--save_all", type=bool, default=False, help="Set to True to save a separate plot for each test segment")
    parser.add_argument(
        "--mix",
        type=str,
        default="esM1",
        choices=[
            "esM1",
            "esM2",
            "enM1",
            "enM2",
            "esF1",
            "esF2",
            "esM1_Rsil",
            "esM1_Lsil",
        ],
        help="choose mix of speakers to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="w2v2_960",
        choices=[
            "w2v2_960",
            "voxpp_es",
            "voxpp_sk",
            "voxpp_it",
            "w2v2_xlsr",
            "w2v2_large",
            "whisper",
            "whisper.en",
            "w2v2_random"
        ],
        help="choose model to use for the transcriptions",
    )
    parser.add_argument(
        "--attn",
        type=str,
        default="encoder",
        choices=[
            "encoder",
            "cross",
            "decoder",
        ],
        help="choose model to use for the transcriptions",
    )
    
    args = parser.parse_args()

    with open('./data/context_mixing/mix_index.txt') as f:
        data = f.read()
    mix_index = json.loads(data)
    args.eng_speaker = mix_index[args.mix]['eng_speaker']
    args.eng_speaker2 = mix_index[args.mix]['eng_speaker2']
    args.spa_speaker = mix_index[args.mix]['spa_speaker']
    args.spa_speaker2 = mix_index[args.mix]['spa_speaker2']
    args.test_speaker = mix_index[args.mix]['test_speaker']

    args.speakers = [args.eng_speaker, args.eng_speaker2, args.spa_speaker, args.spa_speaker2, args.test_speaker]

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # assert torch.cuda.is_available()

    if args.model.startswith('whisper'):
        args.name = args.attn + '_' + args.mix
    else:
        args.name = args.mix

    args.path = './data/context_mixing/'+args.model+'/'
    args.plot_path = 'plots/context_mixing_layers_low/'

    return args

def plot_attention(i, df, plot_data, transcription, args):
    fig, axs = plt.subplots(1, figsize = (12, 6))
    fig.suptitle(df.loc[i].remix + ' testing on ' + df.loc[i].test, size = 15)
    x = range(len(plot_data[0][0]))

    colors = plt.cm.jet(np.linspace(0,1,len(plot_data)))
    axs = set_plot_background(axs, i, df, 'Attention weights', len(plot_data[0][0]), args)


    for j, line in enumerate(plot_data):
        if j < args.plot_from:
            continue
        axs.plot(x, line[0], label = 'layer '+ str(j+1), color = colors[j])

    axs.legend()

    plt.tight_layout()

    plt.savefig(args.plot_path+args.model+'/'+args.name+'__'+df.loc[i].remix+'_'+df.loc[i].test, pad_inches = 0)
    print('Saved : ', args.name+'__'+df.loc[i].remix+'_'+df.loc[i].test)
    plt.close()

def plot_collages(dict, df, args):
    for remix in dict.keys():
        fig, axs = plt.subplots(2, 2, figsize = (12, 8))
        fig.suptitle(args.mix + ' with remix ' + remix, size = 15)

        colors = plt.cm.jet(np.linspace(0,1,len(dict[remix]['trains'][1])))

        x = range(len(dict[remix]['stella'][1][0][0]))
        axs[0, 0] = set_plot_background(axs[0, 0], dict[remix]['stella'][0], df, 'stella', len(x), args)
        for j, line in enumerate(dict[remix]['stella'][1]):
            if j < args.plot_from:
                continue
            axs[0, 0].plot(x, line[0], label = 'layer '+ str(j+1), color = colors[j])

        x = range(len(dict[remix]['bob'][1][0][0]))
        axs[0, 1] = set_plot_background(axs[0, 1], dict[remix]['bob'][0], df, 'bob', len(x), args)
        for j, line in enumerate(dict[remix]['bob'][1]):
            if j < args.plot_from:
                continue
            axs[0, 1].plot(x, line[0], label = 'layer '+ str(j+1), color = colors[j])

        x = range(len(dict[remix]['need'][1][0][0]))
        axs[1, 1] = set_plot_background(axs[1, 1], dict[remix]['need'][0], df, 'need', len(x), args)
        for j, line in enumerate(dict[remix]['need'][1]):
            if j < args.plot_from:
                continue
            axs[1, 1].plot(x, line[0], label = 'layer '+ str(j+1), color = colors[j])

        x = range(len(dict[remix]['trains'][1][0][0]))
        axs[1, 0] = set_plot_background(axs[1, 0], dict[remix]['trains'][0], df, 'trains', len(x), args)
        for j, line in enumerate(dict[remix]['trains'][1]):
            if j < args.plot_from:
                continue
            axs[1, 0].plot(x, line[0], label = 'layer '+ str(j+1), color = colors[j])

        axs[0, 0].legend()

        plt.tight_layout()

        plt.savefig(args.plot_path+args.model+'/'+args.name+'__'+remix, pad_inches = 0)
        print('Saved : ', args.name+'__'+remix)

def main():
    args = parse_option()

    #Create remix saving the timepoints for each section + test section
    if os.path.exists(args.path + args.mix + '.json'):
        df = pd.read_json(args.path + args.mix + '.json')
        print('Audio arrays read!')
    else:
        print('Creating audio arrays...')
        df = create_remix_df(args)

    # df = create_remix_df(args)
    prompt = "PLEASE CALL STELLA ASK HER TO BRING THESE THINGS WITH HER FROM THE STORE SIX SPOONS OF FRESH SNOW PEAS FIVE THICK SLABS OF BLUE CHEESE AND MAYBE A SNACK FOR HER BROTHER BOB WE ALSO NEED A SMALL PLASTIC SNAKE AND A BIG TOY FROG FOR THE KIDS SHE CAN SCOOP THESE THINGS INTO THREE RED BAGS AND WE WILL GO MEET HER WEDNESDAY AT THE TRAIN STATION"

    # Get the attention weights
    # Import model
    if args.model == 'w2v2_960':
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", output_attentions=True).to(args.device)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
    elif args.model == 'w2v2_xlsr':
        model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english", output_attentions=True).to(args.device)
        processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
    elif args.model == 'w2v2_large':
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h", output_attentions=True).to(args.device)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    elif args.model == 'whisper':
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="english")
        processor = WhisperProcessor.from_pretrained("openai/whisper-small", output_attentions = True)
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(args.device)
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language = 'en',task="transcribe")
    elif args.model == 'whisper.en':
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small.en", language="english")
        processor = WhisperProcessor.from_pretrained("openai/whisper-small.en", output_attentions = True)
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small.en").to(args.device)
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language = 'en',task="transcribe")
    elif args.model == 'w2v2_random':
        model = Wav2Vec2ForCTC(AutoConfig.from_pretrained("facebook/wav2vec2-base-960h")).to(args.device)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

    model.eval()
    all_audio_data = df['audio_array'].apply(np.array)

    plot_dict = {
        'eng_eng' : {},
        'eng_spa' : {},
        'spa_eng' : {},
        'spa_spa' : {},
    }

    for i in range(len(df)):
        plot_lines = []
        # tokenize
        # Add decoder ids
        if args.model.startswith('whisper'):
            input_features = processor(all_audio_data[i], return_tensors="pt", sampling_rate = 16000).input_features.to(args.device)
            decoder_ids = torch.tensor([tokenizer(prompt).input_ids]).to(args.device)
            with torch.no_grad():
                output = model.forward(input_features, output_attentions=True, decoder_input_ids = decoder_ids)
            if args.attn == 'encoder':
                attn = output.encoder_attentions
            elif args.attn == 'cross':
                attn = output.cross_attentions
            elif args.attn == 'decoder':
                attn = output.decoder_attentions
            predicted_ids = model.generate(input_features)
            
        else:
            input_values = processor(all_audio_data[i], return_tensors="pt", padding="longest", sampling_rate = 16000).input_values.to(args.device)
            # retrieve logits
            with torch.no_grad():
                if args.model.endswith('_random'):
                    output = model(input_values, output_attentions = True)
                else:
                    output = model(input_values)

            logits = output.logits
            attn = output.attentions

            # take argmax and decode
            predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0].upper()

        # Check shape to know what we are working with (should be a tuple of #layers containing for each a tensor 1 x attention heads x seq_len x seq_len)
        for attn_layer in attn:
            # Pool / average them apporpiately
            attn_layer.squeeze()
            # 1. Average over all attention heads in each layer
            attn_layer = torch.mean(attn_layer, dim = 1)

            # 2. Take element-wise average of all the frames corresponding to the test sequence
            if df.loc[i].test == 'stella':
                end_test = (df.loc[i].timepoints[1]/df.loc[i].timepoints[-1])*attn_layer.size(1)
                end_frame = np.ceil(end_test).astype('int')
                attn_layer = attn_layer[:, :end_frame, :]
            elif df.loc[i].test == 'trains':
                start_test = (df.loc[i].timepoints[-2]/df.loc[i].timepoints[-1])*attn_layer.size(1)
                start_frame = np.ceil(start_test).astype('int')
                attn_layer = attn_layer[:, start_frame:, :]
            elif (df.loc[i].test == 'bob') or (df.loc[i].test == 'need'):
                start_test = (df.loc[i].timepoints[5]/df.loc[i].timepoints[-1])*attn_layer.size(1)
                start_frame = np.ceil(start_test).astype('int')
                end_test = (df.loc[i].timepoints[6]/df.loc[i].timepoints[-1])*attn_layer.size(1)
                end_frame = np.ceil(end_test).astype('int')
                attn_layer = attn_layer[:, start_frame:end_frame, :]

            attn_layer = torch.mean(attn_layer, dim = 1).cpu()
                
            # Add this layer to an array
            plot_lines.append(attn_layer.cpu().numpy().tolist())

        plot_dict[df.loc[i].remix][df.loc[i].test] = [i, plot_lines]
        # Generate a plot for this case with one line per layer
        if args.save_all:
            plot_attention(i, df, plot_lines, transcription, args)

    timepoints_df = df.filter(['test','remix','timepoints'], axis=1)    
    timepoints_df.to_json("./data/context_mixing/" + args.mix +".json")

    plot_collages(plot_dict, df, args)
 
    with open("./data/context_mixing/raw_attn/"+args.model + '_' + args.attn+"_layers_" + args.mix+".json", "w") as outfile:
        json.dump(plot_dict, outfile)

    return 0

main()