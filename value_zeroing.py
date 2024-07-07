import pandas as pd
import numpy as np
import json
import argparse
import torch
import os
import matplotlib.pyplot as plt

from transformers import WhisperProcessor, Wav2Vec2Processor, AutoFeatureExtractor, AutoConfig, WhisperTokenizer
from modeling.customized_modeling_whisper import WhisperForConditionalGeneration
from modeling.customized_modeling_wav2vec2 import Wav2Vec2ForCTC

from sklearn.metrics.pairwise import cosine_distances

from utils.context_mixing_utils import create_remix_df, set_plot_background, plot_attention

MODEL_PATH = {
    "w2v2_960" : "facebook/wav2vec2-base-960h",
    "whisper" : "openai/whisper-small",
    "whisper.en" : "openai/whisper-small.en",
    "whisper_random" : "openai/whisper-small",
}

DIM_AGGREGATOR = 'max'

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
            "same_en1",
            "esM1_Rsil",
            "esM1_Lsil",
        ],
        help="choose mix of speakers to use",
    )
    parser.add_argument("--add_silence", type=bool, default=False, help="Set to true to add silence at the beginning of the audio and then remove it before plottign")
    parser.add_argument(
        "--model",
        type=str,
        default="w2v2_960",
        choices=[
            "w2v2_960",
            "whisper",
            "whisper.en",
            "w2v2_random",
            "whisper_random"
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

    args.name = args.mix
    if args.add_silence:
        args.name = args.name + 'S'
    args.path = './data/context_mixing/'+args.model+'/'
    args.plot_path = 'plots/v0_'+DIM_AGGREGATOR+'/'

    return args

def plot_v0(i, df, plot_data, args):
    fig, axs = plt.subplots(1, figsize = (12, 6))
    fig.suptitle(df.loc[i].remix + ' testing on ' + df.loc[i].test, size = 15)
    x = range(len(plot_data[0]))

    colors = plt.cm.jet(np.linspace(0,1,len(plot_data)))
    axs = set_plot_background(axs, i, df, 'Attention weights', len(plot_data[0]), args)


    for j, line in enumerate(plot_data):
        axs.plot(x, line, label = 'layer '+ str(j+1), color = colors[j])

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
            if j > args.plot_from:
                continue
            axs[0, 0].plot(x, line[0], label = 'layer '+ str(j+1), color = colors[j])

        x = range(len(dict[remix]['bob'][1][0][0]))
        axs[0, 1] = set_plot_background(axs[0, 1], dict[remix]['bob'][0], df, 'bob', len(x), args)
        for j, line in enumerate(dict[remix]['bob'][1]):
            if j > args.plot_from:
                continue
            axs[0, 1].plot(x, line[0], label = 'layer '+ str(j+1), color = colors[j])

        x = range(len(dict[remix]['need'][1][0][0]))
        axs[1, 1] = set_plot_background(axs[1, 1], dict[remix]['need'][0], df, 'need', len(x), args)
        for j, line in enumerate(dict[remix]['need'][1]):
            if j > args.plot_from:
                continue
            axs[1, 1].plot(x, line[0], label = 'layer '+ str(j+1), color = colors[j])

        x = range(len(dict[remix]['trains'][1][0][0]))
        axs[1, 0] = set_plot_background(axs[1, 0], dict[remix]['trains'][0], df, 'trains', len(x), args)
        for j, line in enumerate(dict[remix]['trains'][1]):
            if j > args.plot_from:
                continue
            axs[1, 0].plot(x, line[0], label = 'layer '+ str(j+1), color = colors[j])

        axs[0, 0].legend()

        plt.tight_layout()

        plt.savefig(args.plot_path+args.model+'/'+args.name+'__'+remix, pad_inches = 0)
        print('Saved : ', args.name+'__'+remix)

def get_encoder_word_boundaries(start, end, total_enc_frame, total_audio_time):
    start = total_enc_frame * start / total_audio_time
    end = total_enc_frame * end / total_audio_time
    start = np.ceil(start).astype('int')
    end = np.ceil(end).astype('int')
    return start, end

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

    # Get the attention weights
    # Import model
    if args.model == 'w2v2_960':
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH[args.model]).to(args.device)
        processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH[args.model])
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH[args.model])
    if args.model == "whisper" or args.model == 'whisper.en':
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH[args.model]).to(args.device)
        processor = WhisperProcessor.from_pretrained(MODEL_PATH[args.model], task='transcribe', language='english')
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language = "en", task = "transcribe")
    elif args.model == 'w2v2_random':
        model = Wav2Vec2ForCTC(AutoConfig.from_pretrained("facebook/wav2vec2-base-960h")).to(args.device)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")  
    elif args.model == 'whisper_random':
        model = WhisperForConditionalGeneration(AutoConfig.from_pretrained("openai/whisper-small")).to(args.device)
        processor = WhisperProcessor.from_pretrained("openai/whisper-small", task='transcribe', language='english')
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language = "en", task = "transcribe")        

    model.eval()
    all_audio_data = df['audio_array'].apply(np.array)

    all_enc = [None]*len(df)
    all_dec = [None]*len(df)
    all_cross = [None]*len(df)

    if args.model.startswith('whisper'):
        aux_name = args.name
    for i in range(len(df)):
        if args.add_silence:
            zeros_to_add = np.zeros(shape=(int(np.ceil(len(all_audio_data[i])/10)),))
        plot_lines = []
        # tokenize
        # Use batch size 1, arrays are of different sizes
        if args.add_silence:
            all_audio_data[i] = np.append(zeros_to_add, all_audio_data[i])
        if args.model.startswith('whisper'):
            output = processor(all_audio_data[i], sampling_rate=16000, return_tensors="pt")
            input_values = output.input_features.to(args.device)
            tokenizer = WhisperTokenizer.from_pretrained(MODEL_PATH[args.model], language="english")
            prompt = "PLEASE CALL STELLA ASK HER TO BRING THESE THINGS WITH HER FROM THE STORE SIX SPOONS OF FRESH SNOW PEAS FIVE THICK SLABS OF BLUE CHEESE AND MAYBE A SNACK FOR HER BROTHER BOB WE ALSO NEED A SMALL PLASTIC SNAKE AND A BIG TOY FROG FOR THE KIDS SHE CAN SCOOP THESE THINGS INTO THREE RED BAGS AND WE WILL GO MEET HER WEDNESDAY AT THE TRAIN STATION"
            decoder_ids = torch.tensor([tokenizer(prompt).input_ids]).to(args.device)
        elif args.model == 'w2v2_960'  or args.model == 'w2v2_random': 
            input_values = processor(all_audio_data[i], return_tensors="pt", padding="longest", sampling_rate = 16000).input_values.to(args.device)

        # retrieve logits
        with torch.no_grad():
            if args.model == 'w2v2_960' or args.model == 'w2v2_random':
                original_outputs = model(input_values, 
                                # attention_mask=all_input_features[ex].attention_mask,
                                output_hidden_states=True,
                                return_dict=True)
            elif args.model.startswith('whisper'):
                original_outputs = model(input_values, 
                            decoder_input_ids=decoder_ids,
                            output_hidden_states=True,
                            return_dict=True)

        original_logits = original_outputs.logits
        if args.model == 'w2v2_960' or args.model == 'w2v2_random':
            original_enc_hidden_states = torch.stack(original_outputs['hidden_states'])
        elif args.model.startswith('whisper'):
            original_enc_hidden_states = torch.stack(original_outputs['encoder_hidden_states'])
            original_dec_hidden_states = torch.stack(original_outputs['decoder_hidden_states'])

        num_enc_layers = model.config.num_hidden_layers
        frames_in_interval = 6000 if (args.model == 'w2v2_960'  or args.model == 'w2v2_random') else 1 #If I understood it correctly this should mean every 200ms (3200) or 400ms (6000)
        encoder_aligned_length = int(np.ceil(len(input_values[0])/frames_in_interval))
        all_enc_value_zeroing = []
        if args.model.startswith('whisper'):
            all_dec_value_zeroing = []
            all_cross_value_zeroing = []

        if df.loc[i].test == 'stella':
            end_ratio = (df.loc[i].timepoints[1]/df.loc[i].timepoints[-1])
            end_test = end_ratio*len(input_values)
            end_frame = np.ceil(end_test*encoder_aligned_length).astype('int')
            start_ratio = 0
            start_frame = 0
        elif df.loc[i].test == 'trains':
            start_ratio = (df.loc[i].timepoints[-2]/df.loc[i].timepoints[-1])
            start_test = start_ratio*len(input_values)
            start_frame = np.ceil(start_test*encoder_aligned_length).astype('int')
            end_ratio = 1
            end_frame = encoder_aligned_length
        elif (df.loc[i].test == 'bob') or (df.loc[i].test == 'need'):
            start_ratio = (df.loc[i].timepoints[5]/df.loc[i].timepoints[-1])
            start_test = start_ratio*len(input_values)
            start_frame = np.ceil(start_test*encoder_aligned_length).astype('int')
            end_ratio = (df.loc[i].timepoints[6]/df.loc[i].timepoints[-1])
            end_test = end_ratio*len(input_values)
            end_frame = np.ceil(end_test*encoder_aligned_length).astype('int')

        print(start_frame, end_frame)
        total_enc_dimensions = original_enc_hidden_states.shape[2]
        if args.add_silence:
            start_frame = int((start_ratio*(encoder_aligned_length - encoder_aligned_length/10)+encoder_aligned_length/10))
            end_frame = int(end_ratio*(encoder_aligned_length - np.ceil(encoder_aligned_length/10))+np.ceil(encoder_aligned_length/10)) 

        start_k_enc = start_frame
        end_k_enc = end_frame
        print(start_k_enc, end_k_enc)
        if args.model.startswith('whisper'):
            decoder_length = decoder_ids.shape[-1]
            start_k_dec = int(np.floor(start_k_enc*decoder_length/encoder_aligned_length))
            end_k_dec = int(np.ceil(end_k_enc*decoder_length/encoder_aligned_length))

        if args.model == 'w2v2_960'  or args.model == 'w2v2_random':
            print('starting new example...')
            if args.add_silence:
                vz_enc_matrix = np.zeros(shape=(num_enc_layers, encoder_aligned_length - np.ceil(encoder_aligned_length/11).astype('int')))
            else:
                vz_enc_matrix = np.zeros(shape=(num_enc_layers, encoder_aligned_length))
            print('Encoder:')
            args.attn = 'encoder'
            print(num_enc_layers, encoder_aligned_length, encoder_aligned_length)
            for l, encoder_layer in enumerate(model.wav2vec2.encoder.layers):
                for t in range(encoder_aligned_length):
                    if args.add_silence and t <= np.ceil(encoder_aligned_length/11).astype('int'):
                        continue
                    start_j, end_j = get_encoder_word_boundaries(t, t+1, total_enc_dimensions, encoder_aligned_length)
                    with torch.no_grad():
                        layer_outputs = encoder_layer(
                                            hidden_states=original_enc_hidden_states[l], 
                                            attention_mask=None,# if is_encoder_decoder else attention_mask,
                                            value_zeroing=True,
                                            value_zeroing_index=(start_j, end_j),
                                            value_zeroing_head="all",
                                            )
                    
                    alternative_hidden_states = layer_outputs[0]
                    # last layer is followed by a layer normalization
                    if l == num_enc_layers - 1: 
                        alternative_hidden_states = model.wav2vec2.encoder.layer_norm(alternative_hidden_states)
                    
                    x = alternative_hidden_states.squeeze(0).detach().cpu().numpy()
                    y = original_enc_hidden_states[l+1].squeeze(0).detach().cpu().numpy()
                    distances = cosine_distances(x, y).diagonal()

                    aux = distances[start_k_enc:end_k_enc+1].max() if DIM_AGGREGATOR == "max" else distances[start_k_enc:end_k_enc+1].mean()
                    if args.add_silence:
                        vz_enc_matrix[l, t - np.ceil(encoder_aligned_length/11).astype('int')] = aux
                    else:
                        vz_enc_matrix[l, t] = aux


            vz_enc_matrix = vz_enc_matrix / np.sum(vz_enc_matrix, axis=-1, keepdims=True)
            # print(len(vz_enc_matrix[0]), len(vz_enc_matrix[1]))
            # plot_v0(i, df, vz_enc_matrix, args)

            all_enc[i] =vz_enc_matrix.tolist()

        elif args.model.startswith('whisper'):
            print('starting new example...')
            if args.add_silence:
                vz_enc_matrix = np.zeros(shape=(num_enc_layers, encoder_aligned_length - np.ceil(encoder_aligned_length/11).astype('int')))
            else:
                vz_enc_matrix = np.zeros(shape=(num_enc_layers, encoder_aligned_length))
            print('Encoder:')
            args.attn = 'encoder'
            print(num_enc_layers, encoder_aligned_length, encoder_aligned_length)
            for l, encoder_layer in enumerate(model.model.encoder.layers):
                for t in range(encoder_aligned_length):
                    if args.add_silence and t <= np.ceil(encoder_aligned_length/11).astype('int'):
                        continue
                    start_j, end_j = get_encoder_word_boundaries(t, t+1, total_enc_dimensions, encoder_aligned_length)
                    with torch.no_grad():
                        layer_outputs = encoder_layer(
                                            hidden_states=original_enc_hidden_states[l], 
                                            attention_mask=None,# if is_encoder_decoder else attention_mask,
                                            value_zeroing=True,
                                            value_zeroing_index=(start_j, end_j),
                                            value_zeroing_head="all",
                                        )
                    
                    alternative_hidden_states = layer_outputs[0]
                    # last layer is followed by a layer normalization
                    if l == num_enc_layers - 1: 
                        alternative_hidden_states = model.model.encoder.layer_norm(alternative_hidden_states)
                    
                    x = alternative_hidden_states.squeeze(0).detach().cpu().numpy()
                    y = original_enc_hidden_states[l+1].squeeze(0).detach().cpu().numpy()
                    distances = cosine_distances(x, y).diagonal()

                    aux = distances[start_k_enc:end_k_enc+1].max() if DIM_AGGREGATOR == "max" else distances[start_k_enc:end_k_enc+1].mean()
                    # print('aux', aux)
                    if args.add_silence:
                        vz_enc_matrix[l, t - np.ceil(encoder_aligned_length/11).astype('int')] = aux
                    else:
                        vz_enc_matrix[l, t] = aux

            vz_enc_matrix = vz_enc_matrix / np.sum(vz_enc_matrix, axis=-1, keepdims=True)
            args.name = 'enc_'+aux_name
            # plot_v0(i, df, vz_enc_matrix, args)

            all_enc[i] = vz_enc_matrix.tolist()
            

            print('Decoder:')
            args.attn = 'decoder'
            if args.add_silence:
                vz_dec_matrix = np.zeros(shape=(num_enc_layers, decoder_length - np.ceil(decoder_length/11).astype('int')))
            else:
                vz_dec_matrix = np.zeros(shape=(num_enc_layers, decoder_length))
            for l, decoder_layer in enumerate(model.model.decoder.layers):
                for t in range(decoder_length):
                    if args.add_silence and t <= np.ceil(decoder_length/11).astype('int'):
                        continue
                    # only tokens after t is considerd to see how much they are changed after zeroing t. tokens < t have not seen t yet!
                    if t > end_k_dec:
                        break
                    with torch.no_grad():
                        layer_outputs = decoder_layer(
                                    hidden_states=original_dec_hidden_states[l],
                                    attention_mask=None,
                                    encoder_hidden_states=original_enc_hidden_states[-1],
                                    past_key_value=None,
                                    value_zeroing="decoder",
                                    value_zeroing_index=t,
                                    value_zeroing_head="all",
                                )
        
                        alternative_hidden_states = layer_outputs[0]
                        if l == model.config.decoder_layers - 1: # last layer in whisper is followed by a layer normalization
                            alternative_hidden_states = model.model.decoder.layer_norm(alternative_hidden_states)
                    
                    x = alternative_hidden_states.squeeze(0).detach().cpu().numpy()
                    y = original_dec_hidden_states[l+1].squeeze(0).detach().cpu().numpy()
                    distances = cosine_distances(x, y).diagonal()
                    # print('dist', distances)
                    aux = distances[start_k_dec:end_k_dec+1].max() if DIM_AGGREGATOR == "max" else distances[start_k_dec:end_k_dec+1].mean()
                    # print('aux', aux)
                    if args.add_silence:
                        vz_dec_matrix[l, t - np.ceil(decoder_length/11).astype('int')] = aux
                    else:
                        vz_dec_matrix[l, t] = aux
                    # print(len(vz_dec_matrix))
                    # print(len(vz_dec_matrix[0]), len(vz_dec_matrix[1]))
        
            sums = np.sum(vz_dec_matrix, axis=-1, keepdims=True)
            mask = np.all(sums == 0, axis=-1, keepdims=True)
            vz_dec_matrix = np.divide(vz_dec_matrix, sums, out=np.zeros_like(vz_dec_matrix), where=~mask)

            args.name = 'dec_'+aux_name
            # plot_v0(i, df, vz_dec_matrix, args)

            all_dec[i] = vz_dec_matrix.tolist()

            print('Cross:')
            args.attn = 'cross'
            if args.add_silence:
                vz_cross_matrix = np.zeros(shape=(model.config.decoder_layers, encoder_aligned_length - np.ceil(encoder_aligned_length/11).astype('int')))
            else:
                vz_cross_matrix = np.zeros(shape=(model.config.decoder_layers, encoder_aligned_length))
            for l, decoder_layer in enumerate(model.model.decoder.layers):
                for t in range(encoder_aligned_length):
                    if args.add_silence and t <= np.ceil(encoder_aligned_length/11).astype('int'):
                        continue
                    start_j, end_j = get_encoder_word_boundaries(t, t+1, total_enc_dimensions, encoder_aligned_length)
                    with torch.no_grad():
                        layer_outputs = decoder_layer(
                                    hidden_states=original_dec_hidden_states[l],
                                    encoder_hidden_states=original_enc_hidden_states[-1],
                                    past_key_value=None,
                                    value_zeroing="cross",
                                    value_zeroing_index=(start_j, end_j),
                                    value_zeroing_head="all",
                                )
                    
                    alternative_hidden_states = layer_outputs[0]
                    if l == model.config.decoder_layers - 1: # last layer in whisper is followed by a layer normalization
                        alternative_hidden_states = model.model.decoder.layer_norm(alternative_hidden_states)

                    x = alternative_hidden_states.squeeze(0).detach().cpu().numpy()
                    y = original_dec_hidden_states[l+1].squeeze(0).detach().cpu().numpy()
                    distances = cosine_distances(x, y).diagonal()

                    aux = distances[start_k_enc:end_k_enc+1].max() if DIM_AGGREGATOR == "max" else distances[start_k_enc:end_k_enc+1].mean()
                    if args.add_silence:
                        vz_cross_matrix[l, t - np.ceil(encoder_aligned_length/11).astype('int')] = aux
                    else:
                        vz_cross_matrix[l, t] = aux

            sums = np.sum(vz_cross_matrix, axis=-1, keepdims=True)
            mask = np.all(sums == 0, axis=-1, keepdims=True)
            vz_cross_matrix = np.divide(vz_cross_matrix, sums, out=np.zeros_like(vz_cross_matrix), where=~mask)

            args.name = 'cro_'+aux_name
            # plot_v0(i, df, vz_cross_matrix, args)

            all_cross[i] = vz_cross_matrix.tolist()

        # Generate a plot for this case with one line per layer
        if args.save_all:
            plot_attention(i, df, plot_lines, args)


    if args.add_silence:
        timepoints_df = df.filter(['test','remix','timepoints'], axis=1)    
        timepoints_df.to_json("./data/context_mixing_peak/" + args.mix +".json")
        with open("./data/context_mixing_peak/v0/"+args.model + '_' + 'encoder'+"_" + args.mix+".json", "w") as outfile:
            json.dump(all_enc, outfile)

        if args.model.startswith('whisper'):
            with open("./data/context_mixing_peak/v0/"+args.model + '_' + 'decoder'+"_" + args.mix+".json", "w") as outfile:
                json.dump(all_dec, outfile)
            with open("./data/context_mixing_peak/v0/"+args.model + '_' + 'cross'+"_" + args.mix+".json", "w") as outfile:
                json.dump(all_cross, outfile)
        print('!!!!!!!!!!!!! json saved !!!!!!!!!!!!!!!!!!!!!!!!')
    else:
        timepoints_df = df.filter(['test','remix','timepoints'], axis=1)    
        timepoints_df.to_json("./data/context_mixing/" + args.mix +".json")
        with open("./data/context_mixing/v0/"+args.model + '_' + 'encoder'+"_" + args.mix+".json", "w") as outfile:
            json.dump(all_enc, outfile)

        if args.model.startswith('whisper'):
            with open("./data/context_mixing/v0/"+args.model + '_' + 'decoder'+"_" + args.mix+".json", "w") as outfile:
                json.dump(all_dec, outfile)
            with open("./data/context_mixing/v0/"+args.model + '_' + 'cross'+"_" + args.mix+".json", "w") as outfile:
                json.dump(all_cross, outfile)
        print('!!!!!!!!!!!!! json saved !!!!!!!!!!!!!!!!!!!!!!!!')

    return 0

main()