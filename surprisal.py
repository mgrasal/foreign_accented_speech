# Use a pipeline as a high-level helper
from transformers import pipeline
import numpy as np


def compute_surprisals(pipeline, word_list):
    score_list = []
    input_list = word_list.copy()

    for i, word in enumerate(word_list):
        if word_list[i][-1] == '.':
            aux = word_list[i][:-1]
            input_list[i] = '[MASK].'
        elif word_list[i][-1] == ',':
            aux = word_list[i][:-1]
            input_list[i] = '[MASK],'
        elif word_list[i][-1] == ':':
            aux = word_list[i][:-1]
            input_list[i] = '[MASK]:'
        else:
            aux = word_list[i]
            input_list[i] = '[MASK]'
        in_str = " ".join(input_list)
        
        scores = pipeline(in_str, top_k = 500)
        score_list.append(1e-15) # To prevent any word having probability 0, which would mess up the logs in the surprisal
        for pred in scores:
            if pred['token_str'] == aux:
                score_list[i] = pred['score']

        input_list[i] = word_list[i]

    return score_list

def main():
    pipe = pipeline("fill-mask", model="google-bert/bert-base-uncased")

    prompt = "Please call Stella.  Ask her to bring these things with her from the store:  Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.  We also need a small plastic snake and a big toy frog for the kids.  She can scoop these things into three red bags, and we will go meet her Wednesday at the train station."

    word_list = prompt.lower().split()

    score_list = compute_surprisals(pipe, word_list)

    print('prob   |  word\n--------------')
    for i, word in enumerate(word_list):
        print("{:.4f}".format(round(score_list[i], 4))+' | '+ word)
    print('--------------\n')

    print('Surprisal scores for the test segments:\n')

    # please call stella
    first_word_idx = 0
    prob_seq = score_list[first_word_idx]*score_list[first_word_idx+1]*score_list[first_word_idx+2]
    surprisal = -np.log(prob_seq)
    print(word_list[first_word_idx], word_list[first_word_idx+1], word_list[first_word_idx+2], surprisal)

    # ask her to
    first_word_idx = 3
    prob_seq = score_list[first_word_idx]*score_list[first_word_idx+1]*score_list[first_word_idx+2]
    surprisal = -np.log(prob_seq)
    print(word_list[first_word_idx], word_list[first_word_idx+1], word_list[first_word_idx+2], surprisal)

    # from the store
    first_word_idx = 11
    prob_seq = score_list[first_word_idx]*score_list[first_word_idx+1]*score_list[first_word_idx+2]
    surprisal = -np.log(prob_seq)
    print(word_list[first_word_idx], word_list[first_word_idx+1], word_list[first_word_idx+2], surprisal)

    # six spoons of
    first_word_idx = 14
    prob_seq = score_list[first_word_idx]*score_list[first_word_idx+1]*score_list[first_word_idx+2]
    surprisal = -np.log(prob_seq)
    print(word_list[first_word_idx], word_list[first_word_idx+1], word_list[first_word_idx+2], surprisal)

    # of blue cheese
    first_word_idx = 23
    prob_seq = score_list[first_word_idx]*score_list[first_word_idx+1]*score_list[first_word_idx+2]
    surprisal = -np.log(prob_seq)
    print(word_list[first_word_idx], word_list[first_word_idx+1], word_list[first_word_idx+2], surprisal)

    # her brother bob
    prob_seq = score_list[31]*score_list[32]*score_list[33]
    surprisal = -np.log(prob_seq)
    print(word_list[31], word_list[32], word_list[33], surprisal)

    # We also need a
    prob_seq = score_list[34]*score_list[35]*score_list[36]*score_list[37]
    surprisal = -np.log(prob_seq)
    print(word_list[34], word_list[35], word_list[36], word_list[37], surprisal)

    # small plastic snake
    first_word_idx = 38
    prob_seq = score_list[first_word_idx]*score_list[first_word_idx+1]*score_list[first_word_idx+2]
    surprisal = -np.log(prob_seq)
    print(word_list[first_word_idx], word_list[first_word_idx+1], word_list[first_word_idx+2], surprisal)

    # for the kids
    first_word_idx = 46
    prob_seq = score_list[first_word_idx]*score_list[first_word_idx+1]*score_list[first_word_idx+2]
    surprisal = -np.log(prob_seq)
    print(word_list[first_word_idx], word_list[first_word_idx+1], word_list[first_word_idx+2], surprisal)

    # she can scoop
    first_word_idx = 49
    prob_seq = score_list[first_word_idx]*score_list[first_word_idx+1]*score_list[first_word_idx+2]
    surprisal = -np.log(prob_seq)
    print(word_list[first_word_idx], word_list[first_word_idx+1], word_list[first_word_idx+2], surprisal)

    # three red bags
    first_word_idx = 55
    prob_seq = score_list[first_word_idx]*score_list[first_word_idx+1]*score_list[first_word_idx+2]
    surprisal = -np.log(prob_seq)
    print(word_list[first_word_idx], word_list[first_word_idx+1], word_list[first_word_idx+2], surprisal)

    # the train station
    prob_seq = score_list[-3]*score_list[-2]*score_list[-1]
    surprisal = -np.log(prob_seq)
    print(word_list[-3], word_list[-2], word_list[-1], surprisal)

    print('\n')

    return 0

main() 