# Evaluation codes for KBQG tasks
# This code is modified based on
# https://github.com/hugochan/Graph2Seq-for-KGQG/blob/master/src/core/evaluation/eval.py


from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from collections import defaultdict


class WMD(object):
    """docstring for WMD"""
    def __init__(self, word_emb_file, norm_vec=True):
        super(WMD, self).__init__()
        from gensim.models import KeyedVectors
        self.model = KeyedVectors.load_word2vec_format(word_emb_file, binary=False)
        if norm_vec:
            self.model.init_sims(replace=True)

    def distance(self, a, b):
        distance = self.model.wmdistance(a.split(), b.split())
        return distance


class QGEvalCap:
    def __init__(self, gts, res):
        self.gts = gts
        self.res = res

    def evaluate(self, verbose=False):
        output = {}
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            # (Cider(), "CIDEr")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            # print 'computing %s score...'%(scorer.method())
            score, scores = scorer.compute_score(self.gts, self.res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    if verbose:
                        print("%s: %0.5f"%(m, sc))
                    # output.append(sc)
                    output[m] = sc
            else:
                if verbose:
                    print("%s: %0.5f"%(method, score))
                # output.append(score)
                output[method] = score
        return output


def eval(out_file, src_file, tgt_file, isDIn = False, num_pairs = 500):
    """
        Given a filename, calculate the metric scores for that prediction file

        isDin: boolean value to check whether input file is DirectIn.txt
    """

    pairs = []
    with open(src_file, 'r') as infile:
        for line in infile:
            pair = {}
            pair['tokenized_sentence'] = line[:-1]
            pairs.append(pair)

    with open(tgt_file, "r") as infile:
        cnt = 0
        for line in infile:
            pairs[cnt]['tokenized_question'] = line[:-1]
            cnt += 1

    output = []
    with open(out_file, 'r') as infile:
        for line in infile:
            line = line[:-1]
            output.append(line)


    for idx, pair in enumerate(pairs):
        pair['prediction'] = output[idx]


    ## eval
    from eval import QGEvalCap
    import json
    from json import encoder
    encoder.FLOAT_REPR = lambda o: format(o, '.4f')

    res = defaultdict(lambda: [])
    gts = defaultdict(lambda: [])
    for pair in pairs[:]:
        key = pair['tokenized_sentence']
        #res[key] = [pair['prediction'].encode('utf-8')]
        res[key] = [pair['prediction']]

        ## gts
        #gts[key].append(pair['tokenized_question'].encode('utf-8'))
        gts[key].append(pair['tokenized_question'])

    QGEval = QGEvalCap(gts, res)
    return QGEval.evaluate()


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-out", "--out_file", dest="out_file", default="./output/pred.txt", help="output file to compare")
    parser.add_argument("-src", "--src_file", dest="src_file", default="../data/processed/src-test.txt", help="src file")
    parser.add_argument("-tgt", "--tgt_file", dest="tgt_file", default="../data/processed/tgt-test.txt", help="target file")
    args = parser.parse_args()

    print("scores: \n")
    print(eval(args.out_file, args.src_file, args.tgt_file))
