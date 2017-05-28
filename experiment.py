"""
Copyright (c) 2016 Duolingo Inc. MIT Licence.

Python script that implements spaced repetition models from Settles & Meeder (2016).
Recommended to run with pypy for efficiency. See README.
"""

import argparse
import csv
import gzip
import math
import os
import random
import sys

from collections import defaultdict, namedtuple


# various constraints on parameters and outputs
MIN_HALF_LIFE_DAYS = 15.0 / (24 * 60)    # 15 minutes
MAX_HALF_LIFE_DAYS = 274.                # 9 months
LN2 = math.log(2.)


# data instance object
Instance = namedtuple('Instance', 'p t fv h a lang right wrong ts uid lexeme'.split())


# spaced repetition approaches
HALF_LIFE_REGRESSION = 'hlr'
LOGISTIC_REGRESSION = 'lr'
LEITNER = 'leitner'
PIMSLEUR = 'pimsleur'


class SpacedRepetitionModel(object):
    """
    Spaced repetition model.

    Implements the following approaches:
      - HALF_LIFE_REGRESSION (trainable)
      - LOGISTIC_REGRESSION (trainable)
      - LEITNER (fixed)
      - PIMSLEUR (fixed)
    """
    def __init__(self, method=HALF_LIFE_REGRESSION, omit_h_term=False, initial_weights=None,
                 learning_rate=.001, half_life_weight=.01, regularization_weight=.1, sigma=1.):
        self.method = method
        self.omit_h_term = omit_h_term
        self.weights = defaultdict(float, {} if initial_weights is None else initial_weights)
        self.feature_counts = defaultdict(int)
        self.learning_rate = learning_rate
        self.half_life_weight = half_life_weight
        self.regularization_weight = regularization_weight
        self.sigma = sigma

    def halflife(self, data_instance, base):
        try:
            dp = sum(self.weights[k] * x_k for k, x_k in data_instance.fv)
            return hclip(base ** dp)
        except:
            return MAX_HALF_LIFE_DAYS

    def predict(self, data_instance, base=2.):
        if self.method == HALF_LIFE_REGRESSION:
            h = self.halflife(data_instance, base)
            p = 2. ** (-data_instance.t/h)
            return pclip(p), h
        elif self.method == LEITNER:
            try:
                h = hclip(2. ** data_instance.fv[0][1])
            except OverflowError:
                h = MAX_HALF_LIFE_DAYS
            p = 2. ** (-data_instance.t/h)
            return pclip(p), h
        elif self.method == PIMSLEUR:
            try:
                h = hclip(2. ** (2.35*data_instance.fv[0][1] - 16.46))
            except OverflowError:
                h = MAX_HALF_LIFE_DAYS
            p = 2. ** (-data_instance.t/h)
            return pclip(p), h
        elif self.method == LOGISTIC_REGRESSION:
            dp = sum(self.weights[k] * x_k for k, x_k in data_instance.fv)
            p = 1./(1+math.exp(-dp))
            return pclip(p), random.random()
        else:
            raise Exception

    def train_update(self, data_instance):
        if self.method == HALF_LIFE_REGRESSION:
            base = 2.
            p, h = self.predict(data_instance, base)
            dlp_dw = 2.*(p-data_instance.p)*(LN2**2)*p*(data_instance.t/h)
            dlh_dw = 2.*(h-data_instance.h)*LN2*h
            for (k, x_k) in data_instance.fv:
                rate = (1./(1+data_instance.p)) * self.learning_rate / math.sqrt(1 + self.feature_counts[k])
                # rate = self.learning_rate / math.sqrt(1 + self.feature_counts[k])
                # sl(p) update
                self.weights[k] -= rate * dlp_dw * x_k
                # sl(h) update
                if not self.omit_h_term:
                    self.weights[k] -= rate * self.half_life_weight * dlh_dw * x_k
                # L2 regularization update
                self.weights[k] -= rate * self.regularization_weight * self.weights[k] / self.sigma**2
                # increment feature count for learning rate
                self.feature_counts[k] += 1
        elif self.method == LEITNER or self.method == PIMSLEUR:
            pass
        elif self.method == LOGISTIC_REGRESSION:
            p, _ = self.predict(data_instance)
            err = p - data_instance.p
            for (k, x_k) in data_instance.fv:
                # rate = (1./(1+data_instance.p)) * self.learning_rate   / math.sqrt(1 + self.feature_counts[k])
                rate = self.learning_rate / math.sqrt(1 + self.feature_counts[k])
                # error update
                self.weights[k] -= rate * err * x_k
                # L2 regularization update
                self.weights[k] -= rate * self.regularization_weight * self.weights[k] / self.sigma**2
                # increment feature count for learning rate
                self.feature_counts[k] += 1

    def train(self, trainset):
        if self.method == LEITNER or self.method == PIMSLEUR:
            return
        random.shuffle(trainset)
        for data_instance in trainset:
            self.train_update(data_instance)

    def losses(self, data_instance):
        p, h = self.predict(data_instance)
        slp = (data_instance.p - p)**2
        slh = (data_instance.h - h)**2
        return slp, slh, p, h

    def eval(self, testset, prefix=''):
        results = {'p': [], 'h': [], 'pp': [], 'hh': [], 'slp': [], 'slh': []}
        for data_instance in testset:
            slp, slh, p, h = self.losses(data_instance)
            results['p'].append(data_instance.p)     # ground truth
            results['h'].append(data_instance.h)
            results['pp'].append(p)         # predictions
            results['hh'].append(h)
            results['slp'].append(slp)      # loss function values
            results['slh'].append(slh)
        mae_p = mae(results['p'], results['pp'])
        mae_h = mae(results['h'], results['hh'])
        cor_p = spearmanr(results['p'], results['pp'])
        cor_h = spearmanr(results['h'], results['hh'])
        total_slp = sum(results['slp'])
        total_slh = sum(results['slh'])
        total_l2 = sum(x**2 for x in self.weights.values())
        total_loss = total_slp + self.half_life_weight*total_slh + self.regularization_weight*total_l2
        if prefix:
            sys.stderr.write('%s\t' % prefix)
        sys.stderr.write('%.1f (p=%.1f, h=%.1f, l2=%.1f)\tmae(p)=%.3f\tcor(p)=%.3f\tmae(h)=%.3f\tcor(h)=%.3f\n' % \
            (total_loss, total_slp, self.half_life_weight*total_slh, self.regularization_weight*total_l2, \
            mae_p, cor_p, mae_h, cor_h))

    def dump_weights(self, fname):
        with open(fname, 'wb') as f:
            for (k, v) in self.weights.iteritems():
                f.write('%s\t%.4f\n' % (k, v))

    def dump_predictions(self, fname, testset):
        with open(fname, 'wb') as f:
            f.write('p\tpp\th\thh\tlang\tuser_id\ttimestamp\n')
            for data_instance in testset:
                pp, hh = self.predict(data_instance)
                f.write('%.4f\t%.4f\t%.4f\t%.4f\t%s\t%s\t%d\n' % (data_instance.p, pp,
                        data_instance.h, hh, data_instance.lang, data_instance.uid, data_instance.ts))

    def dump_detailed_predictions(self, fname, testset):
        with open(fname, 'wb') as f:
            f.write('p\tpp\th\thh\tlang\tuser_id\ttimestamp\tlexeme_tag\n')
            for data_instance in testset:
                pp, hh = self.predict(data_instance)
                for i in range(data_instance.right):
                    f.write('1.0\t%.4f\t%.4f\t%.4f\t%s\t%s\t%d\t%s\n' % (pp, data_instance.h, hh,
                            data_instance.lang, data_instance.uid, data_instance.ts, data_instance.lexeme))
                for i in range(data_instance.wrong):
                    f.write('0.0\t%.4f\t%.4f\t%.4f\t%s\t%s\t%d\t%s\n' % (pp, data_instance.h, hh,
                            data_instance.lang, data_instance.uid, data_instance.ts, data_instance.lexeme))


def pclip(p):
    # bound min/max model predictions (helps with loss optimization)
    return min(max(p, 0.0001), .9999)


def hclip(h):
    # bound min/max half-life
    return min(max(h, MIN_HALF_LIFE_DAYS), MAX_HALF_LIFE_DAYS)


def mae(l1, l2):
    # mean average error
    return mean([abs(l1[i] - l2[i]) for i in range(len(l1))])


def mean(lst):
    # the average of a list
    return float(sum(lst))/len(lst)


def spearmanr(l1, l2):
    # spearman rank correlation
    m1 = mean(l1)
    m2 = mean(l2)
    num = 0.
    d1 = 0.
    d2 = 0.
    for i in range(len(l1)):
        num += (l1[i]-m1)*(l2[i]-m2)
        d1 += (l1[i]-m1)**2
        d2 += (l2[i]-m2)**2
    return num/math.sqrt(d1*d2)


def read_data(input_file, method, omit_bias=False, omit_lexemes=False, max_lines=None):
    # read learning trace data in specified format, see README for details
    sys.stderr.write('reading data...')
    instances = list()
    if input_file.endswith('gz'):
        f = gzip.open(input_file, 'rb')
    else:
        f = open(input_file, 'rb')
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if max_lines is not None and i >= max_lines:
            break
        p = pclip(float(row['p_recall']))
        t = float(row['delta'])/(60*60*24)  # convert time delta to days
        h = hclip(-t/(math.log(p, 2)))
        lang = '%s->%s' % (row['ui_language'], row['learning_language'])
        lexeme_id = row['lexeme_id']
        lexeme_string = row['lexeme_string']
        timestamp = int(row['timestamp'])
        user_id = row['user_id']
        seen = int(row['history_seen'])
        right = int(row['history_correct'])
        wrong = seen - right
        right_this = int(row['session_correct'])
        wrong_this = int(row['session_seen']) - right_this
        # feature vector is a list of (feature, value) tuples
        fv = []
        # core features based on method
        if method == LEITNER:
            fv.append((intern('diff'), right-wrong))
        elif method == PIMSLEUR:
            fv.append((intern('total'), right+wrong))
        else:
            # fv.append((intern('right'), right))
            # fv.append((intern('wrong'), wrong))
            fv.append((intern('right'), math.sqrt(1+right)))
            fv.append((intern('wrong'), math.sqrt(1+wrong)))
        # optional flag features
        if method == LOGISTIC_REGRESSION:
            fv.append((intern('time'), t))
        if not omit_bias:
            fv.append((intern('bias'), 1.))
        if not omit_lexemes:
            fv.append((intern('%s:%s' % (row['learning_language'], lexeme_string)), 1.))
        instances.append(Instance(p, t, fv, h, (right+2.)/(seen+4.), lang, right_this, wrong_this, timestamp, user_id, lexeme_string))
        if i % 1000000 == 0:
            sys.stderr.write('%d...' % i)
    sys.stderr.write('done!\n')
    splitpoint = int(0.9 * len(instances))
    return instances[:splitpoint], instances[splitpoint:]


argparser = argparse.ArgumentParser(description='Fit a SpacedRepetitionModel to data.')
argparser.add_argument('-b', action='store_true', default=False, help='omit bias feature')
argparser.add_argument('-l', action='store_true', default=False, help='omit lexeme features')
argparser.add_argument('-t', action='store_true', default=False, help='omit half-life term')
argparser.add_argument('-m', action='store', dest='method', default=HALF_LIFE_REGRESSION,
                       help=' '.join([HALF_LIFE_REGRESSION, LOGISTIC_REGRESSION, LEITNER,
                                      PIMSLEUR]))
argparser.add_argument('-x', action='store', dest='max_lines', type=int, default=None, help='maximum number of lines to read (for dev)')
argparser.add_argument('input_file', action='store', help='log file for training')


if __name__ == '__main__':

    args = argparser.parse_args()

    # model diagnostics
    sys.stderr.write('method = "%s"\n' % args.method)
    if args.b:
        sys.stderr.write('--> omit_bias\n')
    if args.l:
        sys.stderr.write('--> omit_lexemes\n')
    if args.t:
        sys.stderr.write('--> omit_h_term\n')

    # read data set
    trainset, testset = read_data(args.input_file, args.method, args.b, args.l, args.max_lines)
    sys.stderr.write('|train| = %d\n' % len(trainset))
    sys.stderr.write('|test|  = %d\n' % len(testset))

    # train model & print preliminary evaluation info
    model = SpacedRepetitionModel(method=args.method, omit_h_term=args.t)
    model.train(trainset)
    model.eval(testset, 'test')

    # write out model weights and predictions
    filebits = [args.method] + \
        [k for k, v in sorted(vars(args).iteritems()) if v is True] + \
        [os.path.splitext(os.path.basename(args.input_file).replace('.gz', ''))[0]]
    if args.max_lines is not None:
        filebits.append(str(args.max_lines))
    filebase = '.'.join(filebits)
    if not os.path.exists('results/'):
        os.makedirs('results/')
    model.dump_weights('results/'+filebase+'.weights')
    model.dump_predictions('results/'+filebase+'.preds', testset)
    # model.dump_detailed_predictions('results/'+filebase+'.detailed', testset)
