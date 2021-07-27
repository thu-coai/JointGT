# Evaluation code for data-to-text generation
# This code is modified based on
# https://github.com/wenhuchen/Data-to-text-Evaluation-Metric/blob/master/measure_scores.py


import codecs
from argparse import ArgumentParser
from tempfile import mkdtemp
import os
import shutil
import subprocess
import re
import sys

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def read_lines(file_name, multi_ref=False):
    """Read one instance per line from a text file. In multi-ref mode, assumes multiple lines
    (references) per instance & instances separated by empty lines."""
    buf = [[]] if multi_ref else []
    with codecs.open(file_name, 'rb', 'UTF-8') as fh:
        for line in fh:
            line = line.strip()
            if multi_ref:
                if not line:
                    buf.append([])
                else:
                    buf[-1].append(line)
            else:
                buf.append(line)
    if multi_ref and not buf[-1]:
        del buf[-1]
    return buf


def read_tsv(tsv_file):
    """Read a TSV file, check basic integrity."""
    tsv_data = read_lines(tsv_file)
    tsv_data[0] = re.sub(u'\ufeff', '', tsv_data[0])  # remove unicode BOM
    tsv_data = [line.replace(u'Ł', u'£') for line in tsv_data]  # fix Ł
    tsv_data = [line.replace(u'Â£', u'£') for line in tsv_data]  # fix Â£
    tsv_data = [line.replace(u'Ã©', u'é') for line in tsv_data]  # fix Ã©
    tsv_data = [line.replace(u'ã©', u'é') for line in tsv_data]  # fix ã©
    tsv_data = [line.split("\t") for line in tsv_data if line]  # split, ignore empty lines
    if len([line for line in tsv_data if len(line) == 1]) == len(tsv_data):  # split CSV
        tsv_data = [line[0].split('","') for line in tsv_data]

    if re.match(r'^"?mr', tsv_data[0][0], re.I):  # ignore header
        tsv_data = tsv_data[1:]

    errs = [line_no for line_no, item in enumerate(tsv_data, start=1) if len(item) != 2]
    if errs:
        print("%s -- weird number of values" % tsv_file)
        raise ValueError('%s -- Weird number of values on lines: %s' % (tsv_file, str(errs)))

    # remove quotes
    srcs = []
    refs = []
    for src, ref in tsv_data:
        src = re.sub(r'^\s*[\'"]?\s*', r'', src)
        src = re.sub(r'\s*[\'"]?\s*$', r'', src)
        ref = re.sub(r'^\s*[\'"]?\s*', r'', ref)
        ref = re.sub(r'\s*[\'"]?\s*$', r'', ref)
        srcs.append(src)
        refs.append(ref)
    # check quotes
    errs = [line_no for line_no, sys in enumerate(refs, start=1) if '"' in sys]
    if errs:
        print("%s -- has quotes" % tsv_file)
        raise ValueError('%s -- Quotes on lines: %s' % (tsv_file, str(errs)))

    return srcs, refs


def read_and_check_tsv(sys_file, src_file):
    """Read system outputs from a TSV file, check that MRs correspond to a source file."""
    # read
    src_data = read_lines(src_file)
    sys_srcs, sys_outs = read_tsv(sys_file)
    # check integrity
    if len(sys_outs) != len(src_data):
        print("%s -- wrong data length" % sys_file)
        raise ValueError('%s -- SYS data of different length than SRC: %d' % (sys_file, len(sys_outs)))
    # check sameness
    errs = [line_no for line_no, (sys, ref) in enumerate(zip(sys_srcs, src_data), start=1)
            if sys != ref]
    if errs:
        print("%s -- SRC fields not the same as reference" % sys_file)
        raise ValueError('%s -- The SRC fields in SYS data are not the same as reference SRC on lines: %s' % (sys_file, str(errs)))

    # return the checked data
    return src_data, sys_outs


def read_and_group_tsv(ref_file):
    """Read a TSV file with references (and MRs), group the references according to identical MRs
    on consecutive lines."""
    ref_srcs, ref_sents = read_tsv(ref_file)
    refs = []
    cur_src = None
    for src, ref in zip(ref_srcs, ref_sents):
        if src != cur_src:
            refs.append([ref])
            cur_src = src
        else:
            refs[-1].append(ref)
    return refs


def write_tsv(fname, header, data):
    data.insert(0, header)
    with codecs.open(fname, 'wb', 'UTF-8') as fh:
        for item in data:
            fh.write("\t".join(item) + "\n")


def create_coco_refs(data_ref):
    """Create MS-COCO human references JSON."""
    out = {'info': {}, 'licenses': [], 'images': [], 'type': 'captions', 'annotations': []}
    ref_id = 0
    for inst_id, refs in enumerate(data_ref):
        out['images'].append({'id': 'inst-%d' % inst_id})
        for ref in refs:
            out['annotations'].append({'image_id': 'inst-%d' % inst_id,
                                       'id': ref_id,
                                       'caption': ref})
            ref_id += 1
    return out


def create_coco_sys(data_sys):
    """Create MS-COCO system outputs JSON."""
    out = []
    for inst_id, inst in enumerate(data_sys):
        out.append({'image_id': 'inst-%d' % inst_id, 'caption': inst})
    return out


def load_data(ref_file, sys_file, src_file=None):
    """Load the data from the given files."""
    # read SRC/SYS files
    if src_file:
        data_src, data_sys = read_and_check_tsv(sys_file, src_file)
    elif re.search('\.[ct]sv$', sys_file, re.I):
        data_src, data_sys = read_tsv(sys_file)
    else:
        data_sys = read_lines(sys_file)
        # dummy source files (sources have no effect on measures, but MTEval wants them)
        data_src = [''] * len(data_sys)

    # read REF file
    if re.search('\.[ct]sv$', ref_file, re.I):
        data_ref = read_and_group_tsv(ref_file)
    else:
        data_ref = read_lines(ref_file, multi_ref=True)
        if len(data_ref) == 1:  # this was apparently a single-ref file -> fix the structure
            data_ref = [[inst] for inst in data_ref[0]]

    # sanity check
    assert len(data_ref) == len(data_sys) == len(data_src), "{} != {} != {}".format(len(data_ref), len(data_sys), len(data_src))
    return data_src, data_ref, data_sys


def evaluate(data_src, data_ref, data_sys,
             print_as_table=False, print_table_header=False, sys_fname='',
             python=False):
    """Main procedure, running the MS-COCO evaluators on the loaded data."""

    # run the MS-COCO evaluator
    coco_eval = run_coco_eval(data_ref, data_sys)
    scores = {metric: score for metric, score in list(coco_eval.eval.items())}

    metric_names = ['Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr']
    if print_as_table:
        if print_table_header:
            print('\t'.join(['File'] + metric_names))
        print('\t'.join([sys_fname] + ['%.4f' % scores[metric] for metric in metric_names]))
    else:
        print('SCORES:\n==============')
        for metric in metric_names:
            print('%s: %.4f' % (metric, scores[metric]))
        print()


def run_coco_eval(data_ref, data_sys):
    """Run the COCO evaluator, return the resulting evaluation object (contains both
    system- and segment-level scores."""
    # convert references and system outputs to MS-COCO format in-memory
    coco_ref = create_coco_refs(data_ref)
    coco_sys = create_coco_sys(data_sys)

    print('Running MS-COCO evaluator...', file=sys.stderr)
    coco = COCO()
    coco.dataset = coco_ref
    coco.createIndex()

    coco_res = coco.loadRes(resData=coco_sys)
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.evaluate()

    return coco_eval


if __name__ == '__main__':
    ap = ArgumentParser(description='E2E Challenge evaluation -- MS-COCO & MTEval wrapper')
    ap.add_argument('-l', '--sent-level', '--seg-level', '--sentence-level', '--segment-level',
                    type=str, help='Output segment-level scores in a TSV format to the given file?',
                    default=None)
    ap.add_argument('-s', '--src-file', type=str, help='Source file -- if given, system output ' +
                    'should be a TSV with source & output columns, source is checked for integrity',
                    default=None)
    ap.add_argument('-p', '--python', action='store_true',
                    help='Use Python implementation of MTEval instead of Perl?')
    ap.add_argument('-t', '--table', action='store_true', help='Print out results as a line in a'
                    'TSV table?')
    ap.add_argument('-H', '--header', action='store_true', help='Print TSV table header?')
    ap.add_argument('ref_file', type=str, help='References file -- multiple references separated ' +
                    'by empty lines (or single-reference with no empty lines). Can also be a TSV ' +
                    'file with source & reference columns. In that case, consecutive identical ' +
                    'SRC columns are grouped as multiple references for the same source.')
    ap.add_argument('sys_file', type=str, help='System output file to evaluate (text file with ' +
                    'one output per line, or a TSV file with sources & corresponding outputs).')
    args = ap.parse_args()

    data_src, data_ref, data_sys = load_data(args.ref_file, args.sys_file, args.src_file)

    evaluate(data_src, data_ref, data_sys, args.table, args.header, args.sys_file, args.python)
