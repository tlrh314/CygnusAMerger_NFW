# -*- coding: utf-8 -*-

import sys

def p2(a):
    return ((a)*(a))

def p3(a):
    return ((a)*(a)*(a))

def print_progressbar(i, N, whitespace=""):
    pbwidth = 42

    progress = float(i)/(N-1)
    block = int(round(pbwidth*progress))
    text = "\r{0}Progress: [{1}] {2:.1f}%".format(whitespace,
        "#"*block + "-"*(pbwidth-block), progress*100)
    sys.stdout.write(text)
    sys.stdout.flush()

    if i == (N-1):
        print(" .. done")
