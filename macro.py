import sys

def p2(a):
    return ((a)*(a))

def p3(a):
    return ((a)*(a)*(a))

def print_progressbar(i, N):
    # progressbar
    pbwidth = 42

    progress = float(i)/N
    block = int(round(pbwidth*progress))
    text = "\rProgress: [{0}] {1:.1f}%".format( "#"*block + "-"*(pbwidth-block), progress*100)
    sys.stdout.write(text)
    sys.stdout.flush()

    if i == (N-1):
        print " .. done"
