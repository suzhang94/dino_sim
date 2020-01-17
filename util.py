import numpy

def distance(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])

def bound_check(loc, size):
    #print(loc[0])
    if loc[0]<0:
        loc[0] = 0
    if loc[0]>size-1:
        loc[0] = size-1

    if loc[1]<0:
        loc[1]= 0
    if loc[1]>size-1:
        loc[1] = size-1
    return loc

def running_mean(x, N):
    cumsum = numpy.cumsum(numpy.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def loc_to_num(loc, size):
    return loc[0]*size+loc[1]

def num_to_loc(num, size):
    x = num//size
    y = num % size
    loc = [x, y]
    return loc