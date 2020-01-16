import functools
import operator

# flatten a list of list quickly
def functools_reduce_iconcat(a):
    return functools.reduce(operator.iconcat, a, [])