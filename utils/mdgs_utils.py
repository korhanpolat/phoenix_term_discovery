import re

def clean_label(x):

    # if x[0] == '$': return '$'
    x = x.strip('*').strip('^')
    x = re.split('1|2|3|4|5|6|7|8|9|0',x)[0]

    return x


def clean_label_list(labels):
    res = []
    for x in label:
        res.append(clean_label(x))

    return res
