def to_string(x):
    assert type(x) == list
    return " ".join(x)

def mean(x):
    assert type(x) == list
    return sum(x)/len(x)
