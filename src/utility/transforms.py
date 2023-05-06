
# torch.transforms.Lambda can't be pickled : transforms.Lambda(lambda x: (x.repeat(3, 1, 1) if x.size(0)==1 else x))
def EnsureRGB(x):
    return x.repeat(3, 1, 1) if x.size(0)==1 else x