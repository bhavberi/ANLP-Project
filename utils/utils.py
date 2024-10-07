def get_5label_from_11label(label):
    if label=="none":
        return 0

    return int(label[0])

def reverse_dict(dictionary):
    return {v: k for k, v in dictionary.items()}