import sys


def parse_file_name(default_name):
    file_name = default_name
    for i, arg in enumerate(sys.argv):
        if i > 0:
            file_name = str(arg)
    return file_name


def float_try_parse(value):
    try:
        _ = float(value)
        return True
    except ValueError:
        return False


def vfloat_try_parse(value):
    try:
        val = float(value)
        return val, True
    except ValueError:
        return None, False


def get_color(house):
    if house == 'Gryffindor':
        return 'yellow'
    elif house == 'Slytherin':
        return 'green'
    elif house == 'Ravenclaw':
        return 'blue'
    elif house == 'Hufflepuff':
        return 'orange'
    return 'grey'
