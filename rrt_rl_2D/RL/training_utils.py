import inspect


def get_name(base_name):
    return base_name + str(inspect.stack()[1][3])
