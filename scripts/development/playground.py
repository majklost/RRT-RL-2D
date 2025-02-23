import inspect


class CableRadius:
    @staticmethod
    def obs_vel_stronger_fast(cur_map=None, render_mode=None):

        return get_name(__class__.__name__ + '-')


def get_name(base_name):
    return base_name + str(inspect.stack()[1][3])


if __name__ == '__main__':
    t = CableRadius()
    name = t.obs_vel_stronger_fast()
    print(name)
