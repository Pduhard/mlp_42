class Env():

    timer = True
    seed = None

    @classmethod
    def set_timer(cls, action='on'):
        cls.timer = True if action == 'on' else False

    @classmethod
    def set_seed(cls, seed):
        cls.seed = seed
