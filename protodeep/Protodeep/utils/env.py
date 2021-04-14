class Env():

    timer = True

    @classmethod
    def set_timer(cls, action='on'):
        cls.timer = True if action == 'on' else False
