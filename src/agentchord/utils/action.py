class Action:
    def __init__(self, action_name, action):
        self.action_name = action_name
        self.action = action

    def __call__(self, *args, **kwargs):
        return self.action(*args, **kwargs)