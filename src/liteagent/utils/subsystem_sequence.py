class SubsystemSequence(list):
    def __init__(self):
        self.next_subsystem_idx = 0
        self.next_subsystem_name = None
        self.done = False

    def append(self, object):
        super().append(object)
        if self.next_subsystem_idx < len(self):
            self.next_subsystem_name = self[self.next_subsystem_idx]
        else:
            self.next_subsystem_idx = 0
            self.next_subsystem_name = None

    def remove(self, value):
        super().remove(value)
        if self.next_subsystem_idx < len(self):
            self.next_subsystem_name = self[self.next_subsystem_idx]
        else:
            self.next_subsystem_idx = len(self) - 1
            self.next_subsystem_name = self[self.next_subsystem_idx]

    def update_next_subsystem_name(self, idx):
        if idx < len(self):
            raise Exception(f"Index {idx} is out of range!")
        self.next_subsystem_idx = idx
        self.next_subsystem_name = self[self.next_subsystem_idx]

    def update_next_subsystem_idx(self, name):
        if name not in self:
            raise Exception(f"Agent system {name} is not valid!")
        self.next_subsystem_name = name
        self.next_subsystem_idx = self.index(name)

    def update_next_subsystem(self):
        self.next_subsystem_idx = (self.next_subsystem_idx + 1) % len(self)
        self.next_subsystem_name = self[self.next_subsystem_idx]

    def get_current_subsystem_name(self):
        return self.next_subsystem_name
    
    def set_is_done(self):
        self.done = True

    def is_done(self):
        return self.done