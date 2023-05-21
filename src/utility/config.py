import copy

class NotFound:
    pass

class Config:

    def __init__(self, config):
        self.build(config)

    def build(self, config):
        self.__dict__.update(config)
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                self.__dict__[key] = Config(**value)

    def primitive(self, dict=None):
        dict = copy.deepcopy(self.__dict__) if dict is None else dict
        for key, value in dict.items():
            if isinstance(value, Config):
                dict[key] = self.primitive(value)
        return dict

    # def extend(self, new, original=None):
    #     original = self.primitive(original)
    #     if isinstance(new, Config):
    #         new = new.primitive()

    #     for key, value in new.items():
    #         if original.get(key, NotFound) is NotFound:
    #             original[key] = value
    #         else:
    #             if isinstance(value, list):

    def __repr__(self):
        return str(self.primitive())