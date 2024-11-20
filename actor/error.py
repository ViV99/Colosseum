class ActorError(Exception):
    def __init__(self, type: str, code: int):
        self.type = type
        self.code = code
