"""Application-layer exceptions mapped by HTTP layer."""


class InvalidInputError(Exception):
    pass


class InvalidImageError(Exception):
    pass


class NoFaceDetectedError(Exception):
    pass
