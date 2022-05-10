"""
dummy temporary model
"""

from random import randint


class Model:
    def __init__(self):
        self.x = 10
        self.y = 10

    def predict(self):
        self.x += randint(-2, 10)
        self.y += randint(-2, 7)
        return self.x, self.y
