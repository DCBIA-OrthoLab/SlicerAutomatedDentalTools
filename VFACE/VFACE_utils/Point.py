import numpy as np
import logging
import sys

# ===== Logging Configuration =====
logger = logging.getLogger("VFACE_Point")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.handlers:
    logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class Point:
    def __init__(self, name: str, time: str):
        self.name = name
        self.position = []
        self.time = time

    def __str__(self):

        return self.name

    def __getitem__(self, key):
        if key == "name":
            return self.name
        elif key == "position":
            return self.position

    def __setitem__(self, key, value):
        if key == "position":
            """value = {"T1":{"A":[0,3,1],"B":[0,3,5],...},
                  "T2":{"A":[8,3,5],"B":[9,2,5],...}}
      """
            position = value[self.time][self.name.upper()]

            good = False
            if isinstance(position, list):
                if len(position) == 3:
                    if not False in [
                        isinstance(value, (int, float, np.ndarray))
                        for value in position
                    ] and not True in np.isnan(position):
                        good = True
            if not good:
                raise KeyError(self.name, self.time)

            self.position = position

    def __eq__(self, __o: object) -> bool:
        out = False
        if self.name == __o["name"]:
            out = True
        return out
