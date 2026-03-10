from Classes.Point import Point


class Line:
    def __init__(self, point1: Point, point2: Point):
        self.point1 = point1
        self.point2 = point2

    def __str__(self):
        return f"{self.point1}-{self.point2}"

    def __getitem__(self, key):
        if key == "point 1" or key == 1:
            return self.point1
        elif key == "point 2" or key == 2:
            return self.point2

    def __eq__(self, __o: object) -> bool:
        out = False
        if self.point1 == __o[1] and self.point2 == __o[2]:
            out = True
        return out

    def __setitem__(self, key, value):
        if key == "position":
            self.point1["position"] = value
            self.point2["position"] = value
