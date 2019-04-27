from typing import List, Tuple
import numpy as np
import math

class Vec2:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def to_list(self) -> List:
        return [self.x, self.y]

    def to_np(self) -> np.ndarray:
        return np.array(self.to_list())

    def __add__(self, vec):
        return Vec2(self.x + vec.x, self.y + vec.y)

    def __radd__(self, vec):
        _v = Vec2(self.x + vec.x, self.y + vec.y)
        return _v

    def __sub__(self, vec):
        _v = Vec2(self.x - vec.x, self.y - vec.y)
        return _v

    def __rsub__(self, vec):
        _v = Vec2(vec.x - self.x, vec.y - self.y)
        return _v

    def __eq__(self, vec):
        if isinstance(vec, Vec2):
            return self.x == vec.x and self.y == vec.y
        return False

    def __mul__(self, alpha):
        """ alpha is a scalar number """
        return Vec2(alpha*self.x, alpha*self.y)

    def __rmul__(self, alpha):
        """ alpha is a scalar number """
        return Vec2(alpha*self.x, alpha*self.y)

    def __truediv__(self, alpha):
        return Vec2(self.x/alpha, self.y/alpha)

    def __div__(self, alpha):
        return Vec2(self.x/alpha, self.y/alpha)

    def __repr__(self):
        return "Vec2(%r, %r)" % (self.x, self.y)

    def __abs__(self):
        return math.sqrt(self.x*self.x + self.y*self.y)

    def __neg__(self):
        return Vec2(-self.x, -self.y)

    def __getitem__(self, index):
        if (index%2) == 0:
            return self.x
        return self.y

class BoundingBox:
    def __init__(self, tl:Vec2 = Vec2(), br:Vec2 = Vec2(), label: str = ""):
        self.top_left = tl
        self.bottom_right = br
        self.label = label

    def to_list(self):
        """ converts bounding box to list like this [[tl_x tl_y], [br_x, br_y]]"""
        return [self.top_left.to_list(), self.bottom_right.to_list()]

    def size(self) -> Tuple[float, float]:
        """ returns the width and height of the bounding box """
        return self.bottom_right.x - self.top_left.x, self.bottom_right.y - self.top_left.y

    def intersect(self, bbox) -> bool:
        """ returns if the bounding box passed as argument intersects
            this bounding box """
        return (abs(self.intersection(bbox)) > 0)

    def bound(self, tl_x, tl_y, br_x, br_y) -> None:
        self.top_left.x = max(tl_x, self.top_left.x)
        self.top_left.y = max(tl_y, self.top_left.y)
        self.bottom_right.x = min(br_x, self.bottom_right.x)
        self.bottom_right.y = min(br_y, self.bottom_right.y)
    
    def intersection(self, other):
        tl_x = max(self.top_left.x, other.top_left.x)
        tl_y = max(self.top_left.y, other.top_left.y)
        br_x = min(self.bottom_right.x, other.bottom_right.x)
        br_y = min(self.bottom_right.y, other.bottom_right.y)
        return BoundingBox(Vec2(tl_x, tl_y), Vec2(br_x, br_y))
    
    def IoU(self, a, b=None) -> float:
        if b == None:
            b = self
        return abs(a.intersection(b)) / abs(a+b)

    
    def __add__(self, b):
        tl, br = Vec2(), Vec2()
        tl.x = min(self.top_left.x, b.top_left.x)
        tl.y = min(self.top_left.y, b.top_left.y)
        br.x = max(self.bottom_right.x, b.bottom_right.x)
        br.y = max(self.bottom_right.y, b.bottom_right.y)
        return BoundingBox(tl, br)

    def __radd__(self, b):
        return self.__add__(b)

    def __str__(self) -> str:
        tl = self.top_left
        br = self.bottom_right
        return "[[%r, %r], [%r, %r]]" % (tl.x, tl.y, br.x, br.y)

    def __repr__(self) -> str:
        tl = self.top_left
        br = self.bottom_right
        return "BoundingBox(Vec2(%r, %r), Vec2(%r, %r))" % (tl.x, tl.y, br.x, br.y)

    def __mul__(self, alpha:float):
        """ alpha is a scalar number """
        center = (self.top_left + self.bottom_right) / 2.0
        diff = self.bottom_right - self.top_left
        w = alpha * Vec2(abs(diff.x)/2.0, abs(diff.y)/2.0)
        tl = center - w
        tl.x, tl.y = int(tl.x), int(tl.y)
        br = center + w
        br.x, br.y = int(br.x), int(br.y)
        return BoundingBox(tl, br)

    def __rmul__(self, alpha:float):
        return self.__mul__(alpha)

    def __abs__(self) -> float:
        diff = self.bottom_right - self.top_left
        diff.x = max(diff.x, 0)
        diff.y = max(diff.y, 0)
        return float(abs(diff))