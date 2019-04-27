from geometry import BoundingBox

class Prediction:
    def __init__(self, box: BoundingBox, label: str, confidence: float):
        self.bbox = box
        self.label = label
        self.confidence = confidence