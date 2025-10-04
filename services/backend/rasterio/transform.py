from dataclasses import dataclass


@dataclass
class Affine:
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float

    def __call__(self, col: float, row: float) -> tuple[float, float]:
        x = self.a * col + self.b * row + self.c
        y = self.d * col + self.e * row + self.f
        return x, y

    def __iter__(self):
        yield from (self.a, self.b, self.c, self.d, self.e, self.f)

    @classmethod
    def identity(cls) -> "Affine":
        return cls(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)


def from_origin(west: float, north: float, xsize: float, ysize: float) -> Affine:
    return Affine(xsize, 0.0, west, 0.0, -ysize, north)
