from .empty import Empty, RectangleEmpty, ResetableEmpty
from .piped import Piped
from .non_convex import NonConvex
from .stones import StandardStones, ThickStones, AlmostEmpty

__all__ = ["Empty", "RectangleEmpty", "ResetableEmpty", "Piped",
           "NonConvex", "StandardStones", "ThickStones", "AlmostEmpty", "RectangleEmpty"]


str2map = {
    "Empty": Empty,
    "RectangleEmpty": RectangleEmpty,
    "ResetableEmpty": ResetableEmpty,
    "Piped": Piped,
    "NonConvex": NonConvex,
    "StandardStones": StandardStones,
    "ThickStones": ThickStones,
    "AlmostEmpty": AlmostEmpty,
}
