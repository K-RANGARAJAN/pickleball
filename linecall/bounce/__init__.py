"""linecall.bounce — bounce detection + IN/OUT classification.

  classify_bounce(x, y)   -> (verdict, signed_margin_ft)
  detect_bounces_2d(...)  -> per-camera 2D bounces (see bounce_2d.py)
"""

from .bounce import BounceCall, detect_bounces, classify_bounce, IN, OUT, CLOSE_IN

__all__ = ["BounceCall", "detect_bounces", "classify_bounce", "IN", "OUT", "CLOSE_IN"]
