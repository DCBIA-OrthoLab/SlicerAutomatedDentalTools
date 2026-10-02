"""The OFF reader now lives in ADTLib.

The two copies -- this one and the one in `ASO/ASO_Method/IOS_utils/Reader.py`
-- were identical to the character, logging preamble aside. This module stays
for the callers that import it by its original path.
"""
from ADTLib.io.surface import OFFReader  # noqa: F401  (re-exported)
