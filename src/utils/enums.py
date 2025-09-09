from enum import Enum

class League(Enum):
    NCAAMB = "NCAAMB"
    NCAAWB = "NCAAWB"
    NBA = "NBA"
    WNBA = "WNBA"

class SeasonType(Enum):
    PRE = "Pre"
    REGULAR = "Regular"
    POST = "Post"
