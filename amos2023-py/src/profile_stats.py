import pstats
from pstats import SortKey

p = pstats.Stats("./src/main_profile")
p.sort_stats(SortKey.TIME).print_stats(100)