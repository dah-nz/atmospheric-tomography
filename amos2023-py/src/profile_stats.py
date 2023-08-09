import pstats
from pstats import SortKey

p = pstats.Stats("prof")
p.sort_stats(SortKey.TIME).print_stats(100)