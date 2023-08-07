import pstats
from pstats import SortKey

p = pstats.Stats("./src/prof")
p.sort_stats(SortKey.TIME).print_stats(100)