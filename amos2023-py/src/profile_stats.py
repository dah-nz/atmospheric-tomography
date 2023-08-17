import pstats
from pstats import SortKey

p = pstats.Stats("./src/test_data")
p.sort_stats(SortKey.TIME).print_stats(30)