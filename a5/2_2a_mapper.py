#!/usr/bin/env python
import sys

# Skip the header
sys.stdin.readline()

# Where do these lines come from?
# This is done by Hadoop Streaming ...
for line in sys.stdin:

    # Remove whitespace and split up line
    # into words (whitespace as delimiter)
    line = line.strip()

    columns = line.split(",")
    origin = columns[3]
    dep_delay = columns[6]

    if dep_delay and float(dep_delay) > 0:
        print('%s\t%s' % (origin, dep_delay))
