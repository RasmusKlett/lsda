#!/usr/bin/env python
import sys

# Where do these lines come from?
# This is done by Hadoop Streaming ...
for line in sys.stdin:
    # Skip the header
    if line.startswith("\"FL_DATE"):
        continue

    # Remove whitespace and split up line
    # into words (whitespace as delimiter)
    line = line.strip()

    columns = line.split(",")
    origin = columns[3]
    dep_delay = columns[6]

    if dep_delay and float(dep_delay) > 0:
        print('%s\t%s' % (origin, dep_delay))
