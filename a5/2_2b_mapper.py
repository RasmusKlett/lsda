#!/usr/bin/env python
import sys


# Where do these lines come from?
# This is done by Hadoop Streaming ...
for line in sys.stdin:
    # Skip the header
    if line.startswith("\"FL_DATE"):
        continue

    sys.stdin.readline()

    # Remove whitespace and split up line
    # into words (whitespace as delimiter)
    line = line.strip()

    columns = line.split(",")
    destination = columns[4]
    arr_delay = columns[8]

    if arr_delay and float(arr_delay) > 0:
        print('%s\t%s' % (destination, arr_delay))
