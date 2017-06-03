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
    dest = columns[4]
    arr_delay = columns[8]

    if arr_delay and float(arr_delay) > 0:
        print('%s\t%s' % (dest, arr_delay))
