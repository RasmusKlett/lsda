#!/usr/bin/env python
import sys

total_delay = 0
old_origin = None

# Input from stdin (handled via Hadoop Streaming)
for line in sys.stdin:

    # Remove whitespace and split up lines
    line = line.strip()
    line = line.split('\t')
    if len(line) != 2:
        continue

    # Get origin and delay
    origin, dep_delay = line

    try:
        dep_delay = float(dep_delay)
    except ValueError:
        continue

    # This if-statement only works because Hadoop sorts
    # the output of the mapping phase by key (here, by
    # origin) before it is passed to the reducers. Each
    # reducer gets all the values for a given key. Each
    # reducer might get the values for MULTIPLE keys.
    if (old_origin is not None) and (old_origin != origin):
        print('%s\t%s' % (old_origin, total_delay))
        total_delay = 0

    old_origin = origin
    total_delay += dep_delay

# We have to output the origin delay for the last origin!
if old_origin is not None:
    print('%s\t%s' % (old_origin, total_delay))
