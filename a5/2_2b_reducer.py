#!/usr/bin/env python
import sys

max_delay = 0
old_destination = None

# Input from stdin (handled via Hadoop Streaming)
for line in sys.stdin:

    # Remove whitespace and split up lines
    line = line.strip()
    line = line.split('\t')
    if len(line) != 2:
        continue

    # Get destination and delay
    destination, arr_delay = line

    try:
        arr_delay = float(arr_delay)
    except ValueError:
        continue

    # This if-statement only works because Hadoop sorts
    # the output of the mapping phase by key (here, by
    # destination) before it is passed to the reducers. Each
    # reducer gets all the values for a given key. Each
    # reducer might get the values for MULTIPLE keys.
    if (old_destination is not None) and (old_destination != destination):
        print('%s\t%s' % (old_destination, max_delay))
        max_delay = 0

    old_destination = destination
    max_delay = max(max_delay, arr_delay)

# We have to output the destination delay for the last destination!
if old_destination is not None:
    print('%s\t%s' % (old_destination, max_delay))
