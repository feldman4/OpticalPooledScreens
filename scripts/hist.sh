#!/bin/bash

TMP=lbftmp
mkdir $TMP
for f in *.grep.sg
do
  OUTPUT=${f}.hist
  cat $f | sort -T $TMP | uniq -c | sort -b -n -r -T $TMP > $OUTPUT
done
rm -r $TMP
