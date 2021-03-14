#!/bin/bash
for file in *.fastq
do
  OUTPUT=$file.grep
  echo $OUTPUT
   grep -oP "(?<=ACCG)(.{17,27})(?=GTTT)" $file > $OUTPUT.sg
done
