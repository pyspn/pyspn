#!/bin/bash
read n
all_files=""
for (( i=0; i < n; i++))
  do
    type="_prd"
    if ! ((i % 2))
    then
      type="_sum"
    fi

    fname="mask_$i$type"
    all_files="$all_files kdpsilitonga@daytona.cs.uwaterloo.ca:~/cspn/ConvSPN/$fname"

  done

echo "$all_files"
scp all_files ./
