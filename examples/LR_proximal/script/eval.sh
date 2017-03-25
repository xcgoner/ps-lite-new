#!/bin/bash

for entry in "/home/ubuntu/cx2/data/results/a9a/weight_track"*
do
  	if [[ "$entry" != *eval ]]
	then
		python lreval_mpi.py -s 1 -n 8 -S serverhost1 -W workerhost1 -e "$entry" -t "/home/ubuntu/cx2/data/a9a_data/part-" /home/ubuntu/cx2/src/ps-lite-new/bin/lrprox
	    echo "$entry"
	fi
done