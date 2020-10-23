#!/bin/bash

#source ~/anaconda3/etc/profile.d/conda.sh
#conda activate DBQPG
if [ $1 != TRPO ] && [ $1 != NPG ] && [ $1 != vanilla ]; then
	echo Enter a valid optimizing scheme \(vanilla, NPG, TRPO\)
	Error: wrong input given	
	exit 1
fi


#python agent.py --env-name $1 --pg_algorithm $2 --pg_estimator $3 --seed $4
#python agent.py --env-name $1 --pg_algorithm $2 --pg_estimator $3 --seed $4
#python agent.py --env-name $1 --pg_algorithm $2 --pg_estimator $3 --seed $4

