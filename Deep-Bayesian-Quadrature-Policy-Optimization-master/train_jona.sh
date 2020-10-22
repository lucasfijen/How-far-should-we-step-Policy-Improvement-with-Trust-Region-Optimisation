echo "Run 1"
python agent.py --env-name Swimmer-v2 --pg_algorithm TRPO --pg_estimator MC --nr-epochs 700 --seed 42
echo "Run 2"
python agent.py --env-name Swimmer-v2 --pg_algorithm TRPO --pg_estimator MC --nr-epochs 700 --seed 666
echo "Run 3"
python agent.py --env-name Swimmer-v2 --pg_algorithm TRPO --pg_estimator MC --nr-epochs 700 --seed 1337