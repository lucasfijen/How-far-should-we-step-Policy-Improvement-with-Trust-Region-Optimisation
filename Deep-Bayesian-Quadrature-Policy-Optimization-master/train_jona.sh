echo "Run 1 TRPO 1337"
python agent.py --env-name Ant-v2 --pg_algorithm TRPO --pg_estimator MC --nr-epochs 700 --seed 1337
echo "Run 2 NPG"
python agent.py --env-name Ant-v2 --pg_algorithm NPG --pg_estimator MC --nr-epochs 700 --seed 42 --lr 1e-3

echo "Run 3 NPG"
python agent.py --env-name Ant-v2 --pg_algorithm NPG --pg_estimator MC --nr-epochs 700 --seed 666 --lr 1e-3

echo "Run 4 NPG"
python agent.py --env-name Ant-v2 --pg_algorithm NPG --pg_estimator MC --nr-epochs 700 --seed 1337 --lr 1e-3