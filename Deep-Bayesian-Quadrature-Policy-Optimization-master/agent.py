from datetime import date, datetime
import platform
import gym
import torch
import gpytorch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from optimization import update_params
from utils import *
import datetime as dt
from arguments import get_args
import fast_svd
from results_writer import ResultsRow, ResultsWriter

args = get_args()

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
torch.set_default_tensor_type('torch.DoubleTensor')
torch.manual_seed(args.seed)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# Setting up the environment
print('------------ Starting ' + args.env_name + ' Environment ------------')
env = gym.make(args.env_name)
env.seed(args.seed)
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
print(' State Dimensions :- ', num_inputs)
print(' Action Dimensions :- ', num_actions)
print(args.output_directory)

results_writer = ResultsWriter(args.run_label,f'{args.output_directory}', pg_model=args.pg_algorithm)
args.device = torch.device("cuda" if args.pg_estimator == "BQ" else "cpu")

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# Instantiating the policy and value function neural networks, GP's MLL objective function and their respective optimizers
print('Going to run for the following PG algorithms :- ', args.pg_algorithm)
print('Policy Gradient Estimator :- ' +
      (('UAPG' if args.UAPG_flag else 'DBQPG'
        ) if args.pg_estimator == "BQ" else 'MC') + "\n")

policy_net = Policy(num_inputs, num_actions).to(args.device)
# TRPO does not require a policy optimizer since it uses a KL divergence constraint for robust step-size selection.
policy_optimizer = torch.optim.Adam(
    policy_net.parameters(), lr=args.lr,
    weight_decay=1e-3) if args.pg_algorithm != "TRPO" else None

value_net, gp_mll, likelihood, gp_value_optimizer, nn_value_optimizer = None, None, None, None, None
if args.pg_estimator == "BQ":
    # Instantiating the value network with both GP and value heads for approximating Q(s,a) and V(s)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(
            args.likelihood_noise_level)).to(args.device)
    value_net = Value(num_inputs, args.pg_estimator, args.svd_low_rank,
                      likelihood).to(args.device)
    gp_mll = gpytorch.mlls.ExactMarginalLogLikelihood(
        likelihood, value_net)  # Used for computing GP critic's loss
    GP_params = [
        param for name, param in value_net.named_parameters()
        if 'value_head' not in name
    ]
    gp_value_optimizer = torch.optim.Adam(
        GP_params, lr=0.0001,
        weight_decay=1e-3)  # Used for optimizing GP critic
    value_net.train()
    likelihood.train()

if args.advantage_flag:
    if args.pg_estimator == 'MC':
        # Instantiating the value network with only value head for approximating V(s)
        value_net = Value(
            num_inputs,
            args.pg_estimator)  # subsequently used for computing GAE estimates
    NN_params = [
        param for name, param in value_net.named_parameters()
        if ('feature_extractor' in name or 'value_head' in name)
    ]
    nn_value_optimizer = torch.optim.Adam(NN_params,
                                          lr=0.0001,
                                          weight_decay=1e-3)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# Policy Optimization
running_state = ZFilter((num_inputs, ), clip=5)
start_time = dt.datetime.now()
STEPS = 0
for iteration in range(1, args.nr_epochs):
    memory = Memory()
    num_steps = 0
    batch_reward = 0
    num_episodes = 0
    batch_durations = []
    # Collecting sampled data through policy roll-out in the environment
    while num_steps < args.batch_size:
        state = env.reset()
        state = running_state(state)
        reward_sum = 0
        for t in range(args.max_episode_steps):  # Don't infinite loop while learning
            # Simulates one episode, i.e., until the agent reaches the terminal state or has taken 10000 steps in the environment
            action_mean, action_log_std, action_std = policy_net(
                Variable(torch.Tensor([state])).to(args.device))
            action = torch.normal(action_mean,
                                  action_std).detach().data[0].cpu().numpy()
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward
            next_state = running_state(next_state)
            mask = 0 if done else 1
            memory.push(state, np.array([action]), mask, next_state, reward)
            state = next_state
            if args.render:
                env.render()
            if done:
                break
        num_steps += (t - 1)
        num_episodes += 1
        batch_reward += reward_sum

    mean_steps = num_steps / num_episodes
    mean_episode_reward = batch_reward / num_episodes
    batch = memory.sample()
    # Policy & Value Function Optimization
    step_size = update_params(args, batch, policy_net, value_net, policy_optimizer,
                  likelihood, gp_mll, gp_value_optimizer, nn_value_optimizer)
    STEPS += 1

    print('Iteration {:4d} - Average reward {:.3f} - Time elapsed: {:3d}sec'.
          format(iteration, mean_episode_reward,
                 (dt.datetime.now() - start_time).seconds))

    results_writer.add(results=ResultsRow(
        # All `run_` properties remain the same across savings of a run
        run_label=args.run_label,
        run_nr_epochs=args.nr_epochs,
        run_model=args.pg_algorithm,
        env=args.env_name,
        perf=mean_episode_reward,
        timestamp=datetime.now().strftime('%m_%d_%Y_%H_%M_%S'),
        hardware=platform.node(),
        seed=args.seed,
        epoch_duration=(dt.datetime.now() - start_time).seconds,
        epoch=iteration,
        step_size=step_size if args.pg_algorithm is 'TRPO' else args.lr ,
        nr_steps=mean_steps,
        nr_episodes=num_episodes,
    ))

    start_time = dt.datetime.now()