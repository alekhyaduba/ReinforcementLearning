import argparse
import gym
import os
import sys
import pickle
import time

from utils import *
from torch import nn
from agent import Agent

from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy

from point_env import PointEnv
from solutions.point_mass_solutions import estimate_net_grad_v, estimate_net_grad
import solutions.cart_pole_solutions as cp

directory_for_saving_files = 'Results'

parser = argparse.ArgumentParser(description='Pytorch Policy Gradient')
parser.add_argument('--env-name', default="Point-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--learning-rate', type=float, default=0.1, metavar='G',
                    help='gae (default: 3e-4)')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--eval-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size for evaluation (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=100, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--version', type=int, default=1,
                    help='choose 1 for equation 1, 2 for equation 2 and 3 for equation 3')
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
args = parser.parse_args()


def main(result):

    """environment"""
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    try:
        is_disc_action = len(env.action_space.shape) == 0
    except:
        is_disc_action = True

    action_dim = 1 if is_disc_action else env.action_space.shape[0]

    """seeding"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    """cuda setting"""
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        print('using gpu')
        torch.cuda.set_device(args.gpu_index)


    """define actor and critic"""
    if args.env_name == 'Point-v0':
        # we use only a linear policy for this environment
        theta = torch.normal(0, 0.01, size=(state_dim + 1, action_dim))
        policy_net = None
        theta = theta.to(dtype).to(device)
    else:
        # we use both a policy and a critic network for this environment
        policy_net = DiscretePolicy(state_dim, env.action_space.n)
        theta = None
        value_net = Value(state_dim)
        to_device(device, policy_net, value_net)

        # Optimizers
        optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
        optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)


    """create agent"""
    if args.env_name == 'Point-v0':
        agent = Agent(env, args.env_name, device, policy_net, theta, custom_reward=None,
                  running_state=None, num_threads=args.num_threads)
    else:
        agent = Agent(env, args.env_name, device, policy_net, theta, custom_reward=None,
                      running_state=None, num_threads=args.num_threads)


    def update_params(batch, i_iter, last_reward):
        states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
        actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
        rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
        masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)

        if args.env_name == 'CartPole-v0':
            with torch.no_grad():
                values = value_net(states)

            if args.version == 1:
                returns = cp.estimate_returns(rewards, masks, args.gamma, device)
                cp.pg_step(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, returns)

            elif args.version == 2:
                rtg = cp.estimate_rtg(rewards, masks, args.gamma, device)
                cp.pg_step(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, rtg)

            elif args.version == 3:
                values_no_grad = value_net(states)
                advantages, returns = cp.estimate_advantages(rewards, masks, values, args.gamma, device)
                adv_no_grad, ret_no_grad = cp.estimate_advantages(rewards, masks, values_no_grad, args.gamma, device)
                cp.pg_step_adv(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, advantages, adv_no_grad)

        if args.env_name == 'Point-v0':
            # agent.theta = estimate_net_grad(rewards,masks,states,actions,args.gamma,agent.theta,device,args.version)
            #
            # agent.theta += agent.theta*args.learning_rate

            agent.theta, last_reward = estimate_net_grad_v(rewards, masks, states, actions, args.gamma, agent.theta, device, args.version,
                                        args.learning_rate, last_reward)

        return last_reward

    def main_loop(result):
        last_reward = 0
        for i_iter in range(args.max_iter_num):
            # results=[]
            best_reward = -100

            """generate multiple trajectories that reach the minimum batch_size"""
            batch, log = agent.collect_samples(args.min_batch_size, render=args.render)

            t0 = time.time()
            last_reward = update_params(batch, i_iter, last_reward)
            # print(last_reward)
            t1 = time.time()
            """evaluate with determinstic action (remove noise for exploration) For cartpole,set mean action to False"""
            if args.env_name == 'CartPole-v0':
                mean_action = False
            else:
                mean_action = True
            _, log_eval = agent.collect_samples(args.eval_batch_size, mean_action=mean_action)
            t2 = time.time()
            result['iter_num'].append(i_iter)
            result['avg_reward'].append(log['avg_reward'])
            result['eval_reward'].append(log_eval['avg_reward'])

            if i_iter % args.log_interval == 0:
                print('{}\tT_sample {:.4f}\tT_update {:.4f}\ttrain_R_avg {:.2f}\teval_R_avg {:.2f}'.format(
                    i_iter, log['sample_time'], t1-t0, log['avg_reward'], log_eval['avg_reward']))

            if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0 and args.env_name == 'CartPole-v0':
                to_device(torch.device('cpu'), policy_net, value_net)
                # you will have to specify a proper directory to save files

                pickle.dump((policy_net, value_net), open(os.path.join(directory_for_saving_files, 'learned_models/{}_{}policy_grads.p'.format(args.env_name,args.version)), 'wb'))

                to_device(device, policy_net, value_net)
            if args.env_name == 'Point-v0' and log_eval['avg_reward'] >= best_reward:

                pickle.dump(agent.theta, open(os.path.join(directory_for_saving_files,'learned_models/{}_{}theta.track'.format(args.env_name,args.version)),'wb'))


            """clean up gpu memory"""
            torch.cuda.empty_cache()
        pickle.dump(result, open(os.path.join(directory_for_saving_files,
                                                             'rewards/{}_{}_avg_reward.track'.format(args.env_name,
                                                                                                   args.version)),
                                                'wb'))


    main_loop(result)


if __name__ == '__main__':
    result = {'iter_num': [],
              'avg_reward': [],
              'eval_reward': []}
    main(result)
