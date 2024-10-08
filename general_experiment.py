from robust_fl import *
import argparse

color1 = '#1b9e77'
color2 = '#d95f02'
color3 = '#7570b3'

def run_experiment_known_dir(N_experiments, N, N_true, d, batch_size, epsilon,
                             num_test_samples, W_true, a_true, b_true, target_func, **kwargs):
    W_true = torch.randn(N_true, d, device=device)
    W_true /= torch.linalg.norm(W_true, dim=1).reshape(-1, 1)
    a_true = 1 / np.sqrt(N_true) * torch.ones(N_true, device=device)
    b_true = torch.zeros(N_true, device=device)
    
    X_test = torch.randn(10000, d, device=device)
    y_test = target_func(X_test @ W_true.T + b_true) @ a_true
    
    all_costs_adv = []
    all_costs_init_fixed = []
    all_costs_init_trained = []
    N_experiments = 3
    for i in range(N_experiments):
        print(f'Experiment: {i}')
        print()
        init_W = torch.randn(N, d)
        init_W /= torch.linalg.norm(init_W, dim=1).reshape(-1, 1)
        init_b = torch.randn(N)
        init_a = torch.randn(N) / N

        print('Robust test risks - full AD training:')
        model1 = TwoLayerNN(init_W, init_b, init_a).to(device)
        costs1 = online_train(model1, batch_size, epsilon, X_test, y_test, W_true, a_true, b_true, target_func, train_first_layer=True,
                              **kwargs)
        all_costs_adv.append(costs1)
        print()

        print('Robust test risks - W fixed at target u:')
        model2 = TwoLayerNN(W_true.repeat(N, 1), init_b, init_a).to(device)
        costs2 = online_train(model2, batch_size, epsilon, X_test, y_test, W_true, a_true, b_true, target_func, train_first_layer=False,
                              **kwargs)
        all_costs_init_fixed.append(costs2)
        print()

        print('Robust test risks - W init from u:')
        model3 = TwoLayerNN(W_true.repeat(N, 1), init_b, init_a).to(device)
        costs3 = online_train(model3, batch_size, epsilon, X_test, y_test, W_true, a_true, b_true, target_func, train_first_layer=True,
                             **kwargs)
        all_costs_init_trained.append(costs3)
        print()
        
    return torch.tensor(all_costs_adv), torch.tensor(all_costs_init_fixed), torch.tensor(all_costs_init_trained)


def run_experiment_unknown_dir(N_experiments, N, N_true, d, batch_size, epsilon,
                             num_test_samples, W_true, a_true, b_true, target_func, init_ab_random, **kwargs):
    W_true = torch.randn(N_true, d, device=device)
    W_true /= torch.linalg.norm(W_true, dim=1).reshape(-1, 1)
    a_true = 1 / np.sqrt(N_true) * torch.ones(N_true, device=device)
    b_true = torch.zeros(N_true, device=device)
    
    X_test = torch.randn(10000, d, device=device)
    y_test = target_func(X_test @ W_true.T + b_true) @ a_true
    
    all_costs_adv = []
    all_costs_init_fixed = []
    all_costs_init_trained = []
    N_experiments = 3
    for i in range(N_experiments):
        print(f'Experiment: {i}')
        print()
        init_W = torch.randn(N, d)
        init_W /= torch.linalg.norm(init_W, dim=1).reshape(-1, 1)
        init_b = torch.randn(N)
        init_a = torch.randn(N) / N

        print('Robust test risks - full AD training:')
        model1 = TwoLayerNN(init_W, init_b, init_a).to(device)
        costs1 = online_train(model1, batch_size, epsilon, X_test, y_test, W_true, a_true, b_true, target_func, train_first_layer=True,
                              **kwargs)
        all_costs_adv.append(costs1)
        print()
        
        print('Standard test risks - preparing a standard model:')
        standard_model = TwoLayerNN(init_W, init_b, init_a).to(device)
        standard_costs = online_train(standard_model, batch_size, 0, X_test, y_test, W_true, a_true, b_true, target_func, train_first_layer=True,
                                      num_iter_train=kwargs.get('num_iter_std_train', 5000), 
                                      num_iter_adv_train=0, 
                                      num_iter_adv_test=0, 
                                      stepsize_train=kwargs.get('stepsize_std_train', 0.01))
        print()

        print('Robust test risks - W fixed at SD training:')
        if init_ab_random:
            model2 = TwoLayerNN(standard_model.W.data.clone(), init_b, init_a).to(device)
        else:
            model2 = TwoLayerNN(standard_model.W.data.clone(), standard_model.b.data.clone(), standard_model.a.data.clone()).to(device)
        costs2 = online_train(model2, batch_size, epsilon, X_test, y_test, W_true, a_true, b_true, target_func, train_first_layer=False,
                              **kwargs)
        all_costs_init_fixed.append(costs2)
        print()

        print('Robust test risks - W init from SD training:')
        if init_ab_random:
            model3 = TwoLayerNN(standard_model.W.data.clone(), init_b, init_a).to(device)
        else:
            model3 = TwoLayerNN(standard_model.W.data.clone(), standard_model.b.data.clone(), standard_model.a.data.clone()).to(device)
         
        costs3 = online_train(model3, batch_size, epsilon, X_test, y_test, W_true, a_true, b_true, target_func, train_first_layer=True,
                             **kwargs)
        all_costs_init_trained.append(costs3)
        print()
        
    return torch.tensor(all_costs_adv), torch.tensor(all_costs_init_fixed), torch.tensor(all_costs_init_trained)


def plot_experiments(all_costs_adv, all_costs_init_fixed, all_costs_init_trained, target, known_dir,
                    test_points=100, iterations=5000):
    costs_adv_mean = all_costs_adv.mean(dim=0)
    costs_adv_std = all_costs_adv.std(dim=0)

    costs_init_fixed_mean = all_costs_init_fixed.mean(dim=0)
    costs_init_fixed_std = all_costs_init_fixed.std(dim=0)

    costs_init_trained_mean = all_costs_init_trained.mean(dim=0)
    costs_init_trained_std = all_costs_init_trained.std(dim=0)
    
    
    tick_pos = [i * iterations // test_points for i in range(test_points)]

    plt.figure(figsize=(6.4 / 1.5, 4.8 / 1.5))
    if known_dir:
        plt.title(f'Teacher: {target} (Known Direction)')
    else:
        plt.title(f'Teacher: {target} (Unknown Direction)')
    plt.ylabel('Robust Test Risk')
    plt.xlabel('Iterations')

    plt.plot(tick_pos, costs_adv_mean, label='Full AD training', linewidth=2, color=color1)
    plt.fill_between(tick_pos, costs_adv_mean - costs_adv_std, costs_adv_mean + costs_adv_std, color=color1,
                     alpha=0.3)
    
    init_fixed_label = '$W$ fixed at $u$' if known_dir else '$W$ fixed at SD training'
    plt.plot(tick_pos, costs_init_fixed_mean, label=init_fixed_label, linewidth=2, color=color2)
    plt.fill_between(tick_pos, costs_init_fixed_mean - costs_init_fixed_std, costs_init_fixed_mean + costs_init_fixed_std,
                     color=color2, alpha=0.3)
    
    init_trained_label = '$W$ initialized from $u$' if known_dir else '$W$ initialized from SD training'
    plt.plot(tick_pos, costs_init_trained_mean, label=init_trained_label, linewidth=2, color=color3)
    plt.fill_between(tick_pos, costs_init_trained_mean - costs_init_trained_std, costs_init_trained_mean + costs_init_trained_std, 
                     color=color3, alpha=0.3)

    plt.legend()
    if known_dir:
        plt.savefig(f'figures/{target}_known.pdf', bbox_inches='tight')
    else:
        plt.savefig(f'figures/{target}_unknown.pdf', bbox_inches='tight')
    plt.show()

def He2(z):
    return (z ** 2 - 1) / np.sqrt(2)


parser = argparse.ArgumentParser()
parser.add_argument('--N_experiments', type=int, default=3)
parser.add_argument('--N', type=int, default=100)
parser.add_argument('--N_true', type=int, default=1)
parser.add_argument('--d', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--num_test_samples', type=int, default=10000)
parser.add_argument('--epsilon', type=float, default=1.0)
parser.add_argument('--num_iter_train', type=int, default=5000)
parser.add_argument('--num_iter_adv_train', type=int, default=5)
parser.add_argument('--num_iter_adv_test', type=int, default=20)
parser.add_argument('--stepsize_train', type=float, default=0.001)
parser.add_argument('--stepsize_adv_train', type=float, default=0.1)
parser.add_argument('--stepsize_adv_test', type=float, default=0.1)
parser.add_argument('--num_iter_std_train', type=int, default=5000)
parser.add_argument('--stepsize_std_train', type=float, default=0.01)
parser.add_argument('--init_ab_random', action='store_true')

parser.add_argument('--target_func', choices=['relu', 'tanh', 'he2'])
parser.add_argument('--known_dir', action='store_true')
parser.add_argument('--no-known_dir', action='store_false')

if __name__=='__main__':
    torch.manual_seed(42)
    args = parser.parse_args()

    W_true = torch.randn(args.N_true, args.d, device=device)
    W_true /= torch.linalg.norm(W_true, dim=1).reshape(-1, 1)
    a_true = 1 / np.sqrt(args.N_true) * torch.ones(args.N_true, device=device)
    b_true = torch.zeros(args.N_true, device=device)

    if args.target_func == 'relu':
        target_func = F.relu
    if args.target_func == 'tanh':
        target_func = F.tanh
    if args.target_func == 'he2':
        target_func = He2
    
    if args.known_dir:
        all_costs_adv, all_costs_init_fixed, all_costs_init_trained = run_experiment_known_dir(N_experiments=args.N_experiments, 
                                                                                    N=args.N, N_true=args.N_true, 
                                                                                    d=args.d, batch_size=args.batch_size,
                                                                                    epsilon=args.epsilon,
                                                                                    num_test_samples=args.num_test_samples,
                                                                                    W_true=W_true, 
                                                                                    a_true=a_true, b_true=b_true,
                                                                                    target_func=target_func, num_iter_train=args.num_iter_train)

    else:
        all_costs_adv, all_costs_init_fixed, all_costs_init_trained = run_experiment_unknown_dir(N_experiments=args.N_experiments, 
                                                                                    N=args.N, N_true=args.N_true, 
                                                                                    d=args.d, batch_size=args.batch_size,
                                                                                    epsilon=args.epsilon,
                                                                                    num_test_samples=args.num_test_samples,
                                                                                    W_true=W_true, 
                                                                                    a_true=a_true, b_true=b_true,
                                                                                    target_func=target_func, init_ab_random=args.init_ab_random,
                                                                                    num_iter_train=args.num_iter_train,
                                                                                    num_iter_std_train=args.num_iter_std_train,
                                                                                    stepsize_std_train=args.stepsize_std_train)

    if args.known_dir:
        torch.save(all_costs_adv, f'results/{args.target_func}_known_costs_adv.pt')
    else:
        torch.save(all_costs_adv, f'results/{args.target_func}_unknown_costs_adv.pt')
    
    if args.known_dir:
        torch.save(all_costs_init_fixed, f'results/{args.target_func}_known_costs_init_fixed.pt')
    else:
        torch.save(all_costs_init_fixed, f'results/{args.target_func}_unknown_costs_init_fixed.pt')

    if args.known_dir:
        torch.save(all_costs_init_trained, f'results/{args.target_func}_known_costs_init_trained.pt')
    else:
        torch.save(all_costs_init_trained, f'results/{args.target_func}_unknown_costs_init_trained.pt')

    target_name = {'relu': 'ReLU', 'tanh': 'Tanh', 'he2': 'He2'}
    plot_experiments(all_costs_adv, all_costs_init_fixed, all_costs_init_trained, target=target_name[args.target_func],
                                     known_dir=args.known_dir)
