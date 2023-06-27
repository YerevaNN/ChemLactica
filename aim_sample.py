from aim import Run

run = Run(repo=".", experiment="exp 2")
hyper_params_dict = {
    'learning_rate': 0.001,
    'batch_size': 32,
}

run["hyper_params"] = hyper_params_dict

for i in range(10):
    run.track(i, name='number 1')

for j in range(10, 0, -1):
    run.track(j, name='number 1')