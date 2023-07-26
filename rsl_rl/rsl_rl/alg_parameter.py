

class args:
    task = 'LeggedRobotCfgPPO'
    resume = False
    experiment_name = 'RL_minitaur'
    run_name = 'basic_RL'
    load_run = -1
    checkpoint = -1
        
    headless = False
    horovod = False
    rl_device = 'cpu'  # just for now
    sim_device_type = 'cpu'
    compute_device_id = 0
    num_envs = 1 # just for now.. Else 16
    seed = 1234
    max_iterations = 100  # just for now subject to changes