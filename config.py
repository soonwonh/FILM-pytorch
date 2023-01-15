project_name = 'film_style'


data_params = {
    'batch_size': 16
}


model_params = {
    'pyramid_levels': 7,
    'fusion_pyramid_levels': 5,
    'specialized_levels': 3,
    'sub_levels': 4,
    'flow_convs': [3, 3, 3, 3],
    'flow_filters': [32, 64, 128, 256],
    'filters': 64
}

train_params = {
    'learning_rate': 0.0001*0.5,
    'learning_rate_decay_steps': 750000,
    'learning_rate_decay_rate': 0.464158,
    'learning_rate_staircase': True,
    'num_steps': 3000000,
    'weight_decay': 1e-3
}
