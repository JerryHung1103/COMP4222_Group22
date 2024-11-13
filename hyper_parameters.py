HYPERPARAMETERS = {
    "model_embedding_size": 128,  # Increase embedding size
    "model_attention_heads": 8,    # Increase attention heads
    "model_layers": 4,              # Increase layers
    "model_dropout_rate": 0.2,      # Lower dropout rate
    "model_top_k_ratio": 0.5,
    "model_top_k_every_n": 1,
    "model_dense_neurons": 256,     # Increase dense neurons
    "batch_size": 64,                # Smaller batch size
    "learning_rate": 0.01,          # Lower learning rate
    "sgd_momentum": 0.9,
    "weight_decay": 0.0001,            # Adjust weight decay
    "scheduler_gamma": 0.95,
    "pos_weight": 1.5,               # Adjust to give more weight to positive class
}

# HYPERPARAMETERS = {
#     "batch_size": 128,
#     "learning_rate": 0.01,
#     "weight_decay": 0.0001,
#     "sgd_momentum": 0.8,
#     "scheduler_gamma": 0.8,
#     "pos_weight": 1.3,
#     "model_embedding_size": 64,
#     "model_attention_heads": 3,
#     "model_layers": 4,
#     "model_dropout_rate": 0.2,
#     "model_top_k_ratio": 0.5,
#     "model_top_k_every_n": 1,
#     "model_dense_neurons": 256
# }