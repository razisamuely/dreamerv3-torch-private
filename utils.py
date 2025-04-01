import torch


def verify_model_dimensions(agent, checkpoint, config):
    """
    Verifies that the dimensions of the loaded model match the current configuration.
    
    Args:
        agent: The agent object to load the checkpoint into
        checkpoint: The loaded checkpoint dictionary
        config: The current configuration object
    
    Returns:
        bool: True if dimensions match, False otherwise
    """
    try:
        # Get the action space dimension from the agent's dynamics model
        if hasattr(agent._wm.dynamics, '_num_actions'):
            checkpoint_action_dim = agent._wm.dynamics._num_actions
            current_action_dim = config.num_actions
            
            # Check if the dimensions match
            if checkpoint_action_dim != current_action_dim:
                print(f"WARNING: Action dimension mismatch!")
                print(f"Checkpoint action dimension: {checkpoint_action_dim}")
                print(f"Current config action dimension: {current_action_dim}")
                return False
        
        # Check the first layer of the img_in_layers which connects to the action space
        # Get the weight shape of the first linear layer
        if hasattr(agent._wm.dynamics, '_img_in_layers'):
            for module in agent._wm.dynamics._img_in_layers:
                if isinstance(module, torch.nn.Linear):
                    checkpoint_weight = checkpoint["agent_state_dict"]["_wm.dynamics._img_in_layers.0.weight"]
                    current_weight_shape = module.weight.shape
                    
                    if checkpoint_weight.shape != current_weight_shape:
                        print(f"WARNING: Linear layer dimension mismatch!")
                        print(f"Checkpoint weight shape: {checkpoint_weight.shape}")
                        print(f"Current model weight shape: {current_weight_shape}")
                        return False
                    break
        
        return True
    except Exception as e:
        print(f"Error during model dimension verification: {e}")
        return False