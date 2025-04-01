import argparse
import pathlib
import numpy as np
import torch
import yaml
import sys
import time
import cv2
import os

# Environment setting for MuJoCo rendering
os.environ["MUJOCO_GL"] = "glfw"  # Change to "egl" or "osmesa" if needed

# Import modules from the project
import envs.dmc as dmc
import envs.wrappers as wrappers

# Create a simplified visualization script
def main(config):

    
    # Create environment
    print(f"Creating environment: {config.task}")
    domain, task = config.task.split("_", 1)
    
    # Convert size to tuple to avoid concatenation issues
    config.size = tuple(config.size) if isinstance(config.size, list) else config.size
    
    # Create the base environment without wrappers first
    env = dmc.DeepMindControl(task, config.action_repeat, config.size)
    
    # Now add wrappers in the right order
    env = wrappers.NormalizeActions(env)
    env = wrappers.TimeLimit(env, config.time_limit)
    
    # This is the last wrapper before SelectAction, so we can get actions in the right format
    base_env = env
    
    # Add remaining wrappers
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    
    # Load the saved checkpoint (just to check it exists)
    print("Loading checkpoint...")
    
    # Visualization loop
    print("\nStarting visualization with random policy...")
    window_title = f"DreamerV3 - {config.task}"
    episodes = 0
    max_episodes = 5
    
    while episodes < max_episodes:
        print(f"\nStarting episode {episodes+1}")
        obs = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        # Add a resize factor parameter to control window size
        resize_factor = 6.0  # Increase this to make the window larger

        # In the rendering loop, modify the display section:
        # Use fixed display size instead of proportional scaling
        display_width = 500  # Adjust to your preferred width
        display_height = 300  # Adjust to your preferred height

        # In the rendering loop:
        while not done:
            # Generate random action and action dict
            action = np.random.uniform(-1, 1, (base_env.action_space.shape[0],))
            action_dict = {"action": action}
            
            # Render the frame
            frame = env.render(mode='rgb_array')
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Use fixed size for display
            display_frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_CUBIC)
            
            # Add info text with appropriate font size and position
            font_scale = 0.8
            text_thickness = 2
            text_color = (255, 255, 255)  # White
            
            # Add black outline around text for better readability
            cv2.putText(display_frame, f"Step: {step} | Reward: {episode_reward:.2f}", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 
                        text_thickness + 2)
            
            # Add white text over the outline
            cv2.putText(display_frame, f"Step: {step} | Reward: {episode_reward:.2f}", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 
                        text_thickness)
            
            # Display the frame
            cv2.imshow(window_title, display_frame)
            key = cv2.waitKey(20)  # Adjust for speed (higher = slower)
            
            if key == 27 or key == ord('q') or key == ord('Q'):
                print("Visualization stopped by user")
                cv2.destroyAllWindows()
                return
            
            # Environment step with action in the correct format
            obs, reward, done, _ = env.step(action_dict)
            episode_reward += reward
            step += 1
            
            # Slow down visualization slightly
            time.sleep(0.01)
        
        print(f"Episode {episodes+1} finished with reward {episode_reward:.2f} in {step} steps")
        episodes += 1
    
    cv2.destroyAllWindows()
    print("Visualization complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs="+", default=["dmc_vision"], help='Configuration name(s)')
    task_prefix = "dmc_vision"
    parser.add_argument('--task', type=str, help='Override task name (optional)')
    args = parser.parse_args()
    
    # Load configuration from configs.yaml
    configs = yaml.safe_load((pathlib.Path(sys.argv[0]).parent.parent / "configs.yaml").read_text())
    
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value
    
    defaults = {}
    for name in ["defaults"] + args.configs:
        recursive_update(defaults, configs[name])
    
    # Convert to object with attributes
    class Config:
        pass
    
    config = Config()
    for key, value in defaults.items():
        # Try to convert string to float if it looks like a number
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        setattr(config, key, value)

    if args.task:
        setattr(config, "task", args.task)

    main(config)