# Schoomatic Simulator

The Schoomatic Simulator is an advanced differential-drive robot simulator built on top of the CARLA simulation platform. It is designed to facilitate the development and testing of various vision-based robot navigation algorithms, including path planner, motion controller, and reinforcement learning methods.

This simulator is versatile, supporting custom maps, dynamic weather conditions, pedestrian and traffic simulations, and a variety of task settings. It is an ideal tool for evaluating the effectiveness and robustness of different robot navigation strategies.

## File Structure

Below is the directory structure of the Schoomatic Simulator:

```plaintext
.
├── config
│   └── carla_config.yaml         # Simulation and task configurations
├── envs
│   └── env.py                    # Simulation Environment
├── models
│   ├── encoder.py                # Encoders for BEV and FPV images
│   └── lstm.py                   # LSTM for robust perception
├── utils
│   └── misc.py                   # Helper functions, paths to pre-trained models
├── getmap.py                     # Vision-based map retrieval and Robustness State Check
├── test_planner.py               # Path Planner-based Navigation
├── test_policy.py                # Reinforcement Learning-based Navigation
└── ...
```
## Usage

To start using the Schoomatic Simulator, perform the following steps: 
1. Configure your simulation settings in `config/carla_config.yaml`. 
2. (Optional) Adjust `getmap.py` to integrate your perception models. 
3. (Optional) Modify `utils/misc.py` to point to your pre-trained model paths. 

4. Initialize the server with the following command:
```sh
   docker run --rm --gpus all --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw klekkala/carla_schoomatic /bin/bash /LinuxNoEditor/CarlaUE4.sh -RenderOffScreen
```
5. Start the client using either: \
   • `python test_planner.py` for ROS-based path planning and control.  \
   • `python test_policy.py` for evaluating a reinforcement learning policy. 

    
The client script will evaluate the effectiveness of your method by calculating the average number of successfully reached waypoints per rollout.
    
**Note**: The Schoomatic Simulator is not limited to vision-based approaches; it is a general-purpose differential-drive robot simulator. We plan to release our pretrained weights following the acceptance of our research paper.

## Acknowledgments

This code was developed by building upon the [Scoomatic](https://git.rz.uni-augsburg.de/schoerma/scoomatic/-/tree/6cfc841f946b7a6120b7967ce8e62eae718c072f) project. We extend our heartfelt gratitude to the original developers and contributors for their foundational work which made this simulator possible.

