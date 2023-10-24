# FPV_BEV Self-Driving Dataset
This CARLA simulation dataset generation code is designed to create a "first person view + bird eye view" image dataset from a realistic simulation self-driving environment. \
The code is written in Python and is based on the CARLA simulation platform (https://carla.readthedocs.io/en/latest/). \
This README file provides detailed instructions on how to use the code, its dependencies, and how to modify it for your specific use case.

## Dependencies

    Python >= 3.7
    CARLA >= 0.9.13
    PIL
    carla_birdeye_view

## File Structure
    .
    ├── ...
    ├── Town              
    │   ├── Town01.jpg           # Binary Occupancy Map
    │   ├── Town01.npy         # Candidates waypoints on the map
    │   └── _Town01.jpg       # Calibration picture
    ├── generate_data.py      # Main script
    ├── spawn_npc.py          # Script for spawning npc cars and pedestrian
    ├── collect_data.sh       # Bash script for parrallel generation
    ├── config.yaml           # Configuration file
    └── ...

## Usage
To use the code, follow these steps:\
    1. Clone or download this repository. \
    2. Install the required dependencies if not already installed. \
    3. Apply the patch provided in this repository to the installation location of carla_birdeye_view. \
    4. Open the command line and navigate to the directory containing generate_data.py. \
    5. Run the command python generate_data.py to generate the dataset. \
    Note: You can use the -h option to see a list of available arguments, including maps, image resolution, and output path.

## Modifications
This code is designed to be easily modified for your specific use case. Some possible modifications include:\
    1. Modify the configuration file config.yaml to match your specific dataset requirements. The configuration file contains parameters related to the simulation environment.\
    2. If you have multiple GPUs and want to generate data in different environment settings simultaneously, modify and execute the collect_data.sh bash script.

## Support
If you have any questions or issues with this dataset or codebase, please feel free to contact us.

## Acknowledgments
This code was developed based on the CARLA simulator and the carla_birdeye_view library (https://github.com/deepsense-ai/carla-birdeye-view). Special thanks to the developers and contributors of these projects.
