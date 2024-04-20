# Multi-Scale Registration

Multi-Scale Registration is a point cloud alignment algorithm that addresses noise and different scale management. This
project showcases our work in the field of point cloud alignment, providing a framework for aligning point clouds with
varying scales on each coordinate axis. It integrates an inner optimization block with a pattern search algorithm to
achieve accurate alignment.

# Repository Structure

The repository is organized as follows:

## YML folder

This folder contains the YAML file necessary to define the parameters and data required for the alignment process.

## UTIL folder

The UTIL folder contains a file called utils.py, which includes various utility functions that are called by the
algorithm.

## ALIGNMENT folder

The ALIGNMENT folder houses the implemented algorithm.

# Usage

To run the algorithm, follow these steps:

- Specify the path to the .npz point cloud file in the YAML file.
- Open the terminal and navigate to the alignment folder.
- Run the mcs_alignment.py script.
  Please note that this project is currently in development, and further specifications will be provided in this README.

# Equally involved contributors:

- Diego Polimeni: diego.polimeni99@gmail.com
- Alessandro Pannone: a.pannone1798@gmail.com

Feel free to reach out to the contributors if you have any questions or suggestions.

Thank you for your interest in our project!
