# Initial premises
This project is currently in development and further specifications will be included on this README.
Equally involved contributors:
- diego.polimeni99@gmail.com
- a.pannone1798@gmail.com

# Multi-scale-registration 
Pointc loud Registration Algorithm with noise and different scale management.

This project illustrate our piece of work in the field of point cloud alignment offering a framework that manages the presence of noise on source and target point clouds. We integrate the inner optimization block whit a pattern search algorithm in order to align point clouds with different scale on each coordinate axes.

# File structure

## YML folder
Contains the yaml file needed in order to define the parameters and data necessary for the alignment

## UTIL folder
Contains the utils.py file useful functions called by the algorithm

## ALIGNMENT folder
Contains the algorithm implemented.

## Run
To run the algorthm specify the .npz pointcloud path into the yaml and run the mcs_alignment.py from terminal inside the alignment folder.


