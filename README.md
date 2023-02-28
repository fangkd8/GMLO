# Ground and Memory Optimized LiDAR Odometry (GMLO).

The implementation for GMLO in C++ with OpenMP acceleration. 

## Getting Started

Please prepared the point cloud files in [KITTI Odometry dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) and unzip them. Then edit the line 116 of the codes [kitti.cpp](./exec/kitti.cpp) with your dataset storage directory.

## Prerequisites

* `PCL`
* `OpenMP`
* `Sophus`
* `Eigen`
Please note that the codes are developed with Ubuntu 22.04 environments, the libs are with the default version of the system.

## Description

For the Voxel-to-Voxel GICP, please refer to [gicp_mp.hpp](./include/gicp_mp.hpp) and [gicp_mp.cpp](./src/gicp_mp.cpp), which is used as the base method for Scan-to-Mapping.

The PR-GPF algorithm is implemented with [polar_region_gpf.hpp](./include/polar_region_gpf.hpp) and [polar_region_gpf.cpp](./src/polar_region_gpf.cpp).

The joint-matching process is implemented in [joint_matching.cpp](./src/joint_matching.cpp) with Gauss-Newton optimization method.


## Contributing

Kaiduo FANG      - kaiduo8.fang@connect.polyu.hk
Ivan Wang-Hei Ho - ivanwh.ho@polyu.edu.hk

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
