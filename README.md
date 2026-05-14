# RHCR
![test_ubuntu](https://github.com/Jiaoyang-Li/RHCR/actions/workflows/test_ubuntu.yml/badge.svg)
![test_macos](https://github.com/Jiaoyang-Li/RHCR/actions/workflows/test_macos.yml/badge.svg)

Rolling-Horizon Collision Resolution (RHCR) is an efficient algorithm for solving lifelong Multi-Agent Path Finding (MAPF) where we are aksed to plan collision-free paths for a large number of agents that are constanly engaged with new goal locations. RHCR calls a Windowed MAPF solver every h timesteps that resolves collisions only for the next w timesteps (w >= h). More details can be found in our extended abstract at AAMAS 2020 [1] and our full paper at AAAI 2021 [2].

The code requires the external library BOOST (https://www.boost.org/).    
Here is an easy way of installing BOOST in Linux:
```shell script
sudo apt install libboost-all-dev
```

After you installed BOOST and downloaded the source code, go into the directory of the source code and compile it with CMake: 
```
cmake .
make
```


## License
RHCR is released under USC – Research License. See license.md for further details.
 
## References
[1] Jiaoyang Li, Andrew Tinka, Scott Kiesel, Joseph W. Durham, T. K. Satish Kumar and Sven Koenig. Lifelong Multi-Agent Path Finding in Large-Scale Warehouses (extended abstract). In Proceedings of the International Joint Conference on Autonomous Agents and Multiagent Systems (AAMAS), pages 1898-1900, 2020.

[2] Jiaoyang Li, Andrew Tinka, Scott Kiesel, Joseph W. Durham, T. K. Satish Kumar and Sven Koenig. Lifelong Multi-Agent Path Finding in Large-Scale Warehouses. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), (in print), 2021.
