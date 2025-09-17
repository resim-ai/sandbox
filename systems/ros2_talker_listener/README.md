# ROS2 Talker / Listener

This system is intended to demonstrate a trivial [multi-container
build](https://docs.resim.ai/guides/multi-container-builds/) wherein one service contains a ros2
`demo_nodes_cpp` talker and the other contains a listener. Finally, we have an orchestrator service
that terminates the sim once 60 seconds have elapsed.
