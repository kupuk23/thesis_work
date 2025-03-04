cmake_minimum_required(VERSION 3.8)
project(gazebo_camera_sim)


if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
find_package(ros_gz_sim REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
find_package(gz-common5 REQUIRED)
find_package(gz-cmake3 REQUIRED)
find_package(gz-transport13 REQUIRED)
find_package(gz-msgs10 REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)

# add_library(six_dof_velocity_control SHARED src/six_dof_velocity_control.cc)
# target_include_directories(six_dof_velocity_control PUBLIC include)
# target_link_libraries(six_dof_velocity_control
#   gz-plugin2::register
#   gz-common5::gz-common5
#   gz-cmake3::gz-cmake3
#   gz-transport13::gz-transport13
#   gz-msgs10::gz-msgs10
# )
# ament_export_libraries(six_dof_velocity_control)
# ament_target_dependencies(six_dof_velocity_control
#   "ros_gz_sim",
#   "gz-plugin2",
#   "gz-common5",
#   "gz-cmake3",
#   "gz-transport13",
#   "gz-msgs10"
# )




install(DIRECTORY
  launch worlds models src
  DESTINATION share/${PROJECT_NAME}
)

# install(TARGETS
#   SixDOFVelocityControl
#   ARCHIVE DESTINATION lib
#   LIBRARY DESTINATION lib
#   RUNTIME DESTINATION bin
# )

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
