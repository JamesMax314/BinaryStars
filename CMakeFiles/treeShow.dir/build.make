# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/clion/100/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /snap/clion/100/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/james/Documents/University/ComputingProject/leapfrog_BarnesHut/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/james/Documents/University/ComputingProject/Python

# Include any dependencies generated for this target.
include CMakeFiles/treeShow.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/treeShow.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/treeShow.dir/flags.make

CMakeFiles/treeShow.dir/treeShow.cpp.o: CMakeFiles/treeShow.dir/flags.make
CMakeFiles/treeShow.dir/treeShow.cpp.o: /home/james/Documents/University/ComputingProject/leapfrog_BarnesHut/src/treeShow.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/james/Documents/University/ComputingProject/Python/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/treeShow.dir/treeShow.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/treeShow.dir/treeShow.cpp.o -c /home/james/Documents/University/ComputingProject/leapfrog_BarnesHut/src/treeShow.cpp

CMakeFiles/treeShow.dir/treeShow.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/treeShow.dir/treeShow.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/james/Documents/University/ComputingProject/leapfrog_BarnesHut/src/treeShow.cpp > CMakeFiles/treeShow.dir/treeShow.cpp.i

CMakeFiles/treeShow.dir/treeShow.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/treeShow.dir/treeShow.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/james/Documents/University/ComputingProject/leapfrog_BarnesHut/src/treeShow.cpp -o CMakeFiles/treeShow.dir/treeShow.cpp.s

# Object files for target treeShow
treeShow_OBJECTS = \
"CMakeFiles/treeShow.dir/treeShow.cpp.o"

# External object files for target treeShow
treeShow_EXTERNAL_OBJECTS =

libtreeShow.a: CMakeFiles/treeShow.dir/treeShow.cpp.o
libtreeShow.a: CMakeFiles/treeShow.dir/build.make
libtreeShow.a: CMakeFiles/treeShow.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/james/Documents/University/ComputingProject/Python/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libtreeShow.a"
	$(CMAKE_COMMAND) -P CMakeFiles/treeShow.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/treeShow.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/treeShow.dir/build: libtreeShow.a

.PHONY : CMakeFiles/treeShow.dir/build

CMakeFiles/treeShow.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/treeShow.dir/cmake_clean.cmake
.PHONY : CMakeFiles/treeShow.dir/clean

CMakeFiles/treeShow.dir/depend:
	cd /home/james/Documents/University/ComputingProject/Python && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/james/Documents/University/ComputingProject/leapfrog_BarnesHut/src /home/james/Documents/University/ComputingProject/leapfrog_BarnesHut/src /home/james/Documents/University/ComputingProject/Python /home/james/Documents/University/ComputingProject/Python /home/james/Documents/University/ComputingProject/Python/CMakeFiles/treeShow.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/treeShow.dir/depend

