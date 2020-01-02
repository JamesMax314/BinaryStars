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
CMAKE_COMMAND = /snap/clion/99/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /snap/clion/99/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/james/Documents/University/ComputingProject/leapfrog_BarnesHut/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/james/Documents/University/ComputingProject/Python

# Include any dependencies generated for this target.
include CMakeFiles/vecMaths.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/vecMaths.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/vecMaths.dir/flags.make

CMakeFiles/vecMaths.dir/vecMaths.cpp.o: CMakeFiles/vecMaths.dir/flags.make
CMakeFiles/vecMaths.dir/vecMaths.cpp.o: /home/james/Documents/University/ComputingProject/leapfrog_BarnesHut/src/vecMaths.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/james/Documents/University/ComputingProject/Python/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/vecMaths.dir/vecMaths.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vecMaths.dir/vecMaths.cpp.o -c /home/james/Documents/University/ComputingProject/leapfrog_BarnesHut/src/vecMaths.cpp

CMakeFiles/vecMaths.dir/vecMaths.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vecMaths.dir/vecMaths.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/james/Documents/University/ComputingProject/leapfrog_BarnesHut/src/vecMaths.cpp > CMakeFiles/vecMaths.dir/vecMaths.cpp.i

CMakeFiles/vecMaths.dir/vecMaths.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vecMaths.dir/vecMaths.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/james/Documents/University/ComputingProject/leapfrog_BarnesHut/src/vecMaths.cpp -o CMakeFiles/vecMaths.dir/vecMaths.cpp.s

# Object files for target vecMaths
vecMaths_OBJECTS = \
"CMakeFiles/vecMaths.dir/vecMaths.cpp.o"

# External object files for target vecMaths
vecMaths_EXTERNAL_OBJECTS =

libvecMaths.a: CMakeFiles/vecMaths.dir/vecMaths.cpp.o
libvecMaths.a: CMakeFiles/vecMaths.dir/build.make
libvecMaths.a: CMakeFiles/vecMaths.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/james/Documents/University/ComputingProject/Python/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libvecMaths.a"
	$(CMAKE_COMMAND) -P CMakeFiles/vecMaths.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vecMaths.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/vecMaths.dir/build: libvecMaths.a

.PHONY : CMakeFiles/vecMaths.dir/build

CMakeFiles/vecMaths.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/vecMaths.dir/cmake_clean.cmake
.PHONY : CMakeFiles/vecMaths.dir/clean

CMakeFiles/vecMaths.dir/depend:
	cd /home/james/Documents/University/ComputingProject/Python && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/james/Documents/University/ComputingProject/leapfrog_BarnesHut/src /home/james/Documents/University/ComputingProject/leapfrog_BarnesHut/src /home/james/Documents/University/ComputingProject/Python /home/james/Documents/University/ComputingProject/Python /home/james/Documents/University/ComputingProject/Python/CMakeFiles/vecMaths.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/vecMaths.dir/depend

