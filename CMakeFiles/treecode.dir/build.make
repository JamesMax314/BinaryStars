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
CMAKE_COMMAND = /snap/clion/103/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /snap/clion/103/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/james/Documents/University/ComputingProject/leapfrog_BarnesHut/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/james/Documents/University/ComputingProject/Python

# Include any dependencies generated for this target.
include CMakeFiles/treecode.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/treecode.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/treecode.dir/flags.make

CMakeFiles/treecode.dir/pyInterface.cpp.o: CMakeFiles/treecode.dir/flags.make
CMakeFiles/treecode.dir/pyInterface.cpp.o: /home/james/Documents/University/ComputingProject/leapfrog_BarnesHut/src/pyInterface.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/james/Documents/University/ComputingProject/Python/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/treecode.dir/pyInterface.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/treecode.dir/pyInterface.cpp.o -c /home/james/Documents/University/ComputingProject/leapfrog_BarnesHut/src/pyInterface.cpp

CMakeFiles/treecode.dir/pyInterface.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/treecode.dir/pyInterface.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/james/Documents/University/ComputingProject/leapfrog_BarnesHut/src/pyInterface.cpp > CMakeFiles/treecode.dir/pyInterface.cpp.i

CMakeFiles/treecode.dir/pyInterface.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/treecode.dir/pyInterface.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/james/Documents/University/ComputingProject/leapfrog_BarnesHut/src/pyInterface.cpp -o CMakeFiles/treecode.dir/pyInterface.cpp.s

# Object files for target treecode
treecode_OBJECTS = \
"CMakeFiles/treecode.dir/pyInterface.cpp.o"

# External object files for target treecode
treecode_EXTERNAL_OBJECTS =

treecode.cpython-36m-x86_64-linux-gnu.so: CMakeFiles/treecode.dir/pyInterface.cpp.o
treecode.cpython-36m-x86_64-linux-gnu.so: CMakeFiles/treecode.dir/build.make
treecode.cpython-36m-x86_64-linux-gnu.so: libtrees.a
treecode.cpython-36m-x86_64-linux-gnu.so: libvecMaths.a
treecode.cpython-36m-x86_64-linux-gnu.so: libleapfrog.a
treecode.cpython-36m-x86_64-linux-gnu.so: libbodies.a
treecode.cpython-36m-x86_64-linux-gnu.so: libtreeShow.a
treecode.cpython-36m-x86_64-linux-gnu.so: libpoisson.a
treecode.cpython-36m-x86_64-linux-gnu.so: libtpm.a
treecode.cpython-36m-x86_64-linux-gnu.so: /usr/lib/gcc/x86_64-linux-gnu/7/libgomp.so
treecode.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libpthread.so
treecode.cpython-36m-x86_64-linux-gnu.so: CMakeFiles/treecode.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/james/Documents/University/ComputingProject/Python/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module treecode.cpython-36m-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/treecode.dir/link.txt --verbose=$(VERBOSE)
	/usr/bin/strip /home/james/Documents/University/ComputingProject/Python/treecode.cpython-36m-x86_64-linux-gnu.so

# Rule to build all files generated by this target.
CMakeFiles/treecode.dir/build: treecode.cpython-36m-x86_64-linux-gnu.so

.PHONY : CMakeFiles/treecode.dir/build

CMakeFiles/treecode.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/treecode.dir/cmake_clean.cmake
.PHONY : CMakeFiles/treecode.dir/clean

CMakeFiles/treecode.dir/depend:
	cd /home/james/Documents/University/ComputingProject/Python && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/james/Documents/University/ComputingProject/leapfrog_BarnesHut/src /home/james/Documents/University/ComputingProject/leapfrog_BarnesHut/src /home/james/Documents/University/ComputingProject/Python /home/james/Documents/University/ComputingProject/Python /home/james/Documents/University/ComputingProject/Python/CMakeFiles/treecode.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/treecode.dir/depend

