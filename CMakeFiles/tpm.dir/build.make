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
include CMakeFiles/tpm.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/tpm.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tpm.dir/flags.make

CMakeFiles/tpm.dir/tpm.cpp.o: CMakeFiles/tpm.dir/flags.make
CMakeFiles/tpm.dir/tpm.cpp.o: /home/james/Documents/University/ComputingProject/leapfrog_BarnesHut/src/tpm.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/james/Documents/University/ComputingProject/Python/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tpm.dir/tpm.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tpm.dir/tpm.cpp.o -c /home/james/Documents/University/ComputingProject/leapfrog_BarnesHut/src/tpm.cpp

CMakeFiles/tpm.dir/tpm.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tpm.dir/tpm.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/james/Documents/University/ComputingProject/leapfrog_BarnesHut/src/tpm.cpp > CMakeFiles/tpm.dir/tpm.cpp.i

CMakeFiles/tpm.dir/tpm.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tpm.dir/tpm.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/james/Documents/University/ComputingProject/leapfrog_BarnesHut/src/tpm.cpp -o CMakeFiles/tpm.dir/tpm.cpp.s

# Object files for target tpm
tpm_OBJECTS = \
"CMakeFiles/tpm.dir/tpm.cpp.o"

# External object files for target tpm
tpm_EXTERNAL_OBJECTS =

libtpm.a: CMakeFiles/tpm.dir/tpm.cpp.o
libtpm.a: CMakeFiles/tpm.dir/build.make
libtpm.a: CMakeFiles/tpm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/james/Documents/University/ComputingProject/Python/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libtpm.a"
	$(CMAKE_COMMAND) -P CMakeFiles/tpm.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tpm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tpm.dir/build: libtpm.a

.PHONY : CMakeFiles/tpm.dir/build

CMakeFiles/tpm.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tpm.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tpm.dir/clean

CMakeFiles/tpm.dir/depend:
	cd /home/james/Documents/University/ComputingProject/Python && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/james/Documents/University/ComputingProject/leapfrog_BarnesHut/src /home/james/Documents/University/ComputingProject/leapfrog_BarnesHut/src /home/james/Documents/University/ComputingProject/Python /home/james/Documents/University/ComputingProject/Python /home/james/Documents/University/ComputingProject/Python/CMakeFiles/tpm.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tpm.dir/depend

