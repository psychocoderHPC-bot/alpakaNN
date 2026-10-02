#
# SPDX-FileCopyrightText: René Widera
# SPDX-License-Identifier: MPL-2.0
#

cmake_minimum_required(VERSION 3.25)

# The quickstart is compiled (not run as a test) so that the public API shown in
# README.md is checked by the regular build.
add_executable(alpakaNN_quickstart quickstart.cpp)
target_link_libraries(alpakaNN_quickstart PRIVATE alpaka::nn alpaka::alpaka)
alpaka_finalize(alpakaNN_quickstart)
