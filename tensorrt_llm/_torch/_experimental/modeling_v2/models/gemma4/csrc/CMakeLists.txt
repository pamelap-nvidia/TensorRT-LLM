#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

# gemma4 native code. Each target keeps its own in <target>/csrc/; see the
# modeling_v2 README.
add_modeling_v2_family(gemma4)

# The Kernel Factory solution the decoder-stack megakernel comes from was tuned
# with fast math. Set before the target is declared: its objects read the
# property when they are created.
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/../e2b_it_fp8__sm_120__tp1/csrc/gemma4E2bDecodeKernel.cu
  PROPERTIES COMPILE_OPTIONS "--use_fast_math")

# The megakernel is written for SM120 (RTX PRO 6000 Blackwell Server Edition):
# FP8 mma.sync and cp.async.bulk, no tcgen05.
add_modeling_v2_target(
  gemma4 e2b_it_fp8__sm_120__tp1
  ARCHS 120a
  SOURCES *.cu *.cpp)
