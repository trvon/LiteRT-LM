// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_QWEN35_DATA_PROCESSOR_CONFIG_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_QWEN35_DATA_PROCESSOR_CONFIG_H_

#include <string>

namespace litert::lm {

// Config for Qwen35DataProcessor.
// Qwen3.5 is architecturally distinct from Qwen3: it uses a hybrid
// SSM/full-attention architecture (some layers use linear attention with
// conv_state + recurrent_state tensors) and supports image input.
struct Qwen35DataProcessorConfig {
  // Vision modality token strings for Qwen3.5 multimodal.
  // These are different from Gemma's <start_of_image>/<end_of_image>.
  std::string start_of_vision_token = "<|vision_start|>";
  std::string image_pad_token = "<|image_pad|>";
  std::string end_of_vision_token = "<|vision_end|>";

  // Image dimensions the vision encoder expects after preprocessing.
  int image_tensor_height = 392;
  int image_tensor_width = 392;

  // Tool calling options (same convention as Qwen3).
  std::string code_fence_start = "<tool_call>";
  std::string code_fence_end = "</tool_call>";
  bool escape_fence_strings = true;
  std::string tool_code_regex = "";
};

struct Qwen35DataProcessorArguments {};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_QWEN35_DATA_PROCESSOR_CONFIG_H_
