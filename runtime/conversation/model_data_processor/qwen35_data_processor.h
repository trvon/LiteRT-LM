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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_QWEN35_DATA_PROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_QWEN35_DATA_PROCESSOR_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "runtime/components/preprocessor/image_preprocessor.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/qwen35_data_processor_config.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {

// Qwen35DataProcessor handles Qwen3.5 multimodal models.
//
// Qwen3.5 is architecturally distinct from Qwen3:
//   - Hybrid SSM/full-attention: some layers use linear attention with
//     conv_state + recurrent_state tensors alongside standard KV-cache layers.
//   - Multimodal: image input via <|vision_start|>/<|image_pad|>/<|vision_end|>
//     tokens, NOT Gemma's <start_of_image>/<end_of_image>.
//
// This processor:
//   1. Emits InputImage entries (image-first) before the InputText for each
//      multimodal turn, matching the 'image_first' token ordering expected by
//      the LiteRT-LM multimodal runtime.
//   2. Uses the correct Qwen3.5 vision token placeholders in template rendering.
class Qwen35DataProcessor
    : public TypeSafeModelDataProcessor<Qwen35DataProcessorConfig,
                                        Qwen35DataProcessorArguments> {
 public:
  static absl::StatusOr<std::unique_ptr<ModelDataProcessor>> Create(
      Qwen35DataProcessorConfig config,
      std::optional<Preface> preface = std::nullopt,
      std::unique_ptr<ImagePreprocessor> image_preprocessor = nullptr);

  absl::StatusOr<nlohmann::ordered_json> FormatTools(
      const nlohmann::ordered_json& tools) const override {
    return tools;
  }

  absl::StatusOr<nlohmann::ordered_json> MessageToTemplateInput(
      const nlohmann::ordered_json& message) const override;

  absl::string_view CodeFenceStart() const override;
  absl::string_view CodeFenceEnd() const override;

  const Qwen35DataProcessorConfig& GetConfig() const override {
    return config_;
  }

 private:
  explicit Qwen35DataProcessor(
      Qwen35DataProcessorConfig config, std::optional<Preface> preface,
      std::unique_ptr<ImagePreprocessor> image_preprocessor)
      : config_(std::move(config)),
        preface_(std::move(preface)),
        image_preprocessor_(std::move(image_preprocessor)) {}

  absl::StatusOr<std::vector<InputData>> ToInputDataVectorImpl(
      const std::string& rendered_template_prompt,
      const nlohmann::ordered_json& messages,
      const Qwen35DataProcessorArguments& args) const override;

  absl::StatusOr<Message> ToMessageImpl(
      const Responses& responses,
      const Qwen35DataProcessorArguments& args) const override;

  absl::Status CloneStateImpl(
      const TypeSafeModelDataProcessor<Qwen35DataProcessorConfig,
                                       Qwen35DataProcessorArguments>& other)
      override {
    ABSL_VLOG(2) << "Qwen35DataProcessor::CloneStateImpl is a no-op.";
    return absl::OkStatus();
  }

  Qwen35DataProcessorConfig config_;
  std::optional<Preface> preface_;
  std::unique_ptr<ImagePreprocessor> image_preprocessor_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_QWEN35_DATA_PROCESSOR_H_
