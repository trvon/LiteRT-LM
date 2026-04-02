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

#include "runtime/conversation/model_data_processor/qwen35_data_processor.h"

#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "litert/cc/litert_layout.h"  // from @litert
#include "runtime/components/preprocessor/image_preprocessor.h"
#include "runtime/components/preprocessor/stb_image_preprocessor.h"
#include "runtime/components/tool_use/parser_utils.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/data_utils.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/qwen35_data_processor_config.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

/*static*/ absl::StatusOr<std::unique_ptr<ModelDataProcessor>>
Qwen35DataProcessor::Create(Qwen35DataProcessorConfig config,
                             std::optional<Preface> preface,
                             std::unique_ptr<ImagePreprocessor> image_preprocessor) {
  if (image_preprocessor == nullptr) {
    image_preprocessor = std::make_unique<StbImagePreprocessor>();
  }
  ABSL_VLOG(2) << "Creating Qwen35DataProcessor (hybrid SSM/attention, "
                  "multimodal). vision_start='"
               << config.start_of_vision_token << "'";
  return absl::WrapUnique(new Qwen35DataProcessor(
      std::move(config), std::move(preface), std::move(image_preprocessor)));
}

absl::StatusOr<nlohmann::ordered_json>
Qwen35DataProcessor::MessageToTemplateInput(
    const nlohmann::ordered_json& message) const {
  if (message["content"].is_array()) {
    const auto& content = message["content"];
    if (content.size() == 1 && content[0].contains("text")) {
      // Collapse single-text-item arrays to plain string (matches Qwen3 behavior).
      auto result = nlohmann::ordered_json::object(
          {{"role", message["role"]}, {"content", content[0]["text"]}});
      return result;
    }
  }
  return message;
}

absl::StatusOr<std::vector<InputData>>
Qwen35DataProcessor::ToInputDataVectorImpl(
    const std::string& rendered_template_prompt,
    const nlohmann::ordered_json& messages,
    const Qwen35DataProcessorArguments& args) const {
  const std::string& vision_start = config_.start_of_vision_token;
  const std::string& vision_end = config_.end_of_vision_token;

  // Collect image files in turn order from messages.
  std::deque<std::unique_ptr<MemoryMappedFile>> image_files;
  for (const auto& message : messages) {
    if (message.contains("content") && message["content"].is_array()) {
      for (const auto& item : message["content"]) {
        if (!item.is_string() && item.contains("type") &&
            item["type"] == "image") {
          ASSIGN_OR_RETURN(std::unique_ptr<MemoryMappedFile> mmap_file,
                           LoadItemData(item));
          image_files.push_back(std::move(mmap_file));
        }
      }
    }
  }

  if (image_files.empty()) {
    if (rendered_template_prompt.find(vision_start) != std::string::npos) {
      return absl::InvalidArgumentError(
          "Qwen35DataProcessor: template contains image placeholders but "
          "messages do not provide any images.");
    }
    // Text-only path: return the rendered prompt as a single InputText.
    ABSL_VLOG(2) << "Qwen35DataProcessor: text-only path, no images in messages.";
    std::vector<InputData> input_data;
    input_data.emplace_back(InputText(rendered_template_prompt));
    return input_data;
  }

  // Multimodal path: split the rendered prompt on the vision-start token and
  // interleave InputImage entries (image-first ordering).
  //
  // The Jinja template for kQwen3_5 emits
  //   <|vision_start|><|image_pad|><|vision_end|>
  // as the placeholder for each image. We split on <|vision_start|> and then
  // consume everything up to and including <|vision_end|>, emitting:
  //   InputText(text_before_vision_start)
  //   InputImage(preprocessed_image)
  //   InputText(<|vision_start|>...<|vision_end|> remainder)
  //
  // NOTE: The LiteRT-LM multimodal pipeline replaces the image_pad section
  // with actual vision embeddings at the native layer, so we pass the full
  // vision token sequence through as rendered_text and let the runtime handle
  // the embedding injection. We only need to signal the image data boundary.

  std::vector<InputData> input_data;

  ImagePreprocessParameter image_params;
  image_params.SetTargetDimensions(Dimensions(
      {1, config_.image_tensor_height, config_.image_tensor_width, 3}));

  std::string remaining = rendered_template_prompt;
  while (true) {
    auto start_pos = remaining.find(vision_start);
    if (start_pos == std::string::npos) {
      // No more image placeholders; emit the remaining text.
      if (!remaining.empty()) {
        input_data.emplace_back(InputText(std::move(remaining)));
      }
      break;
    }

    // Emit text before the vision_start token.
    if (start_pos > 0) {
      input_data.emplace_back(InputText(remaining.substr(0, start_pos)));
    }

    // Find the matching vision_end.
    auto end_pos = remaining.find(vision_end, start_pos);
    if (end_pos == std::string::npos) {
      // Malformed template — no closing vision_end. Emit remainder as text.
      ABSL_LOG(WARNING)
          << "Qwen35DataProcessor: found '" << vision_start
          << "' without matching '" << vision_end
          << "' in rendered template. Treating as text.";
      input_data.emplace_back(InputText(remaining.substr(start_pos)));
      break;
    }

    // Emit the image data before the vision token sequence.
    if (image_files.empty()) {
      return absl::InvalidArgumentError(
          "Qwen35DataProcessor: more image placeholders in template than "
          "images provided in messages.");
    }
    auto image_file = std::move(image_files.front());
    image_files.pop_front();

    ASSIGN_OR_RETURN(
        auto preprocessed_image,
        image_preprocessor_->Preprocess(
            InputImage(std::string(
                static_cast<const char*>(image_file->data()),
                image_file->length())),
            image_params));
    input_data.emplace_back(InputImage(std::move(preprocessed_image)));

    // Emit the vision token sequence (vision_start ... vision_end) as text so
    // the runtime knows where to inject the embeddings.
    size_t vision_seq_end = end_pos + vision_end.size();
    input_data.emplace_back(
        InputText(remaining.substr(start_pos, vision_seq_end - start_pos)));

    remaining = remaining.substr(vision_seq_end);
  }

  if (!image_files.empty()) {
    return absl::InvalidArgumentError(
        "Qwen35DataProcessor: more images provided in messages than image "
        "placeholders present in template.");
  }

  ABSL_VLOG(2) << "Qwen35DataProcessor: multimodal path, InputData count="
               << input_data.size();
  return input_data;
}

absl::StatusOr<Message> Qwen35DataProcessor::ToMessageImpl(
    const Responses& responses,
    const Qwen35DataProcessorArguments& args) const {
  absl::string_view response_text = responses.GetTexts()[0];
  nlohmann::ordered_json message = {{"role", "assistant"}};
  if (preface_.has_value() && std::holds_alternative<JsonPreface>(*preface_) &&
      !std::get<JsonPreface>(*preface_).tools.empty()) {
    ASSIGN_OR_RETURN(
        nlohmann::ordered_json content_and_tool_calls,
        ParseTextAndToolCalls(response_text, config_.code_fence_start,
                              config_.code_fence_end, SyntaxType::kJson,
                              config_.escape_fence_strings,
                              config_.tool_code_regex));
    if (content_and_tool_calls.contains("content")) {
      message["content"] = content_and_tool_calls["content"];
    }
    if (content_and_tool_calls.contains("tool_calls")) {
      message["tool_calls"] = content_and_tool_calls["tool_calls"];
    }
  } else {
    message["content"] = nlohmann::ordered_json::array(
        {{{"type", "text"}, {"text", std::string(response_text)}}});
  }
  return message;
}

absl::string_view Qwen35DataProcessor::CodeFenceStart() const {
  return config_.code_fence_start;
}

absl::string_view Qwen35DataProcessor::CodeFenceEnd() const {
  return config_.code_fence_end;
}

}  // namespace litert::lm
