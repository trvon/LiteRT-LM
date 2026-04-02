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

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/preprocessor/image_preprocessor.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/qwen35_data_processor_config.h"
#include "runtime/engine/io_types.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/llm_model_type.pb.h"
#include "runtime/util/model_type_utils.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using json = nlohmann::ordered_json;
using ::testing::ElementsAre;

// ─── FakeImagePreprocessor ─────────────────────────────────────────────────
//
// Bypasses real image decoding so multimodal tests do not require valid
// image files. Returns a small dummy InputImage for any raw-bytes input.

class FakeImagePreprocessor : public ImagePreprocessor {
 public:
  absl::StatusOr<InputImage> Preprocess(
      const InputImage& /*input_image*/,
      const ImagePreprocessParameter& /*parameter*/) override {
    return InputImage(std::string("fake_preprocessed_bytes"));
  }
};

// Minimal 1×1 white PNG, base64-encoded.
// LoadItemData accepts {"type":"image","blob":"<base64>"} without a file path.
constexpr char kOnePxPngBase64[] =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAC0lEQVQI12NgAAIABQ"
    "AABjkB6QAAAABJRU5ErkJggg==";

// ─── Matchers ──────────────────────────────────────────────────────────────

MATCHER_P(IsInputTextContaining, substr, "") {
  if (!std::holds_alternative<InputText>(arg)) return false;
  auto text = std::get<InputText>(arg).GetRawTextString();
  if (!text.ok()) return false;
  return absl::StrContains(*text, substr);
}

MATCHER(IsInputImage, "") {
  return std::holds_alternative<InputImage>(arg);
}

// ─── Test 1: Text-only path ────────────────────────────────────────────────
//
// When messages contain no images, ToInputDataVectorImpl must return a single
// InputText with the full rendered prompt and must NOT contain any Gemma
// image token strings.

TEST(Qwen35DataProcessorTest, TextOnlyPathReturnsSingleInputText) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       Qwen35DataProcessor::Create(Qwen35DataProcessorConfig{}));

  const std::string rendered =
      "<|im_start|>user\nHello world<|im_end|>\n<|im_start|>assistant\n";
  const json messages = json::array(
      {{{"role", "user"}, {"content", "Hello world"}}});

  ASSERT_OK_AND_ASSIGN(auto input_data,
                       processor->ToInputDataVector(rendered, messages, {}));

  ASSERT_EQ(input_data.size(), 1u);
  ASSERT_TRUE(std::holds_alternative<InputText>(input_data[0]));
  auto text = std::get<InputText>(input_data[0]).GetRawTextString();
  ASSERT_OK(text);
  EXPECT_EQ(*text, rendered);

  // Must NOT contain any Gemma image tokens.
  EXPECT_FALSE(absl::StrContains(*text, "<start_of_image>"))
      << "Text-only path must not contain Gemma '<start_of_image>' token";
}

// ─── Test 2: Template token gate ──────────────────────────────────────────
//
// GetDefaultJinjaPromptTemplate for kQwen3_5 must NOT produce a template
// containing '<start_of_image>' — that is the Gemma token. The Qwen3.5
// template must use '<|vision_start|>'.
// This test exercises model_type_utils.cc:GetDefaultJinjaPromptTemplate.

TEST(Qwen35DataProcessorTest, Qwen35JinjaTemplateUsesCorrectVisionTokens) {
  proto::LlmModelType model_type;
  model_type.mutable_qwen3_5();  // set qwen3_5 oneof

  proto::PromptTemplates prompt_templates;
  prompt_templates.mutable_user()->set_prefix("<|im_start|>user\n");
  prompt_templates.mutable_user()->set_suffix("<|im_end|>\n");
  prompt_templates.mutable_model()->set_prefix("<|im_start|>assistant\n");
  prompt_templates.mutable_model()->set_suffix("<|im_end|>\n");
  prompt_templates.mutable_system()->set_prefix("<|im_start|>system\n");
  prompt_templates.mutable_system()->set_suffix("<|im_end|>\n");

  ASSERT_OK_AND_ASSIGN(auto tmpl,
                       GetDefaultJinjaPromptTemplate(prompt_templates, model_type));

  // Must not contain the Gemma image token.
  EXPECT_FALSE(absl::StrContains(tmpl, "<start_of_image>"))
      << "kQwen3_5 template must not contain Gemma '<start_of_image>'";

  // Must contain the correct Qwen3.5 vision start token.
  EXPECT_TRUE(absl::StrContains(tmpl, "<|vision_start|>"))
      << "kQwen3_5 template must contain '<|vision_start|>'";
}

// ─── Test 3: MessageToTemplateInput flattens single-text arrays ───────────
//
// A message whose content is a single-element text array should be collapsed
// to a plain string (matching Qwen3 behavior for text-only compatibility).

TEST(Qwen35DataProcessorTest, MessageToTemplateInputFlattensSingleTextArray) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       Qwen35DataProcessor::Create(Qwen35DataProcessorConfig{}));

  json message = {{"role", "user"},
                  {"content", json::array({{{"type", "text"},
                                            {"text", "hello"}}})}};

  ASSERT_OK_AND_ASSIGN(auto result,
                       processor->MessageToTemplateInput(message));
  ASSERT_TRUE(result.contains("content"));
  EXPECT_TRUE(result["content"].is_string());
  EXPECT_EQ(result["content"].get<std::string>(), "hello");
}

// ─── Test 4: ToMessage wraps response as content array ────────────────────

TEST(Qwen35DataProcessorTest, ToMessageReturnsContentArray) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       Qwen35DataProcessor::Create(Qwen35DataProcessorConfig{}));

  ASSERT_OK_AND_ASSIGN(
      auto message,
      processor->ToMessage(
          Responses(TaskState::kProcessing, {"test response"}),
          std::monostate{}));

  ASSERT_TRUE(std::holds_alternative<json>(message));
  const auto& j = std::get<json>(message);
  EXPECT_EQ(j["role"], "assistant");
  ASSERT_TRUE(j["content"].is_array());
  EXPECT_EQ(j["content"][0]["type"], "text");
  EXPECT_EQ(j["content"][0]["text"], "test response");
}

// ─── Test 5: CodeFence defaults match Qwen3 convention ────────────────────

TEST(Qwen35DataProcessorTest, CodeFenceDefaults) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       Qwen35DataProcessor::Create(Qwen35DataProcessorConfig{}));
  EXPECT_EQ(processor->CodeFenceStart(), "<tool_call>");
  EXPECT_EQ(processor->CodeFenceEnd(), "</tool_call>");
}

// ─── Test 6: Multimodal success path — InputImage before vision token text ─
//
// With a valid image blob and a FakeImagePreprocessor, confirms that the
// output vector has InputImage BEFORE the InputText segment containing
// <|image_pad|>. This is the "image-first" ordering contract: the runtime
// needs the image embedding data before the token markers it will replace.

TEST(Qwen35DataProcessorTest, MultimodalPathSucceedsImageFirst) {
  ASSERT_OK_AND_ASSIGN(
      auto processor,
      Qwen35DataProcessor::Create(Qwen35DataProcessorConfig{},
                                  /*preface=*/std::nullopt,
                                  std::make_unique<FakeImagePreprocessor>()));

  const std::string rendered =
      "<|im_start|>user\n"
      "<|vision_start|><|image_pad|><|vision_end|>"
      "Describe this image<|im_end|>\n"
      "<|im_start|>assistant\n";

  const json messages = json::array(
      {{{"role", "user"},
        {"content", json::array(
             {{{"type", "image"}, {"blob", kOnePxPngBase64}},
              {{"type", "text"}, {"text", "Describe this image"}}})}}});

  ASSERT_OK_AND_ASSIGN(auto input_data,
                       processor->ToInputDataVector(rendered, messages, {}));

  // Find the first InputImage and confirm an InputText with <|image_pad|>
  // follows it — verifying image-first ordering.
  bool found_image = false;
  bool image_pad_after_image = false;
  for (const auto& item : input_data) {
    if (std::holds_alternative<InputImage>(item)) {
      found_image = true;
    } else if (found_image && std::holds_alternative<InputText>(item)) {
      auto text = std::get<InputText>(item).GetRawTextString();
      if (text.ok() && absl::StrContains(*text, "<|image_pad|>")) {
        image_pad_after_image = true;
      }
    }
  }
  EXPECT_TRUE(found_image) << "Expected at least one InputImage in output";
  EXPECT_TRUE(image_pad_after_image)
      << "InputImage must precede the <|image_pad|> token text";

  // No Gemma image tokens anywhere in the output.
  for (const auto& item : input_data) {
    if (std::holds_alternative<InputText>(item)) {
      auto text = std::get<InputText>(item).GetRawTextString();
      if (text.ok()) {
        EXPECT_FALSE(absl::StrContains(*text, "<start_of_image>"))
            << "Multimodal path must not emit Gemma '<start_of_image>'";
      }
    }
  }
}

// ─── Test 7: Multimodal error — placeholder without image ─────────────────

TEST(Qwen35DataProcessorTest,
     MultimodalPathErrorsWhenPromptHasPlaceholderButMessagesHaveNoImage) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       Qwen35DataProcessor::Create(Qwen35DataProcessorConfig{}));

  const std::string rendered =
      "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
      "Describe this<|im_end|>\n<|im_start|>assistant\n";
  const json messages = json::array(
      {{{"role", "user"},
        {"content", json::array(
             {{{"type", "text"}, {"text", "Describe this"}}})}}});

  auto result = processor->ToInputDataVector(rendered, messages, {});
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

// ─── Test 8: Multimodal error — extra image without placeholder ───────────

TEST(Qwen35DataProcessorTest,
     MultimodalPathErrorsWhenMessagesHaveMoreImagesThanPromptPlaceholders) {
  ASSERT_OK_AND_ASSIGN(
      auto processor,
      Qwen35DataProcessor::Create(Qwen35DataProcessorConfig{},
                                  /*preface=*/std::nullopt,
                                  std::make_unique<FakeImagePreprocessor>()));

  const std::string rendered =
      "<|im_start|>user\nDescribe this image<|im_end|>\n"
      "<|im_start|>assistant\n";
  const json messages = json::array(
      {{{"role", "user"},
        {"content", json::array(
             {{{"type", "image"}, {"blob", kOnePxPngBase64}},
              {{"type", "text"}, {"text", "Describe this image"}}})}}});

  auto result = processor->ToInputDataVector(rendered, messages, {});
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace litert::lm
