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

#include "runtime/util/model_type_utils.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/prompt_template.h"
#include "runtime/components/tokenizer.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/llm_model_type.pb.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::NiceMock;
using ::testing::Return;

class MockTokenizer : public Tokenizer {
 public:
  MOCK_METHOD(absl::StatusOr<std::string>, TokenIdsToText,
              (const std::vector<int>& token_ids), (override));
  MOCK_METHOD(absl::StatusOr<std::vector<int>>, TextToTokenIds,
              (absl::string_view text), (override));
  MOCK_METHOD(absl::StatusOr<int>, TokenToId, (absl::string_view token),
              (override));
  MOCK_METHOD(TokenizerType, GetTokenizerType, (), (const, override));
  MOCK_METHOD(std::vector<std::string>, GetTokens, (), (const, override));
};

TEST(ModelTypeUtilsTest, InferLlmModelTypeGemma3N) {
  NiceMock<MockTokenizer> tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText)
      .Times(testing::AnyNumber())
      .WillRepeatedly(Return("<start_of_turn>"));
  EXPECT_CALL(tokenizer, TextToTokenIds("<start_of_audio>"))
      .WillRepeatedly(Return(std::vector<int>{256000}));
  ASSERT_OK_AND_ASSIGN(auto model_type,
                       InferLlmModelType(proto::LlmMetadata(), &tokenizer));
  EXPECT_THAT(model_type.has_gemma3n(), true);
}

TEST(ModelTypeUtilsTest, InferLlmModelTypeGemma3NWrongAudioToken) {
  NiceMock<MockTokenizer> tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText)
      .Times(testing::AnyNumber())
      .WillRepeatedly(Return("<start_of_turn>"));
  EXPECT_CALL(tokenizer, TextToTokenIds("<start_of_audio>"))
      .WillRepeatedly(Return(
          // The encoded ids for "<start_of_audio>" in the Gemma3 1B tokenizer.
          std::vector<int>{256001}));
  ASSERT_OK_AND_ASSIGN(auto model_type,
                       InferLlmModelType(proto::LlmMetadata(), &tokenizer));
  EXPECT_THAT(model_type.has_gemma3n(), false);
}

TEST(ModelTypeUtilsTest, InferLlmModelTypeGemma3) {
  NiceMock<MockTokenizer> tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText)
      .Times(testing::AnyNumber())
      .WillRepeatedly(Return("<start_of_turn>"));
  EXPECT_CALL(tokenizer, TextToTokenIds("<start_of_audio>"))
      .WillRepeatedly(Return(
          // The encoded ids for "<start_of_audio>" in the Gemma3 1B tokenizer.
          std::vector<int>{236820, 3041, 236779, 1340, 236779, 20156, 236813}));
  ASSERT_OK_AND_ASSIGN(auto model_type,
                       InferLlmModelType(proto::LlmMetadata(), &tokenizer));
  EXPECT_THAT(model_type.has_gemma3(), true);
}

TEST(ModelTypeUtilsTest, InferLlmModelTypeGenericModel) {
  NiceMock<MockTokenizer> tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("Hello"));
  EXPECT_CALL(tokenizer, TextToTokenIds("<start_of_audio>"))
      .WillRepeatedly(Return(std::vector<int>{256000}));
  ASSERT_OK_AND_ASSIGN(auto model_type,
                       InferLlmModelType(proto::LlmMetadata(), &tokenizer));
  EXPECT_THAT(model_type.has_generic_model(), true);
}

TEST(ModelTypeUtilsTest, GetDefaultJinjaPromptTemplate) {
  proto::PromptTemplates prompt_templates;
  prompt_templates.mutable_user()->set_prefix("<start_of_turn>user\n");
  prompt_templates.mutable_model()->set_prefix("<start_of_turn>model\n");
  prompt_templates.mutable_system()->set_prefix("<start_of_turn>system\n");
  prompt_templates.mutable_user()->set_suffix("<end_of_turn>\n");
  prompt_templates.mutable_model()->set_suffix("<end_of_turn>\n");
  prompt_templates.mutable_system()->set_suffix("<end_of_turn>\n");
  proto::LlmModelType llm_model_type;
  llm_model_type.mutable_generic_model();
  ASSERT_OK_AND_ASSIGN(
      auto jinja_prompt_template,
      GetDefaultJinjaPromptTemplate(prompt_templates, llm_model_type));
  PromptTemplate prompt_template(jinja_prompt_template);
  PromptTemplateInput prompt_template_input;
  prompt_template_input.messages = {
      {{"role", "system"}, {"content", "This is a system message"}},
      {{"role", "user"}, {"content", "This is a user message"}},
      {{"role", "model"}, {"content", "This is a model message"}}};
  ASSERT_OK_AND_ASSIGN(auto rendered_prompt,
                       prompt_template.Apply(prompt_template_input));
  EXPECT_EQ(rendered_prompt,
            "<start_of_turn>system\nThis is a system message<end_of_turn>\n"
            "<start_of_turn>user\nThis is a user message<end_of_turn>\n"
            "<start_of_turn>model\nThis is a model message<end_of_turn>\n"
            "<start_of_turn>model\n");
}

TEST(ModelTypeUtilsTest, GetDefaultJinjaPromptTemplateEmpty) {
  proto::PromptTemplates prompt_templates;
  proto::LlmModelType llm_model_type;
  llm_model_type.mutable_generic_model();
  ASSERT_OK_AND_ASSIGN(
      auto jinja_prompt_template,
      GetDefaultJinjaPromptTemplate(prompt_templates, llm_model_type));
  PromptTemplate prompt_template(jinja_prompt_template);
  PromptTemplateInput prompt_template_input;
  prompt_template_input.messages = {
      {{"role", "system"}, {"content", "This is a system message"}},
      {{"role", "user"}, {"content", "This is a user message"}},
      {{"role", "model"}, {"content", "This is a model message"}}};
  ASSERT_OK_AND_ASSIGN(auto rendered_prompt,
                       prompt_template.Apply(prompt_template_input));
  EXPECT_EQ(rendered_prompt,
            "This is a system message"
            "This is a user message"
            "This is a model message");
}

TEST(ModelTypeUtilsTest, GetDefaultJinjaPromptTemplateWithImageAndAudio) {
  proto::PromptTemplates prompt_templates;
  prompt_templates.mutable_user()->set_prefix("<start_of_turn>user\n");
  prompt_templates.mutable_model()->set_prefix("<start_of_turn>model\n");
  prompt_templates.mutable_system()->set_prefix("<start_of_turn>system\n");
  prompt_templates.mutable_user()->set_suffix("<end_of_turn>\n");
  prompt_templates.mutable_model()->set_suffix("<end_of_turn>\n");
  prompt_templates.mutable_system()->set_suffix("<end_of_turn>\n");
  proto::LlmModelType llm_model_type;
  llm_model_type.mutable_generic_model();
  ASSERT_OK_AND_ASSIGN(
      auto jinja_prompt_template,
      GetDefaultJinjaPromptTemplate(prompt_templates, llm_model_type));
  PromptTemplate prompt_template(jinja_prompt_template);
  PromptTemplateInput prompt_template_input;
  prompt_template_input.messages = {
      {{"role", "system"}, {"content", "This is a system message"}},
      {
          {"role", "user"},
          {"content",
           {
               {{"type", "text"}, {"text", "Here is a user image "}},
               {{"type", "image"}, {"image", "image_bytes"}},
               {{"type", "text"}, {"text", " Here is a user audio "}},
               {{"type", "audio"}, {"audio", "audio_bytes"}},
           }},
      },
      {{"role", "model"}, {"content", "This is a model message"}}};
  ASSERT_OK_AND_ASSIGN(auto rendered_prompt,
                       prompt_template.Apply(prompt_template_input));
  EXPECT_EQ(rendered_prompt,
            "<start_of_turn>system\nThis is a system message<end_of_turn>\n"
            "<start_of_turn>user\nHere is a user image <start_of_image> Here "
            "is a user audio <start_of_audio><end_of_turn>\n"
            "<start_of_turn>model\nThis is a model message<end_of_turn>\n"
            "<start_of_turn>model\n");
}

// ─── Qwen3.5 template token gate ──────────────────────────────────────────
//
// GetDefaultJinjaPromptTemplate for kQwen35 must NOT produce a template
// containing '<start_of_image>' (the Gemma token) and MUST contain
// '<|vision_start|>' (the Qwen3.5 token). Also confirms no bleed into kQwen3.

TEST(ModelTypeUtilsTest, GetDefaultJinjaPromptTemplateQwen35) {
  proto::PromptTemplates prompt_templates;
  prompt_templates.mutable_user()->set_prefix("<|im_start|>user\n");
  prompt_templates.mutable_user()->set_suffix("<|im_end|>\n");
  prompt_templates.mutable_model()->set_prefix("<|im_start|>assistant\n");
  prompt_templates.mutable_model()->set_suffix("<|im_end|>\n");
  prompt_templates.mutable_system()->set_prefix("<|im_start|>system\n");
  prompt_templates.mutable_system()->set_suffix("<|im_end|>\n");

  proto::LlmModelType qwen35_type;
  qwen35_type.mutable_qwen3_5();
  ASSERT_OK_AND_ASSIGN(
      auto qwen35_tmpl,
      GetDefaultJinjaPromptTemplate(prompt_templates, qwen35_type));

  EXPECT_FALSE(absl::StrContains(qwen35_tmpl, "<start_of_image>"))
      << "kQwen35 template must not contain Gemma '<start_of_image>'";
  EXPECT_TRUE(absl::StrContains(qwen35_tmpl, "<|vision_start|>"))
      << "kQwen35 template must contain '<|vision_start|>'";

  // kQwen3 must not bleed Qwen3.5 vision tokens.
  proto::LlmModelType qwen3_type;
  qwen3_type.mutable_qwen3();
  ASSERT_OK_AND_ASSIGN(
      auto qwen3_tmpl,
      GetDefaultJinjaPromptTemplate(prompt_templates, qwen3_type));
  EXPECT_FALSE(absl::StrContains(qwen3_tmpl, "<|vision_start|>"))
      << "kQwen3 template must not contain Qwen3.5 '<|vision_start|>'";
}

}  // namespace
}  // namespace litert::lm
