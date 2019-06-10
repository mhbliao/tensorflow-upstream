/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/service/gpu/empty_rewriter.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"

namespace xla {
namespace gpu {
namespace {

class Visitor : public DfsHloVisitorWithDefault {
 public:
  explicit Visitor(HloComputation* computation) : computation_(computation) {}

  static bool Run(HloComputation* computation) {
    Visitor visitor(computation);
    TF_CHECK_OK(computation->Accept(&visitor));
    return visitor.changed_;
  }

  Status DefaultAction(HloInstruction* /*hlo_instruction*/) override {
    return Status::OK();
  }

  Status HandleReduce(HloInstruction* reduce) override;

 private:
  bool changed_ = false;
  HloComputation* computation_;
};

Status Visitor::HandleReduce(HloInstruction* reduce) {
  std::vector<HloInstruction*> operands;
  operands.push_back(reduce);
  HloInstruction* new_root =
      computation_->AddInstruction(HloInstruction::CreateCustomCall(
          reduce->shape(), operands, kEmptyCallTarget));
  computation_->set_root_instruction(new_root);
  changed_ = true;
  return Status::OK();
}

}  // anonymous namespace

StatusOr<bool> EmptyRewriter::Run(HloModule* module) {
  VLOG(2) << "EmptyRewriter::Run(), before:";
  XLA_VLOG_LINES(2, module->ToString());

  bool changed = false;
  for (auto* comp : module->MakeNonfusionComputations()) {
    if (Visitor::Run(comp)) {
      changed = true;
    }
  }

  VLOG(2) << "EmptyRewriter::Run(), after:";
  XLA_VLOG_LINES(2, module->ToString());
  return changed;
}

}  // namespace gpu
}  // namespace xla
