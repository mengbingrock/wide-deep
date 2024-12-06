#ifndef KUIPER_INCLUDE_OP_RMSNORM_H_
#define KUIPER_INCLUDE_OP_RMSNORM_H_
#include "layer.h"
#include <base/cuda_config.h>
namespace op {
class RmsNormLayer : public LayerFp32Param {
 public:
  explicit RmsNormLayer(base::DeviceType device_type, int32_t dim);

  base::Status check() const override;

  base::Status forward() override;

 private:
  int32_t dim_ = 0;
};
}  // namespace op
#endif  // KUIPER_INCLUDE_OP_RMSNORM_H_