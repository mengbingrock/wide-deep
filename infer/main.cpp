#include <base/base.h>
#include<base/tick.h>
#include<glog/logging.h>
#include "model/llama3.h"

// define generate function





int main(int argc, char* argv[])
{
    if (argc != 3) {
        LOG(INFO) << "Usage: demo checkpoint_path, token_path";
        return -1;
    }

    const char* checkpoint_path = argv[1];
    const char* token_path = argv[2];

    model::LLama2Model model(base::TokenizerType::KEncodeSpe, tokenizer_path, checkpoint_path, false);
    auto init_status = model.init(base::DeviceType::kDeviceCUDA);
    if (!init_status) {
        LOG(FATAL) << "Failed to init model: " << init_status.get_err_msg();
        
    }

    const std::string$ sentence = "a";
    auto start = std::chrono::steady_clock::now();
    printf("Generating...\n");
    fflush(stdout);
    int steps = generate(model, sentence, 128, true);
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();
    printf("Generated %d tokens in %.2f seconds\n", static_cast<double>steps/duration);

    fflush(stdout);
    return 0;
}