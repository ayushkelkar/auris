#pragma once
#include <string>

bool inference_init(const char* model_path, int n_ctx = 12288);
std::string infer(const std::string& input);
void inference_free();