# This is the pre-production document for the Inferencing Module

## 09-05-2026 06:50 :-
- Inferencing is based on `llama.cpp` and is a submodule in `/third_party` [Credit Given in `README.md`].
- Refer `inf.cpp` for details.
- Basic Chatting with the LLM Model works as of now. 
- Did not integrate the `mmu` into the chat yet. 

## 25-05-2026 - :-
- Added GPU Inferencing (via Vulkan) from `llama.cpp`.
- Nothing much else was done.

## 01-06-2026 - :-
- Migrated to SYCL-based inferencing such that now its hybrid CPU+iGP inferencing. 
- Precision will be FP16 since Xe2's XMX Matrix engines are optimised for FP16s the best.