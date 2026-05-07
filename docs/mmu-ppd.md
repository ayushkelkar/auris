# This is the pre-production document for the MMU Module (Memory Management Unit)

## 27-04-2026 16:48:-
Note: Researched sqlite-vec, its a vector database engine made for small/consumer or non enterprise applications like this, however its kind of shit
because its effectively in alpha/beta, so the code will change at any given point in time, and issue is I can't find a C/C++ guide or docs for it.

Tl;dr: Will develop own vector database engine, probably within MMU itself. So it'll be a component of the MMU.

## 07-05-2026 02:00 - 05:00 :-
### Architecture Decisions
- **Algorithm**: DiskANN/Vamana (from papers. DiskANN is MIT-licensed, so will have to credit that later)
  - AVX-512 is kind of a must for this. So will need AVX-512 (My 228V does it fine. i5-12400 not so much T_T)
  - AVX-512 VNNI on Lion Cove P-cores (beam search, distance compute)
  - AVX2 VNNI on Skymont E-cores (I/O, background ops)
  #### Why DiskANN?
    - Every other index is not disk-first (FAISS, ScaNN, etc.)
    - Graph-based (Vamana)
    - Published papers so I can code it myself (Microsoft's DiskANN has fluff I don't need)
    - Its already implemented in Bing, so its already proven.
- **Storage**: Disk-first
- **Similarity metric**: Cosine similarity → collapses to dot product after normalization
- So far, made an n-dim `vec.cpp` which finds magnitude, cosine (direction), normalize, of a vector or two.
- Catastrophic Forgetting is a newfound problem. Discovered it today.

### Catastrophic Forgetting
When a neural network is fine-tuned on new data, it updates its weights
to fit the new data better. In doing so, it can overwrite knowledge
encoded in those same weights from previous training. This is
catastrophic forgetting.

Its also effectively Probabilistic.

#### Solution
Training 1 → D1  
Training 2 → D1 + D2  
Training 3 → D1 + D2 + D3  
Training N → D1 + D2 + ... + DN
> Each training includes the previous training as well.
>> **Important**: Revisit only if scale becomes an issue, shouldn't be a problem for the time being.