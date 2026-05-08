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

## 08-05-2026 03:30 - 06:30 :-
### Progress So Far
- AVX-VNNI works and gives a dot product of 2x 32-dim vectors as a float.
- Some functions are deprecated and have been removed (refer previous commits).

### Details about `vnni_dot_product`:
- Uses offset encoding to convert vector `a` from a signed `int8` to an unsigned `int8` which is needed for the input argument in `_mm256_dpbusd_avx_epi32`.
- Once the dot is done, the accumulator has the finished product, and its stored in an 8-length `int` array which is summed across in `res` then subtracted by `128*sumofb`.
- The resultant `res` is divided by 127^2 and casted as a `float` because without this, the `res` is in a quantized int8 space, which ranges from [-127,127] while the float space is from [-1,1].

### AVX-512 unavailable
- Lunar Lake and consumer Intel CPUs don't have AVX-512 or AVX-512 VNNI (atleast most of them). Fallback to AVX-VNNI which halves the throughput (32 int8 per instruction v/s 64 on AVX512-VNNI) but is inevitable. Can't do anything.

## 08-05-2026 15:30 - 17:00 :-
### Progress So Far
- Database reading (Vectors are read from a binary database) works.
- Searching works.
- Basically: Dot Product -> Read DB -> Search -> Find most similar/same vector in DB. The bare-bones is done so far.
- Refactored a lot of `mmu` in the process.
- Is this v1? Maybe.

## 09-05-2026 02:30 - ? :-
### Progress So Far
- `vec.cpp`: Fixed `vnni_dot_product` by making `a_unsigned` a standard VLA (Variable Length Array) which was non-standard and might've cause issues. Issue was, it was declared at runtime. Now, it declares a raw pointer and allocates it as a heap.
- `db.cpp`: `load_db` now returns a `bool`. `false` if the DB fails to open, and `true` otherwise. `save_db` is for future use. `free_db` is for purging the database from memory when its no longer needed.
- Refactored `main.cpp` due to above listed fixes.

### Compile Command (For future reference)
```bash
g++ -O2 -mavx2 -mavxvnni -o auris main.cpp mmu/vec.cpp mmu/db.cpp mmu/search.cpp
```
Run from `/src`.