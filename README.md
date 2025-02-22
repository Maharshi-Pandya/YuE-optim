# YuE-exllamav2

Optimized implementation of [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE) in exllamav2.

## Benchmark
### Inference on 1x H100

- bf16 model, Flash Attention 2 enabled, CFG enabled
- updated genres.txt and lyrics.txt
- 4 segments in length
- default batch size 16 for stage 2 and cache size for both stages is set to 32k

#### ExLlamav2 implementation

Total time: **7.49 minutes**

```
>> Creating pipeline for stage 1 using exl2=True ...
Stage 1 Pipeline creation execution time: 5882.6748 ms

>> Creating pipeline for stage 2 using exl2=True ...
Stage 2 pipeline preparation execution time: 1632.8529 ms

>> Generating stage 1...
100%|█████████████████████████████████████████████████████████████████████| 2048/2048 [00:51<00:00, 39.55it/s]
100%|█████████████████████████████████████████████████████████████████████| 2048/2048 [00:53<00:00, 38.19it/s]
100%|█████████████████████████████████████████████████████████████████████| 2048/2048 [01:38<00:00, 20.76it/s]
100%|█████████████████████████████████████████████████████████████████████| 2048/2048 [01:17<00:00, 26.60it/s]
100%|███████████████████████████████████████████████████████████████████████████| 4/4 [04:41<00:00, 70.44s/it]
Stage 1 execution time: 281836.0000 ms

>> Preprocessing for stage 2...
Stage 2 pre-processing time: 2.3964 ms
>> Generating stage 2...
2it [00:00, 3659.95it/s]
100%|███████████████████████████████████████████████████████████████████████| 300/300 [00:41<00:00,  7.24it/s]
100%|███████████████████████████████████████████████████████████████████████| 300/300 [00:41<00:00,  7.15it/s]
100%|███████████████████████████████████████████████████████████████████████| 300/300 [00:41<00:00,  7.26it/s]
100%|███████████████████████████████████████████████████████████████████████| 196/196 [00:26<00:00,  7.28it/s]
100%|███████████████████████████████████████████████████████████████████████████| 4/4 [02:31<00:00, 37.99s/it]
Stage 2 execution time: 152191.8125 ms

>> Postprocessing final...
Processing 8 samples
Compressed shape: (8, 4096)
/root/YuE-optim/src/yue/vocoder.py:45: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requiresgrad(True), rather than torch.tensor(sourceTensor).
  compressed = torch.tensor(compressed).to(device)
Decoded in 0.07s (1161.54x RTF)
Saved: ./output/vocoder/stems/vtrack.mp3
Processing 8 samples
Compressed shape: (8, 4096)
Decoded in 0.04s (2255.51x RTF)
Saved: ./output/vocoder/stems/itrack.mp3
Created mix: ./output/vocoder/mix/mixed.mp3
Successfully created 'mixed.mp3' with matched low-frequency energy.
Final post-processing time: 7957.9150 ms

>> Done.
```


#### HF transformers implementation

Total time: **14.17 minutes**

```
>> Creating pipeline for stage 1 using exl2=False ...
Loading checkpoint shards: 100%|████████████████████████████████████████████████| 3/3 [00:16<00:00,  5.39s/it]
Stage 1 Pipeline creation execution time: 16393.1289 ms

>> Creating pipeline for stage 2 using exl2=False ...
Stage 2 pipeline preparation execution time: 4530.2231 ms

>> Generating stage 1...
  0%|                                                                                   | 0/4 [00:00<?, ?it/s]The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's attention_mask to obtain reliable results.
100%|██████████████████████████████████████████████████████████████████████████| 4/4 [10:02<00:00, 150.70s/it]
Stage 1 execution time: 602793.3125 ms

>> Preprocessing for stage 2...
Stage 2 pre-processing time: 2.4055 ms
>> Generating stage 2...
100%|██████████████████████████████████████████████████████████████████████████| 2/2 [03:38<00:00, 109.10s/it]
Stage 2 execution time: 218191.3594 ms

>> Postprocessing final...
Processing 8 samples
Compressed shape: (8, 4096)
/root/YuE-optim/src/yue/vocoder.py:45: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requiresgrad(True), rather than torch.tensor(sourceTensor).
  compressed = torch.tensor(compressed).to(device)
Decoded in 0.05s (1627.00x RTF)
Saved: ./output/vocoder/stems/vtrack.mp3
Processing 8 samples
Compressed shape: (8, 4096)
Decoded in 0.04s (2263.67x RTF)
Saved: ./output/vocoder/stems/itrack.mp3
Created mix: ./output/vocoder/mix/mixed.mp3
Successfully created 'mixed.mp3' with matched low-frequency energy.
Final post-processing time: 8406.6221 ms

>> Done.
```
