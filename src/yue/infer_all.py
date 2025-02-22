import os
import random
import re
import math
import copy

from typing import Tuple, List
from dataclasses import dataclass
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf

from codecmanipulator import CodecManipulator
from common import BlockTokenRangeProcessor, parser, seed_everything, get_cache_class
from einops import rearrange
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2Sampler
from mmtokenizer import _MMSentencePieceTokenizer
from models.soundstream_hubert_new import SoundStream
from omegaconf import OmegaConf
from torchaudio.transforms import Resample
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LogitsProcessorList
from transformers.cache_utils import StaticCache
from post_process_audio import replace_low_freq_with_energy_matched
from vocoder import build_codec_model, process_audio
from vocos import VocosDecoder

from .utils import empty_gpu_cache


@dataclass
class SampleSettings:
    # Here is suggested decoding config
    top_p = 0.93
    temperature = 1
    repetition_penalty = 1.1
    guidance_scale_seg0 = 1.5  # None to disable cfg
    guidance_scale = 1.2  # None to disable cfg

    def __init__(self, use_guidance: bool = True, repetition_penalty: float = 1.1):
        if not use_guidance:
            self.guidance_scale_seg0 = None
            self.guidance_scale = None
        self.repetition_penalty = repetition_penalty


def load_audio_mono(filepath, sampling_rate=16000):
    audio, sr = torchaudio.load(filepath)
    # Convert to mono
    audio = torch.mean(audio, dim=0, keepdim=True)
    # Resample if needed
    if sr != sampling_rate:
        resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
        audio = resampler(audio)
    return audio


def encode_audio(codec_model, audio_prompt, device, target_bw=0.5):
    if len(audio_prompt.shape) < 3:
        audio_prompt.unsqueeze_(0)
    with torch.no_grad():
        raw_codes = codec_model.encode(audio_prompt.to(device), target_bw=target_bw)
    raw_codes = raw_codes.transpose(0, 1)
    raw_codes = raw_codes.cpu().numpy().astype(np.int16)
    return raw_codes


# =========================================
#           STAGE 1 PIPELINE
# =========================================

class Stage1Pipeline:
    def __init__(self, device: torch.device, basic_model_config: str, resume_path: str, mmtokenizer: _MMSentencePieceTokenizer = None, codec_model: SoundStream = None):
        self.device = device
        self.codec_tool = CodecManipulator("xcodec", 0, 1)
        self.basic_model_config = basic_model_config
        self.resume_path = resume_path

        # Load tokenizer
        self.mmtokenizer = mmtokenizer or _MMSentencePieceTokenizer(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mm_tokenizer_v0.2_hf", "tokenizer.model"))
        self.codec_model = codec_model or None
        self.start_of_segment = self.mmtokenizer.tokenize("[start_of_segment]")
        self.end_of_segment = self.mmtokenizer.tokenize("[end_of_segment]")

        self.is_cuda = torch.cuda.is_available()

    def load_codec_model(self):
        if self.codec_model is not None:
            return
        model_config = OmegaConf.load(self.basic_model_config)
        assert model_config.generator.name == "SoundStream"
        self.codec_model = SoundStream(**model_config.generator.config).to(self.device)
        parameter_dict = torch.load(self.resume_path, map_location=self.device, weights_only=False)
        self.codec_model.load_state_dict(parameter_dict["codec_model"])
        self.codec_model.eval()
        empty_gpu_cache(self.is_cuda)

    def get_prompt_texts(self, genres: str, lyrics: str):
        def split_lyrics(lyrics):
            pattern = r"\[(\w+)\](.*?)(?=\[|\Z)"
            segments = re.findall(pattern, lyrics, re.DOTALL)
            structured_lyrics = [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]
            return structured_lyrics

        lyrics = split_lyrics(lyrics)
        full_lyrics = "\n".join(lyrics)
        prompt_texts = [f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"]
        prompt_texts += lyrics
        return lyrics, prompt_texts

    def get_audio_prompt_ids(
        self,
        use_dual_tracks_prompt: bool,
        vocal_track_prompt_path: str,
        instrumental_track_prompt_path: str,
        use_audio_prompt: bool,
        audio_prompt_path: str,
        prompt_start_time: int,
        prompt_end_time: int,
    ):
        self.load_codec_model()
        if use_dual_tracks_prompt:
            vocals_ids = load_audio_mono(vocal_track_prompt_path)
            instrumental_ids = load_audio_mono(instrumental_track_prompt_path)
            vocals_ids = encode_audio(self.codec_model, vocals_ids, self.device, target_bw=0.5)
            instrumental_ids = encode_audio(self.codec_model, instrumental_ids, self.device, target_bw=0.5)
            vocals_ids = self.codec_tool.npy2ids(vocals_ids[0])
            instrumental_ids = self.codec_tool.npy2ids(instrumental_ids[0])
            ids_segment_interleaved = rearrange([np.array(vocals_ids), np.array(instrumental_ids)], "b n -> (n b)")
            audio_prompt_codec = ids_segment_interleaved[int(prompt_start_time * 50 * 2) : int(prompt_end_time * 50 * 2)]
            audio_prompt_codec = audio_prompt_codec.tolist()
        elif use_audio_prompt:
            audio_prompt = load_audio_mono(audio_prompt_path)
            raw_codes = encode_audio(self.codec_model, audio_prompt, self.device, target_bw=0.5)
            # Format audio prompt
            code_ids = self.codec_tool.npy2ids(raw_codes[0])
            audio_prompt_codec = code_ids[int(prompt_start_time * 50) : int(prompt_end_time * 50)]  # 50 is tps of xcodec
        audio_prompt_codec_ids = [self.mmtokenizer.soa] + self.codec_tool.sep_ids + audio_prompt_codec + [self.mmtokenizer.eoa]
        sentence_ids = self.mmtokenizer.tokenize("[start_of_reference]") + audio_prompt_codec_ids + self.mmtokenizer.tokenize("[end_of_reference]")
        return sentence_ids

    def get_first_segment_prompt(
        self,
        segment_p: str,
        prompt_text_0: str,
        use_dual_tracks_prompt: bool,
        vocal_track_prompt_path: str,
        instrumental_track_prompt_path: str,
        use_audio_prompt: bool,
        audio_prompt_path: str,
        prompt_start_time: int,
        prompt_end_time: int,
    ):
        section_text = segment_p.replace("[start_of_segment]", "").replace("[end_of_segment]", "")
        head_id = self.mmtokenizer.tokenize(prompt_text_0)
        if use_dual_tracks_prompt or use_audio_prompt:
            head_id += self.get_audio_prompt_ids(
                use_dual_tracks_prompt,
                vocal_track_prompt_path,
                instrumental_track_prompt_path,
                use_audio_prompt,
                audio_prompt_path,
                prompt_start_time,
                prompt_end_time,
            )
        return head_id + self.start_of_segment + self.mmtokenizer.tokenize(section_text) + [self.mmtokenizer.soa] + self.codec_tool.sep_ids

    def get_segment_prompt(self, segment_p: str):
        section_text = segment_p.replace("[start_of_segment]", "").replace("[end_of_segment]", "")
        return self.end_of_segment + self.start_of_segment + self.mmtokenizer.tokenize(section_text) + [self.mmtokenizer.soa] + self.codec_tool.sep_ids

    def post_process_for_next_stage(self, raw_output: torch.Tensor, use_audio_prompt: bool, use_dual_tracks_prompt: bool) -> Tuple[np.ndarray, np.ndarray]:
        # save raw output and check sanity
        ids = raw_output[0].cpu().numpy()
        soa_idx = np.where(ids == self.mmtokenizer.soa)[0].tolist()
        eoa_idx = np.where(ids == self.mmtokenizer.eoa)[0].tolist()
        if len(soa_idx) != len(eoa_idx):
            raise ValueError(f"invalid pairs of soa and eoa, Num of soa: {len(soa_idx)}, Num of eoa: {len(eoa_idx)}")

        vocals = []
        instrumentals = []
        range_begin = 1 if use_audio_prompt or use_dual_tracks_prompt else 0
        for i in range(range_begin, len(soa_idx)):
            codec_ids = ids[soa_idx[i] + 1: eoa_idx[i]]
            if codec_ids[0] == 32016:
                codec_ids = codec_ids[1:]
            codec_ids = codec_ids[: 2 * (codec_ids.shape[0] // 2)]
            vocals_ids = self.codec_tool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[0])
            vocals.append(vocals_ids)
            instrumentals_ids = self.codec_tool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[1])
            instrumentals.append(instrumentals_ids)
        vocals = np.concatenate(vocals, axis=1)
        instrumentals = np.concatenate(instrumentals, axis=1)

        return vocals, instrumentals

    def save(self, raw_output: torch.Tensor, output_dir: str, use_audio_prompt: bool, use_dual_tracks_prompt: bool):
        vocals, instrumentals = self.post_process_for_next_stage(raw_output, use_audio_prompt, use_dual_tracks_prompt)

        stage1_output_dir = os.path.join(output_dir, "stage1")
        os.makedirs(stage1_output_dir, exist_ok=True)
        vocal_save_path = os.path.join(stage1_output_dir, "vtrack.npy")
        inst_save_path = os.path.join(stage1_output_dir, "itrack.npy")
        np.save(vocal_save_path, vocals)
        np.save(inst_save_path, instrumentals)

    def shorten_input(self, seq: torch.Tensor, max_context: int):
        # Iteratively drop the oldest segment in the context until the sequence fits in context
        pattern = torch.tensor(self.start_of_segment)
        pattern_length = pattern.numel()
        while seq.shape[-1] > max_context:
            windows = seq[0].unfold(0, pattern_length, 1)
            matches = (windows == pattern).all(dim=1)
            match_indices = torch.nonzero(matches).flatten()
            if match_indices.numel() < 3:
                # Ensure that at least one other segment remains before the current segment for continuity
                print("Unable to keep enough segments for smart context, falling back to simple truncation. " f"Now using the last {max_context} tokens.")
                return seq[:, -max_context:]
            first_segment_start = match_indices[0].item()
            second_segment_start = match_indices[1].item()
            seq = torch.cat((seq[:, :first_segment_start], seq[:, second_segment_start:]), dim=-1)
        return seq


class Stage1Pipeline_HF(Stage1Pipeline):
    def __init__(self, model_path: str, device: torch.device, cache_size: int, **kwargs):
        super().__init__(device, **kwargs)

        # Load HF model
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map=self.device)
        self.model.eval()
        # if torch.__version__ >= "2.0.0":
        #     self.model = torch.compile(self.model)
        empty_gpu_cache(self.is_cuda)
        self.cache_size = cache_size

    def generate(
        self,
        use_dual_tracks_prompt: bool,
        vocal_track_prompt_path: str,
        instrumental_track_prompt_path: str,
        use_audio_prompt: bool,
        audio_prompt_path: str,
        genres: str,
        lyrics: str,
        run_n_segments: int,
        max_new_tokens: int,
        prompt_start_time: int,
        prompt_end_time: int,
        sample_settings: SampleSettings,
    ) -> torch.Tensor:

        lyrics, prompt_texts = self.get_prompt_texts(genres, lyrics)
        run_n_segments = min(run_n_segments, len(lyrics))

        for i in tqdm(range(run_n_segments)):

            # Get prompt
            if i == 0:
                prompt_ids = self.get_first_segment_prompt(
                    prompt_texts[1],
                    prompt_texts[0],
                    use_dual_tracks_prompt,
                    vocal_track_prompt_path,
                    instrumental_track_prompt_path,
                    use_audio_prompt,
                    audio_prompt_path,
                    prompt_start_time,
                    prompt_end_time,
                )
            else:
                prompt_ids = self.get_segment_prompt(prompt_texts[i + 1])
            prompt_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(self.device)
            input_ids = torch.cat([raw_output, prompt_ids], dim=1) if i > 0 else prompt_ids

            # Use window slicing in case output sequence exceeds the context of model
            max_context = self.cache_size - max_new_tokens - 1
            if input_ids.shape[-1] > max_context:
                print(f"Section {i}: output length {input_ids.shape[-1]} exceeding context length {max_context}, " f"dropping early segment(s) from prompt.")
                input_ids = self.shorten_input(input_ids, max_context)

            past_key_values = StaticCache(
                self.model.config, max_batch_size=1, max_cache_len=input_ids.shape[-1] + max_new_tokens, device=self.model.device, dtype=self.model.dtype
            )

            processors = LogitsProcessorList([BlockTokenRangeProcessor(0, 32002), BlockTokenRangeProcessor(32016, 32016)])

            output_seq = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                min_new_tokens=100,
                do_sample=True,
                top_p=sample_settings.top_p,
                temperature=sample_settings.temperature,
                repetition_penalty=sample_settings.repetition_penalty,
                eos_token_id=self.mmtokenizer.eoa,
                pad_token_id=self.mmtokenizer.eoa,
                logits_processor=processors,
                guidance_scale=sample_settings.guidance_scale_seg0 if i == 0 else sample_settings.guidance_scale,
                past_key_values=past_key_values,
            )

            if output_seq[0][-1].item() != self.mmtokenizer.eoa:
                tensor_eoa = torch.tensor([[self.mmtokenizer.eoa]], dtype=torch.long, device=output_seq.device)
                output_seq = torch.cat((output_seq, tensor_eoa), dim=1)
            if i > 0:
                raw_output = torch.cat([raw_output, prompt_ids, output_seq[:, input_ids.shape[-1] :]], dim=1)
            else:
                raw_output = output_seq

        empty_gpu_cache(self.is_cuda)
        return raw_output


class Stage1Pipeline_EXL2(Stage1Pipeline):
    def __init__(self, model_path: str, device: torch.device, cache_size: int, cache_mode: str, **kwargs):
        super().__init__(device, **kwargs)

        assert device != "cpu", "ExLlamaV2 does not support CPU inference."

        # Load EXL2 model
        device_idx = self.device.index
        gpu_split = [0] * torch.cuda.device_count()
        gpu_split[device_idx] = 9999
        exl2_config = ExLlamaV2Config(model_path)
        exl2_config.no_sdpa = True  # TODO: Figure out why SDPA slows to a crawl when given custom attn mask
        self.model = ExLlamaV2(exl2_config)
        self.model.load(gpu_split)

        # Load tokenizer (only needed for vocab size in disallow_tokens)
        self.tokenizer = ExLlamaV2Tokenizer(exl2_config)

        # Define cache
        self.cache_size = cache_size
        self.cache_mode = get_cache_class(cache_mode)

        # TODO: Output layer could be trimmed here to avoid masking out the first 32k tokens during generation

    def generate(
        self,
        use_dual_tracks_prompt: bool,
        vocal_track_prompt_path: str,
        instrumental_track_prompt_path: str,
        use_audio_prompt: bool,
        audio_prompt_path: str,
        genres: str,
        lyrics: str,
        run_n_segments: int,
        max_new_tokens: int,
        prompt_start_time: int,
        prompt_end_time: int,
        sample_settings: SampleSettings,
    ) -> torch.Tensor:

        if sample_settings.guidance_scale_seg0 is None:
            bsz = 1
            cfg = False
            position_offsets = None
            input_mask = None
        else:
            bsz = 2
            cfg = True

        lyrics, prompt_texts = self.get_prompt_texts(genres, lyrics)
        run_n_segments = min(run_n_segments, len(lyrics))

        # Cache for the whole output sequence
        cache = self.cache_mode(self.model, batch_size=bsz, max_seq_len=self.cache_size)

        # Collect output here
        seq = torch.empty((bsz, 0), dtype=torch.long)

        # Sample settings
        gen_settings = ExLlamaV2Sampler.Settings(
            top_k=0, top_p=sample_settings.top_p, token_repetition_penalty=sample_settings.repetition_penalty, temperature=sample_settings.temperature
        )
        gen_settings.allow_tokens(self.tokenizer, [32002] + list(range(45334, 56722)))

        # RNG for sampling, could seed here
        rng = random.Random()

        for i in tqdm(range(run_n_segments)):

            # Get prompt for this segment
            if i == 0:
                prompt_ids = self.get_first_segment_prompt(
                    prompt_texts[1],
                    prompt_texts[0],
                    use_dual_tracks_prompt,
                    vocal_track_prompt_path,
                    instrumental_track_prompt_path,
                    use_audio_prompt,
                    audio_prompt_path,
                    prompt_start_time,
                    prompt_end_time,
                )
            else:
                prompt_ids = self.get_segment_prompt(prompt_texts[i + 1])
            prompt_ids = torch.tensor([prompt_ids] * bsz, dtype=torch.long)

            # Accept prompt tokens
            seq = torch.cat((seq, prompt_ids), dim=-1)

            # Use window slicing in case output sequence exceeds the context of model
            max_context = self.cache_size - max_new_tokens - 1
            if seq.shape[-1] > max_context:
                print(f"Section {i}: output length {seq.shape[-1]} exceeding context length {max_context}, " f"dropping early segment(s) from prompt.")
                cache.current_seq_len = 0
                full_ids = self.shorten_input(seq, max_context)
                incremental_ids = full_ids
            else:
                full_ids = seq
                incremental_ids = prompt_ids

            # For the unconditional context, mask out all but the last token
            if cfg:
                mask_len = full_ids.shape[-1] - 1
                full_mask = torch.zeros((2, cache.max_seq_len), dtype=torch.half, device=self.device)
                full_mask[1, :mask_len] = -65504.0
                position_offsets = torch.tensor([[0], [-mask_len]], dtype=torch.int)
                input_mask = full_mask[:, : full_ids.shape[-1]]

            # Forward prompt
            logits = self.model.forward(incremental_ids[:, :], cache=cache, input_mask=input_mask, position_offsets=position_offsets, last_id_only=True)

            # Generate until EOS or max_new_tokens
            for new_tokens in tqdm(range(max_new_tokens)):

                # Transformers-equiv. CFG
                if cfg:
                    cfg_scale = sample_settings.guidance_scale_seg0 if i == 0 else sample_settings.guidance_scale
                    logits = logits.float()
                    logits = F.log_softmax(logits, dim=-1)
                    logits = cfg_scale * logits[0] + (1 - cfg_scale) * logits[1]
                    logits = logits.unsqueeze(0)

                # Sample
                logits = logits.float().cpu()
                sample, _, _, _, _ = ExLlamaV2Sampler.sample(logits, gen_settings, full_ids[:1], rng.random(), self.tokenizer)
                if cfg:
                    sample = torch.cat((sample, sample), dim=0)

                # Accept token
                full_ids = torch.cat((full_ids, sample), dim=-1)
                seq = torch.cat((seq, sample), dim=-1)

                # Get next logits (update cache even if sample is EOA and we don't need next logits)
                if cfg:
                    input_mask = full_mask[:, : full_ids.shape[-1]]
                logits = self.model.forward(sample, cache=cache, input_mask=input_mask, position_offsets=position_offsets)

                # End on EOA
                if sample[0].item() == self.mmtokenizer.eoa:
                    break

            # Make sure sequence ends with EOA if we reached max_new_tokens
            else:
                sample = torch.tensor([[self.mmtokenizer.eoa]] * bsz, dtype=torch.long)
                seq = torch.cat((seq, sample), dim=-1)
                # Update cache with forced token
                self.model.forward(sample, cache=cache)

        raw_output = seq[:1, :]
        empty_gpu_cache(self.is_cuda)
        return raw_output


# =========================================
#           STAGE 2 PIPELINE
# =========================================

def align(n, m):
    return ((n + m - 1) // m) * m


def split_bsz(bsz, maxbsz):
    n_sub_batches = math.ceil(bsz / maxbsz)
    base_size = bsz // n_sub_batches
    remainder = bsz % n_sub_batches
    sub_batch_sizes = [base_size + 1] * remainder + [base_size] * (n_sub_batches - remainder)
    indices = []
    start = 0
    for size in sub_batch_sizes:
        end = start + size
        indices.append((start, end))
        start = end
    return indices


class Stage2Pipeline:
    def __init__(self, device: torch.device, mmtokenizer: _MMSentencePieceTokenizer = None):
        self.device = device

        self.codec_tool = CodecManipulator("xcodec", 0, 1)
        self.codec_tool_stage2 = CodecManipulator("xcodec", 0, 8)

        # Load tokenizer
        self.mmtokenizer = mmtokenizer or _MMSentencePieceTokenizer(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mm_tokenizer_v0.2_hf", "tokenizer.model"))

        self.is_cuda = torch.cuda.is_available()

    def get_codec_ids(self, prompt: np.array):
        codec_ids = self.codec_tool.unflatten(prompt, n_quantizer=1)
        codec_ids = self.codec_tool.offset_tok_ids(
            codec_ids, global_offset=self.codec_tool.global_offset, codebook_size=self.codec_tool.codebook_size, num_codebooks=self.codec_tool.num_codebooks
        ).astype(np.int32)
        return codec_ids

    def fix_output(self, output):
        # Fix invalid codes (a dirty solution, which may harm the quality of audio)
        # We are trying to find better one
        fixed_output = copy.deepcopy(output)
        for i, line in enumerate(output):
            for j, element in enumerate(line):
                if element < 0 or element > 1023:
                    counter = Counter(line)
                    most_frequant = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
                    fixed_output[i, j] = most_frequant
        return fixed_output

    def save(self, output_dir: str, outputs):
        for output_name, output in outputs.items():
            # save output
            stage2_output_dir = os.path.join(output_dir, "stage2")
            os.makedirs(stage2_output_dir, exist_ok=True)
            output_filename = os.path.join(stage2_output_dir, output_name)
            np.save(output_filename, output)

    def get_stage1_prompt(self, output_dir: str, output_name: str):
        stage1_output_dir = os.path.join(output_dir, "stage1")
        prompt = np.load(os.path.join(stage1_output_dir, output_name)).astype(np.int32)
        return prompt

    def prepare_prompt_batch(self, prompt: np.array, batch_size: int):
        codec_ids = self.get_codec_ids(prompt)

        # Prepare prompt_ids based on batch size or single input
        if batch_size > 1:
            codec_list = []
            for i in range(batch_size):
                idx_begin = i * 300
                idx_end = (i + 1) * 300
                codec_list.append(codec_ids[:, idx_begin:idx_end])

            codec_ids = np.concatenate(codec_list, axis=0)
            prompt_ids = np.concatenate(
                [np.tile([self.mmtokenizer.soa, self.mmtokenizer.stage_1], (batch_size, 1)), codec_ids, np.tile([self.mmtokenizer.stage_2], (batch_size, 1))],
                axis=1,
            )
        else:
            prompt_ids = np.concatenate(
                [
                    np.array([self.mmtokenizer.soa, self.mmtokenizer.stage_1]),
                    codec_ids.flatten(),  # Flatten the 2D array to 1D
                    np.array([self.mmtokenizer.stage_2]),
                ]
            ).astype(np.int32)
            prompt_ids = prompt_ids[np.newaxis, ...]

        codec_ids = torch.as_tensor(codec_ids, dtype=torch.long)
        prompt_ids = torch.as_tensor(prompt_ids, dtype=torch.long)
        return codec_ids, prompt_ids


class Stage2Pipeline_HF(Stage2Pipeline):

    def __init__(self, model_path: str, device: torch.device, batch_size: int, **kwargs):
        super().__init__(device, **kwargs)
        self.batch_size = batch_size

        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2")
        self.model.to(device)
        self.model.eval()
        # if torch.__version__ >= "2.0.0":
        #     self.model = torch.compile(self.model)
        empty_gpu_cache(self.is_cuda)

    def generate_batch(self, prompt: np.array, batch_size: int):
        codec_ids, prompt_ids = self.prepare_prompt_batch(prompt, batch_size)
        len_prompt = prompt_ids.shape[-1]

        # Teacher forcing generate loop
        codec_ids = codec_ids.to(self.device)
        prompt_ids = prompt_ids.to(self.device)
        block_list = LogitsProcessorList([BlockTokenRangeProcessor(0, 46358), BlockTokenRangeProcessor(53526, self.mmtokenizer.vocab_size)])
        past_key_values = StaticCache(
            self.model.config,
            max_batch_size=batch_size,
            max_cache_len=prompt_ids.shape[1] + codec_ids.shape[1] * 8,
            device=self.model.device,
            dtype=self.model.dtype,
        )
        for frames_idx in range(codec_ids.shape[1]):
            cb0 = codec_ids[:, frames_idx : frames_idx + 1]
            prompt_ids = torch.cat([prompt_ids, cb0], dim=1)
            input_ids = prompt_ids

            stage2_output = self.model.generate(
                input_ids=input_ids,
                min_new_tokens=7,
                max_new_tokens=7,
                eos_token_id=self.mmtokenizer.eoa,
                pad_token_id=self.mmtokenizer.eoa,
                logits_processor=block_list,
                past_key_values=past_key_values,
            )

            assert stage2_output.shape[1] - prompt_ids.shape[1] == 7, f"output new tokens={stage2_output.shape[1]-prompt_ids.shape[1]}"
            prompt_ids = stage2_output

        # Return output based on batch size
        if batch_size > 1:
            output = prompt_ids.cpu().numpy()[:, len_prompt:]
            output_list = [output[i] for i in range(batch_size)]
            output = np.concatenate(output_list, axis=0)
        else:
            output = prompt_ids[0].cpu().numpy()[len_prompt:]

        return output

    def generate(self, vocals: np.ndarray, instrumentals: np.ndarray, output_dir: str) -> List[np.array]:
        """
        Returns vocals and instrumentals as numpy arrays.
        """
        outputs = []
        for prompt in tqdm([vocals, instrumentals]):
            # Only accept 6s segments
            output_duration = prompt.shape[-1] // 50 // 6 * 6
            num_batch = output_duration // 6

            if num_batch <= self.batch_size:
                # If num_batch is less than or equal to batch_size, we can infer the entire prompt at once
                output = self.generate_batch(prompt[:, : output_duration * 50], batch_size=num_batch)
            else:
                # If num_batch is greater than batch_size, process in chunks of batch_size
                segments = []
                num_segments = (num_batch // self.batch_size) + (1 if num_batch % self.batch_size != 0 else 0)

                for seg in range(num_segments):
                    start_idx = seg * self.batch_size * 300
                    # Ensure the end_idx does not exceed the available length
                    end_idx = min((seg + 1) * self.batch_size * 300, output_duration * 50)  # Adjust the last segment
                    current_batch_size = self.batch_size if seg != num_segments - 1 or num_batch % self.batch_size == 0 else num_batch % self.batch_size
                    segment = self.generate_batch(prompt[:, start_idx:end_idx], batch_size=current_batch_size)
                    segments.append(segment)

                # Concatenate all the segments
                output = np.concatenate(segments, axis=0)

            # Process the ending part of the prompt
            if output_duration * 50 != prompt.shape[-1]:
                ending = self.generate_batch(prompt[:, output_duration * 50 :], batch_size=1)
                output = np.concatenate([output, ending], axis=0)

            output = self.codec_tool_stage2.ids2npy(output)

            output = self.fix_output(output)
            outputs.append(output)
            empty_gpu_cache(self.is_cuda)
        return outputs


class Stage2Pipeline_EXL2(Stage2Pipeline):

    def __init__(self, model_path: str, device: torch.device, cache_size: int, cache_mode: str, **kwargs):
        super().__init__(device, **kwargs)

        self.cache_size = cache_size

        assert device != "cpu", "ExLlamaV2 does not support CPU inference."

        # Load EXL2 model
        device_idx = self.device.index
        gpu_split = [0] * torch.cuda.device_count()
        gpu_split[device_idx] = 9999
        exl2_config = ExLlamaV2Config(model_path)
        self.model = ExLlamaV2(exl2_config)
        self.model.load(gpu_split)

        # Move embedding layer to GPU to avoid CPU sync during argmax gen loop
        self.model.modules[0].device_idx = self.model.modules[1].device_idx
        self.model.modules[0].reload()

        # Load tokenizer (only needed for vocab size in disallow_tokens)
        self.tokenizer = ExLlamaV2Tokenizer(exl2_config)

        # Define cache
        self.cache_mode = get_cache_class(cache_mode)

    def generate(self, vocals: np.ndarray, instrumentals: np.ndarray, output_dir: str) -> List[np.array]:
        """
        Returns 2 parts as numpy arrays.
        """
        parts = [vocals, instrumentals]
        full_batch = []

        # Collect up to 300 token (6s) segments for all parts
        for output_idx, prompt in tqdm(enumerate(parts)):
            prompt = self.get_codec_ids(prompt)
            prompt = torch.as_tensor(prompt, dtype=torch.long)

            segs = torch.split(prompt, 300, dim=-1)

            for seg_idx, seg in enumerate(segs):
                seg_len = seg.shape[-1]
                full_batch.append((seg_len, seg_idx, output_idx, seg))

        # Prepare segments
        prefix = torch.tensor([[self.mmtokenizer.soa, self.mmtokenizer.stage_1]], dtype=torch.long)
        suffix = torch.tensor([[self.mmtokenizer.stage_2]], dtype=torch.long)
        for i in range(len(full_batch)):
            seg_len, seg_idx, output_idx, codec_ids = full_batch[i]
            prompt_ids = torch.cat((prefix, codec_ids, suffix), dim=-1)
            full_batch[i] = (seg_len, seg_idx, output_idx, codec_ids, prompt_ids)

        # Group prompts by length
        batches = {}
        for seq in full_batch:
            if not seq[0] in batches:
                batches[seq[0]] = []
            batches[seq[0]].append(seq)

        # Split into on minibatches
        split_batch = []
        for idx, (seg_len, batch) in enumerate(batches.items()):
            b_seg_order = [b[1] for b in batch]
            b_part_order = [b[2] for b in batch]
            b_codec_ids = torch.cat([b[3] for b in batch], dim=0)
            b_prompt_ids = torch.cat([b[4] for b in batch], dim=0)

            max_bsz = self.cache_size // align(b_prompt_ids.shape[1] + b_codec_ids.shape[1] * 8, 32)
            assert max_bsz > 0
            for a, b in split_bsz(b_prompt_ids.shape[0], max_bsz):
                split_batch.append((b_seg_order[a:b], b_part_order[a:b], b_codec_ids[a:b], b_prompt_ids[a:b]))

        # Inference
        output_parts = []
        for _ in parts:
            output_parts.append([])

        for seg_order, part_order, codec_ids, prompt_ids in tqdm(split_batch):
            codec_ids = codec_ids.to(self.device)
            prompt_ids = prompt_ids.to(self.device)
            batch_size, len_prompt = prompt_ids.shape

            cache = self.cache_mode(self.model, batch_size=batch_size, max_seq_len=align(prompt_ids.shape[1] + codec_ids.shape[1] * 8, 32))
            output_ids = torch.empty((batch_size, 0), dtype=torch.long, device=self.device)

            for frames_idx in tqdm(range(codec_ids.shape[1])):
                cb0 = codec_ids[:, frames_idx : frames_idx + 1]

                # Append the initial prompt to the first codec frame
                if frames_idx == 0:
                    cb0 = torch.cat([prompt_ids, cb0], dim=-1)

                # Forward prompt
                output_ids = torch.cat((output_ids, cb0), dim=-1)
                logits = self.model.forward(cb0, cache=cache, last_id_only=True)

                for i in range(7):

                    # Slice logits instead of biasing start and end of distribution
                    first_logit = 46358
                    last_logit = 53526
                    logits = logits[:, :, first_logit:last_logit]

                    # Greedy sampling
                    sample = logits.argmax(dim=-1) + first_logit
                    output_ids = torch.cat((output_ids, sample), dim=-1)

                    # TODO: Here, original asserts that we didn't sample mmtokenizer.eoa (can we just mask it out?)

                    # Forward sample
                    logits = self.model.forward(sample, cache=cache)

            # Trim prompt
            output_ids = output_ids[:, len_prompt:]

            # Split outputs
            for i in range(batch_size):
                output_parts[part_order[i]].append((seg_order[i], output_ids[i : i + 1, :]))

            # Release cache tensors
            del cache
            empty_gpu_cache(self.is_cuda)

        # Unshuffle and recombine output parts
        output = []
        for i, p in enumerate(output_parts):
            p = sorted(p, key=lambda x: x[0])
            part_o = torch.cat([pp[1] for pp in p], dim=-1).flatten().cpu().numpy()
            part_o = self.codec_tool_stage2.ids2npy(part_o)
            part_o = self.fix_output(part_o)
            output.append(part_o)

        return output


# =========================================
#           POST PROCESSING
# =========================================

def save_audio(wav: torch.Tensor, path, sample_rate: int, rescale: bool = False):
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    limit = 0.99
    max_val = wav.abs().max()
    wav = wav * min(limit / max_val, 1) if rescale else wav.clamp(-limit, limit)
    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding="PCM_S", bits_per_sample=16)


def post_process(vocals: np.ndarray, instrumentals: np.ndarray, codec_model: SoundStream, vocal_decoder: VocosDecoder, inst_decoder: VocosDecoder, device: torch.device, output_dir: str, rescale: bool):
    # reconstruct tracks
    recons_output_dir = os.path.join(output_dir, "recons")
    recons_mix_dir = os.path.join(recons_output_dir, "mix")
    os.makedirs(recons_mix_dir, exist_ok=True)

    stage2_result = [vocals, instrumentals]
    result_filenames = ["vtrack.npy", "itrack.npy"]

    tracks = []
    for idx, codec_result in enumerate(stage2_result):
        decodec_rlt = []
        decoded_waveform = codec_model.decode(torch.as_tensor(codec_result.astype(np.int16), dtype=torch.long).unsqueeze(0).permute(1, 0, 2).to(device))
        decoded_waveform = decoded_waveform.cpu().squeeze(0)
        decodec_rlt.append(torch.as_tensor(decoded_waveform))
        decodec_rlt = torch.cat(decodec_rlt, dim=-1)
        save_path = os.path.join(recons_output_dir, os.path.splitext(os.path.basename(result_filenames[idx]))[0] + ".mp3")
        tracks.append(save_path)
        save_audio(decodec_rlt, save_path, 16000)

    # mix tracks
    for inst_path in tracks:
        try:
            if (inst_path.endswith(".wav") or inst_path.endswith(".mp3")) and "itrack" in inst_path:
                # find pair
                vocal_path = inst_path.replace("itrack", "vtrack")
                if not os.path.exists(vocal_path):
                    continue
                # mix
                recons_mix = os.path.join(recons_mix_dir, os.path.basename(inst_path).replace("itrack", "mixed"))
                vocal_stem, sr = sf.read(inst_path)
                instrumental_stem, _ = sf.read(vocal_path)
                mix_stem = (vocal_stem + instrumental_stem) / 1
                sf.write(recons_mix, mix_stem, sr)
        except Exception as e:
            print(e)

    # vocoder to upsample audios
    vocoder_output_dir = os.path.join(output_dir, "vocoder")
    vocoder_stems_dir = os.path.join(vocoder_output_dir, "stems")
    vocoder_mix_dir = os.path.join(vocoder_output_dir, "mix")
    os.makedirs(vocoder_mix_dir, exist_ok=True)
    os.makedirs(vocoder_stems_dir, exist_ok=True)
    for i, input_array in enumerate(stage2_result):
        if i == 1:
            # Process instrumental
            instrumental_output = process_audio(input_array, os.path.join(vocoder_stems_dir, "itrack.mp3"), rescale, device, inst_decoder, codec_model)
        else:
            # Process vocal
            vocal_output = process_audio(input_array, os.path.join(vocoder_stems_dir, "vtrack.mp3"), rescale, device, vocal_decoder, codec_model)

    # mix tracks
    try:
        mix_output = instrumental_output + vocal_output
        vocoder_mix = os.path.join(vocoder_mix_dir, os.path.basename(recons_mix))
        save_audio(mix_output, vocoder_mix, 44100, rescale)
        print(f"Created mix: {vocoder_mix}")
    except RuntimeError as e:
        print(e)
        print(f"mix {vocoder_mix} failed! inst: {instrumental_output.shape}, vocal: {vocal_output.shape}")

    # Post process
    replace_low_freq_with_energy_matched(
        a_file=recons_mix, b_file=vocoder_mix, c_file=os.path.join(output_dir, os.path.basename(recons_mix)), cutoff_freq=5500.0  # 16kHz  # 48kHz
    )


# =========================================
#         RUN THE WHOLE FKN THING
# =========================================

def main():
    args = parser.parse_args()
    if args.use_audio_prompt and not args.audio_prompt_path:
        raise FileNotFoundError("Please offer audio prompt filepath using '--audio_prompt_path', when you enable 'use_audio_prompt'!")
    if args.use_dual_tracks_prompt and not args.vocal_track_prompt_path and not args.instrumental_track_prompt_path:
        raise FileNotFoundError(
            "Please offer dual tracks prompt filepath using '--vocal_track_prompt_path' and '--inst_decoder_path', when you enable '--use_dual_tracks_prompt'!"
        )
    if args.seed is not None:
        seed_everything(args.seed)

    device = torch.device(f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu")

    with open(args.genre_txt) as f:
        genres = f.read().strip()
    with open(args.lyrics_txt) as f:
        lyrics = f.read().strip()

    # loads required models
    mmtokenizer = _MMSentencePieceTokenizer(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mm_tokenizer_v0.2_hf", "tokenizer.model"))
    empty_gpu_cache(torch.cuda.is_available())

    model_config = OmegaConf.load(args.basic_model_config)
    assert model_config.generator.name == "SoundStream"
    codec_model = SoundStream(**model_config.generator.config).to(device)
    parameter_dict = torch.load(args.resume_path, map_location=device, weights_only=False)
    codec_model.load_state_dict(parameter_dict["codec_model"])
    codec_model.eval()
    empty_gpu_cache(torch.cuda.is_available())

    vocal_decoder, inst_decoder = build_codec_model(args.config_path, args.vocal_decoder_path, args.inst_decoder_path)
    empty_gpu_cache(torch.cuda.is_available())

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print(f">> Creating pipeline for stage 1 using exl2={args.stage1_use_exl2} ...")
    start_event.record()
    if args.stage1_use_exl2:
        pipeline = Stage1Pipeline_EXL2(
            model_path=args.stage1_model,
            device=device,
            basic_model_config=args.basic_model_config,
            resume_path=args.resume_path,
            cache_size=args.stage1_cache_size,
            cache_mode=args.stage1_cache_mode,
            mmtokenizer=mmtokenizer,
            codec_model=codec_model,
        )
    else:
        pipeline = Stage1Pipeline_HF(
            model_path=args.stage1_model,
            device=device,
            basic_model_config=args.basic_model_config,
            resume_path=args.resume_path,
            cache_size=args.stage1_cache_size,
            mmtokenizer=mmtokenizer,
            codec_model=codec_model,
        )
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Stage 1 Pipeline creation execution time: {elapsed_time_ms:.4f} ms\n")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print(f">> Creating pipeline for stage 2 using exl2={args.stage2_use_exl2} ...")    
    start_event.record()
    if args.stage2_use_exl2:
        pipeline2 = Stage2Pipeline_EXL2(
            model_path=args.stage2_model,
            device=device,
            cache_size=args.stage2_cache_size,
            cache_mode=args.stage2_cache_mode,
            mmtokenizer=mmtokenizer,
        )
    else:
        pipeline2 = Stage2Pipeline_HF(
            model_path=args.stage2_model,
            device=device,
            batch_size=args.stage2_batch_size,
            mmtokenizer=mmtokenizer,
        )
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Stage 2 pipeline preparation execution time: {elapsed_time_ms:.4f} ms\n")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print(f">> Generating stage 1...")
    start_event.record()
    # Load tokenizer and models
    raw_output = pipeline.generate(
        use_dual_tracks_prompt=args.use_dual_tracks_prompt,
        vocal_track_prompt_path=args.vocal_track_prompt_path,
        instrumental_track_prompt_path=args.instrumental_track_prompt_path,
        use_audio_prompt=args.use_audio_prompt,
        audio_prompt_path=args.audio_prompt_path,
        genres=genres,
        lyrics=lyrics,
        run_n_segments=args.run_n_segments,
        max_new_tokens=args.max_new_tokens,
        prompt_start_time=args.prompt_start_time,
        prompt_end_time=args.prompt_end_time,
        sample_settings=SampleSettings(use_guidance=not args.stage1_no_guidance, repetition_penalty=args.repetition_penalty),
    )
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Stage 1 execution time: {elapsed_time_ms:.4f} ms\n")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print(f">> Preprocessing for stage 2...")
    start_event.record()
    #### STAGE 2
    vocals, instrumentals = pipeline.post_process_for_next_stage(
        raw_output, args.use_audio_prompt, args.use_dual_tracks_prompt
    )
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Stage 2 pre-processing time: {elapsed_time_ms:.4f} ms\n")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print(">> Generating stage 2...")
    start_event.record()
    outputs = pipeline2.generate(vocals, instrumentals, args.output_dir)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Stage 2 execution time: {elapsed_time_ms:.4f} ms\n")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print(">> Postprocessing final...")
    start_event.record()
    post_process(outputs[0], outputs[1], codec_model, vocal_decoder, inst_decoder, device, args.output_dir, args.rescale)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Final post-processing time: {elapsed_time_ms:.4f} ms\n")

    print(f">> Done.")
