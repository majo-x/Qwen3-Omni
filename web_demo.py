import io
import os
import sys
import torch

os.environ['VLLM_USE_V1'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
from argparse import ArgumentParser

# Voice-cloning helpers from the model-training repo. Required: predict() uses
# generate_audio_talker_direct() (via the splice path) instead of the noisy
# integrated model.generate() Talker.
_scripts_path = os.path.expanduser("~/repos/internal_repos/model-training/scripts")
if _scripts_path not in sys.path:
    sys.path.insert(0, _scripts_path)
from inference_tts import (  # noqa: E402
    _load_speaker_embedding,
    build_spliced_embeds as _build_spliced_embeds,
    generate_audio_talker_direct as _generate_audio_talker_direct,
)
from train_talker import DEFAULT_USER_INSTRUCTION as _DEFAULT_USER_INSTRUCTION  # noqa: E402

import gradio as gr
import gradio.processing_utils as processing_utils
import numpy as np
import soundfile as sf

from gradio_client import utils as client_utils
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeProcessor


def _load_model_processor(args):
    if args.use_transformers:
        import json, torch as _torch
        from transformers import Qwen3OmniMoeForConditionalGeneration
        attn = 'flash_attention_2' if args.flash_attn2 else 'sdpa'

        # Dispatch on checkpoint_info.json written by scripts/train_talker.py:
        #   checkpoint_type == "talker_only"      -> self-contained 15-shard checkpoint
        #       with trained Talker overlaid; load directly from args.checkpoint_path.
        #       Processor files may not be present; fall back to base_model_path.
        #   checkpoint_type == "full_safetensors" -> deprecated alias, same handling.
        #   (no checkpoint_info.json)             -> pristine HF checkpoint.
        _ckpt_info_path = os.path.join(args.checkpoint_path, "checkpoint_info.json")
        _processor_path = args.checkpoint_path
        _ckpt_type = None
        _ckpt_info = None
        if os.path.exists(_ckpt_info_path):
            with open(_ckpt_info_path) as _f:
                _ckpt_info = json.load(_f)
            _ckpt_type = _ckpt_info.get("checkpoint_type")

        if _ckpt_type in ("talker_only", "full_safetensors"):
            print(f"Talker-only checkpoint detected. Loading from {args.checkpoint_path} ...")
            # Processor files are not stored in talker-only checkpoints;
            # fall back to base_model_path for tokenizer / preprocessor.
            if _ckpt_info and "base_model_path" in _ckpt_info:
                _base_path = os.path.expanduser(_ckpt_info["base_model_path"])
                if os.path.exists(os.path.join(_base_path, "preprocessor_config.json")):
                    _processor_path = _base_path

        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            args.checkpoint_path, device_map="cuda:0", dtype='auto', attn_implementation=attn)
    else:
        from vllm import LLM
        model = LLM(
            model=args.checkpoint_path, trust_remote_code=True, gpu_memory_utilization=0.95,
            tensor_parallel_size=torch.cuda.device_count(),
            limit_mm_per_prompt={'image': 1, 'video': 5, 'audio': 10},
            max_num_seqs=1,
            max_model_len=32768,
            seed=1234,
        )

    _final_processor_path = args.processor_path if args.processor_path else _processor_path
    # For full-model checkpoints saved without processor files, read the base
    # path from config.json's _name_or_path field as a last resort.
    if not os.path.exists(os.path.join(_final_processor_path, "preprocessor_config.json")):
        import json as _json
        _cfg = os.path.join(_final_processor_path, "config.json")
        if os.path.exists(_cfg):
            with open(_cfg) as _f:
                _name_or_path = _json.load(_f).get("_name_or_path", "")
            if _name_or_path and os.path.isdir(_name_or_path) and \
                    os.path.exists(os.path.join(_name_or_path, "preprocessor_config.json")):
                print(f"preprocessor_config.json not found in checkpoint; falling back to {_name_or_path}")
                _final_processor_path = _name_or_path
    processor = Qwen3OmniMoeProcessor.from_pretrained(_final_processor_path)
    return model, processor


def _extract_last_user_ids(
    input_ids: "torch.Tensor", tokenizer
) -> "torch.Tensor | None":
    """Return a [1, L] tensor with the last user turn from the Thinker's input_ids.

    The slice spans <|im_start|>user\\n ... <|im_end|>\\n and may contain audio
    placeholder tokens inserted by the processor.  When passed to
    build_spliced_embeds as *user_ids*, every token (including audio placeholders)
    is embedded via the layer-0 embedding table — the tp(e_aud) path for Step 2.

    Returns None if no user turn is found (should not happen in normal usage).
    """
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")

    user_header_ids = tokenizer(
        "<|im_start|>user\n", return_tensors="pt", add_special_tokens=False
    ).input_ids[0].tolist()
    header_len = len(user_header_ids)

    ids = input_ids[0].tolist()
    n = len(ids)

    # The last <|im_start|> belongs to the generation prompt (<|im_start|>assistant\n).
    # Everything from the previous user-header up to that point is the last user turn,
    # including the trailing \n after <|im_end|>.
    im_start_positions = [i for i, tok in enumerate(ids) if tok == im_start_id]
    if len(im_start_positions) < 2:
        return None
    gen_prompt_start = im_start_positions[-1]

    last_user_start = None
    for i in range(gen_prompt_start - header_len, -1, -1):
        if ids[i : i + header_len] == user_header_ids:
            last_user_start = i
            break

    if last_user_start is None:
        return None

    return input_ids[0, last_user_start:gen_prompt_start].unsqueeze(0)


def _launch_demo(args, model, processor):
    # --- Start of Function: UNCHANGED settings ---
    # Voice settings
    VOICE_LIST = ['Chelsie', 'Ethan', 'Aiden']
    DEFAULT_VOICE = 'Chelsie'
    default_system_prompt = ''
    use_transformers = args.use_transformers
    generate_audio = args.generate_audio
    # When True: talker input uses only layer0 (token embeddings, matching training).
    # When False: model.generate() runs the integrated talker which uses layer24
    #             (thinker hidden states) for audio token positions.
    talker_layer0_only = args.layer0_only
    if not use_transformers:
        if generate_audio:
            print("Generating audio is not supported with vLLM. Please use the 'python web_demo.py --use-transformers --generate-audio --flash-attn2' instead.")
        from vllm import SamplingParams
    
    if use_transformers:
        if generate_audio:
            default_system_prompt = """You are a virtual voice assistant with no gender or age.\nYou are communicating with the user.\nIn user messages, “I/me/my/we/our” refer to the user and “you/your” refer to the assistant. In your replies, address the user as “you/your” and yourself as “I/me/my”; never mirror the user’s pronouns—always shift perspective. Keep original pronouns only in direct quotes; if a reference is unclear, ask a brief clarifying question.\nInteract with users using short(no more than 50 words), brief, straightforward language, maintaining a natural tone.\nNever use formal phrasing, mechanical expressions, bullet points, overly structured language. \nYour output must consist only of the spoken content you want the user to hear. \nDo not include any descriptions of actions, emotions, sounds, or voice changes. \nDo not use asterisks, brackets, parentheses, or any other symbols to indicate tone or actions. \nYou must answer users' audio or text questions, do not directly describe the video content. \nYou should communicate in the same language strictly as the user unless they request otherwise.\nWhen you are uncertain (e.g., you can't see/hear clearly, don't understand, or the user makes a comment rather than asking a question), use appropriate questions to guide the user to continue the conversation.\nKeep replies concise and conversational, as if talking face-to-face."""
        else:
            model.disable_talker()

    def to_mp4(path):
        import subprocess
        if path and path.endswith(".webm"):
            mp4_path = path.replace(".webm", ".mp4")
            subprocess.run([
                "ffmpeg", "-y",
                "-i", path,
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "fastdecode",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "128k",
                "-threads", "0",
                "-f", "mp4",
                mp4_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return mp4_path
        return path

    def format_history(history: list, system_prompt: str):
        messages = []
        if system_prompt != "":
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        current_user_content = []
        for item in history:
            role, content = item['role'], item['content']
            if role != "user":
                if current_user_content:
                    messages.append({"role": "user", "content": current_user_content})
                    current_user_content = []
                if isinstance(content, str):
                    messages.append({"role": role, "content": [{"type": "text", "text": content}]})
                else: pass
                continue
            if isinstance(content, str):
                current_user_content.append({"type": "text", "text": content})
            elif isinstance(content, (list, tuple)):
                for file_path in content:
                    mime_type = client_utils.get_mimetype(file_path)
                    media_type = "image" if mime_type.startswith("image") else "video" if mime_type.startswith("video") else "audio" if mime_type.startswith("audio") else None
                    if media_type == "video":
                        file_path = to_mp4(file_path)
                    if media_type:
                        current_user_content.append({"type": media_type, media_type: file_path})
                    else:
                        current_user_content.append({"type": "text", "text": file_path})
        if current_user_content:
            media_items = [item for item in current_user_content if item["type"] != "text"]
            text_items = [item for item in current_user_content if item["type"] == "text"]
            messages.append({"role": "user", "content": media_items + text_items})

        IMAGE_TURN_LIMIT = 1
        VIDEO_TURN_LIMIT = 5
        AUDIO_TURN_LIMIT = 5

        image_turn_indices = []
        video_turn_indices = []
        audio_turn_indices = []

        for i, message in enumerate(messages):
            if message['role'] == 'user':
                has_image = False
                has_video = False
                has_audio = False
                for item in message.get('content', []):
                    if item.get('type') == 'image':
                        has_image = True
                    elif item.get('type') == 'video':
                        has_video = True
                    elif item.get('type') == 'audio':
                        has_audio = True
                if has_image:
                    image_turn_indices.append(i)
                if has_video:
                    video_turn_indices.append(i)
                if has_audio:
                    audio_turn_indices.append(i)

        indices_to_delete = set()
        while len(image_turn_indices) > IMAGE_TURN_LIMIT:
            turn_start_index = image_turn_indices.pop(0)
            if turn_start_index in indices_to_delete:
                continue
            turn_end_index = turn_start_index + 1
            while (turn_end_index < len(messages) and 
                messages[turn_end_index]['role'] == 'assistant'):
                turn_end_index += 1
            for i in range(turn_start_index, turn_end_index):
                indices_to_delete.add(i)
        while len(video_turn_indices) > VIDEO_TURN_LIMIT:
            turn_start_index = video_turn_indices.pop(0)
            if turn_start_index in indices_to_delete:
                continue
            turn_end_index = turn_start_index + 1
            while (turn_end_index < len(messages) and 
                messages[turn_end_index]['role'] == 'assistant'):
                turn_end_index += 1
            for i in range(turn_start_index, turn_end_index):
                indices_to_delete.add(i)
        while len(audio_turn_indices) > AUDIO_TURN_LIMIT:
            turn_start_index = audio_turn_indices.pop(0)
            if turn_start_index in indices_to_delete:
                continue
            turn_end_index = turn_start_index + 1
            while (turn_end_index < len(messages) and 
                messages[turn_end_index]['role'] == 'assistant'):
                turn_end_index += 1
            for i in range(turn_start_index, turn_end_index):
                indices_to_delete.add(i)

        if indices_to_delete:
            final_messages = [msg for i, msg in enumerate(messages) if i not in indices_to_delete]
            messages = final_messages

        return messages

    def predict(messages, voice_choice=DEFAULT_VOICE, temperature=0.7, top_p=0.8, top_k=20):
        print('predict history: ', messages)
        if use_transformers:
            text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
            inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
            inputs = inputs.to(model.device).to(model.dtype)

            if talker_layer0_only:
                # Layer0-only path: thinker generates text, then the talker is driven
                # via build_spliced_embeds() + generate_audio_talker_direct(), which
                # uses only layer0 (token embeddings) for all talker input positions.
                # This matches the training procedure and avoids the layer24 mismatch.
                text_ids, _ = model.generate(
                    **inputs,
                    thinker_return_dict_in_generate=True,
                    thinker_max_new_tokens=32768,
                    thinker_do_sample=True,
                    thinker_temperature=temperature,
                    thinker_top_p=top_p,
                    thinker_top_k=top_k,
                    speaker=voice_choice,
                    use_audio_in_video=True,
                    return_audio=False,
                )
                reply_ids = text_ids.sequences[:, inputs["input_ids"].shape[1]:]
                response = processor.batch_decode(reply_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                yield {"type": "text", "data": response}

                # Build the talker's user segment.
                #
                # Step 2: extract the last user turn directly from inputs["input_ids"].
                #   This tensor may contain audio placeholder tokens (mic input), so
                #   the talker sees tp(e_aud) — their layer-0 embeddings — for each
                #   audio position.  For text-only input the result is equivalent to
                #   Step 1 (same tokens, same layer-0 path).
                #
                # Step 1 fallback: if extraction fails, use the last typed message
                #   as plain text (conversational format, no audio tokens).
                #
                # Step 0 fallback: if no text either, use the TTS instruction frame.
                _user_ids = _extract_last_user_ids(inputs["input_ids"], processor.tokenizer)

                _last_user_text = None
                if _user_ids is None:
                    # Extraction failed — fall back to Step 1 (text) or Step 0 (TTS).
                    for _msg in reversed(messages):
                        if _msg.get("role") == "user":
                            _content = _msg.get("content", [])
                            if isinstance(_content, list):
                                _texts = [c["text"] for c in _content if isinstance(c, dict) and c.get("type") == "text"]
                                if _texts:
                                    _last_user_text = " ".join(_texts)
                            elif isinstance(_content, str):
                                _last_user_text = _content
                            break

                user_embeds, asst_embeds = _build_spliced_embeds(
                    model, processor.tokenizer, reply_ids[0],
                    user_instruction=_DEFAULT_USER_INSTRUCTION,
                    user_text=_last_user_text,
                    user_ids=_user_ids,
                )
                audio_wav = _generate_audio_talker_direct(
                    model,
                    processor.tokenizer,
                    response,
                    speaker=voice_choice,
                    speaker_embedding=_speaker_embedding,  # None unless --ref-audio given
                    user_instruction=_DEFAULT_USER_INSTRUCTION,
                    max_new_tokens=2048,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=1.05,
                    max_text_length=512,
                )
                if audio_wav is not None:
                    audio_np = np.array(audio_wav.reshape(-1).float().detach().cpu().numpy() * 32767).astype(np.int16)
                    wav_io = io.BytesIO()
                    sf.write(wav_io, audio_np, samplerate=24000, format="WAV")
                    audio_path = processing_utils.save_bytes_to_cache(wav_io.getvalue(), "audio.wav", cache_dir=demo.GRADIO_CACHE)
                    yield {"type": "audio", "data": audio_path}
            else:
                # Original path: integrated model.generate() runs both thinker and
                # talker together. Audio token positions in the user segment receive
                # layer24 (thinker hidden states via hidden_projection).
                text_ids, audio = model.generate(
                    **inputs,
                    thinker_return_dict_in_generate=True,
                    thinker_max_new_tokens=32768,
                    thinker_do_sample=True,
                    thinker_temperature=temperature,
                    thinker_top_p=top_p,
                    thinker_top_k=top_k,
                    speaker=voice_choice,
                    use_audio_in_video=True,
                )
                response = processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                yield {"type": "text", "data": response}
                if audio is not None:
                    audio = np.array(audio.reshape(-1).float().detach().cpu().numpy() * 32767).astype(np.int16)
                    wav_io = io.BytesIO()
                    sf.write(wav_io, audio, samplerate=24000, format="WAV")
                    audio_path = processing_utils.save_bytes_to_cache(wav_io.getvalue(), "audio.wav", cache_dir=demo.GRADIO_CACHE)
                    yield {"type": "audio", "data": audio_path}
        else:
            sampling_params = SamplingParams(temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=16384)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
            inputs = {'prompt': text, 'multi_modal_data': {}, "mm_processor_kwargs": {"use_audio_in_video": True}}
            if images is not None: inputs['multi_modal_data']['image'] = images
            if videos is not None: inputs['multi_modal_data']['video'] = videos
            if audios is not None: inputs['multi_modal_data']['audio'] = audios
            outputs = model.generate(inputs, sampling_params=sampling_params)
            response = outputs[0].outputs[0].text
            yield {"type": "text", "data": response}

    def media_predict(audio, video, history, system_prompt, voice_choice, temperature, top_p, top_k):
        yield (None, None, history, gr.update(visible=False), gr.update(visible=True))
        files = [audio, video];
        for f in files:
            if f: history.append({"role": "user", "content": (f,)})
        yield (None, None, history, gr.update(visible=True), gr.update(visible=False))
        formatted_history = format_history(history=history, system_prompt=system_prompt)
        history.append({"role": "assistant", "content": ""})
        for chunk in predict(formatted_history, voice_choice, temperature, top_p, top_k):
            if chunk["type"] == "text":
                history[-1]["content"] = chunk["data"]
                yield (None, None, history, gr.update(visible=False), gr.update(visible=True))
            if chunk["type"] == "audio":
                history.append({"role": "assistant", "content": gr.Audio(chunk["data"], autoplay=False)})
        yield (None, None, history, gr.update(visible=True), gr.update(visible=False))

    def chat_predict(text, audio, image, video, history, system_prompt, voice_choice, temperature, top_p, top_k):
        user_content = []
        if audio: user_content.append((audio,))
        if image: user_content.append((image,))
        if video: user_content.append((video,))
        if user_content:
            flat_content = tuple(item for sublist in user_content for item in sublist)
            history.append({"role": "user", "content": flat_content})
        if text:
            history.append({"role": "user", "content": text})
        formatted_history = format_history(history=history, system_prompt=system_prompt)
        yield gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), history
        history.append({"role": "assistant", "content": ""})
        for chunk in predict(formatted_history, voice_choice, temperature, top_p, top_k):
            if chunk["type"] == "text":
                history[-1]["content"] = chunk["data"]
                yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), history
            if chunk["type"] == "audio":
                history.append({"role": "assistant", "content": gr.Audio(chunk["data"], autoplay=False)})
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), history
    
    # --- CORRECTED UI LAYOUT ---
    with gr.Blocks(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"]), css=".gradio-container {max-width: none !important;}") as demo:
        gr.Markdown("# Qwen3-Omni Demo")
        gr.Markdown("**Instructions**: Interact with the model through text, audio, images, or video. Use the tabs to switch between Online and Offline chat modes.")
        gr.Markdown("**使用说明**：1️⃣ 点击音频录制按钮，或摄像头-录制按钮 2️⃣ 输入音频或者视频 3️⃣ 点击提交并等待模型的回答")
        
        with gr.Row(equal_height=False):
            with gr.Column(scale=1): 
                gr.Markdown("### ⚙️ Parameters (参数)")
                system_prompt_textbox = gr.Textbox(label="System Prompt", value=default_system_prompt, lines=4, max_lines=8)
                voice_choice = gr.Dropdown(label="Voice Choice", choices=VOICE_LIST, value=DEFAULT_VOICE, visible=generate_audio)
                temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, value=0.6, step=0.1)
                top_p = gr.Slider(label="Top P", minimum=0.05, maximum=1.0, value=0.95, step=0.05)
                top_k = gr.Slider(label="Top K", minimum=1, maximum=100, value=20, step=1)
                
            with gr.Column(scale=3): 
                with gr.Tabs():
                    with gr.TabItem("Online"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### Audio-Video Input (音视频输入)")
                                microphone = gr.Audio(sources=['microphone'], type="filepath", label="Record Audio (录制音频)")
                                webcam = gr.Video(sources=['webcam', "upload"], label="Record/Upload Video (录制/上传视频)", elem_classes="media-upload")
                                with gr.Row():
                                    submit_btn_online = gr.Button("Submit (提交)", variant="primary", scale=2)
                                    stop_btn_online = gr.Button("Stop (停止)", visible=False, scale=1)
                                clear_btn_online = gr.Button("Clear History (清除历史)")
                            with gr.Column(scale=2):
                                # FIX: Re-added type="messages"
                                media_chatbot = gr.Chatbot(label="Chat History (对话历史)", type="messages", height=650, layout="panel", bubble_full_width=False, allow_tags=["think"], render=False)
                                media_chatbot.render()

                        def clear_history_online():
                            return [], None, None

                        submit_event_online = submit_btn_online.click(
                            fn=media_predict,
                            inputs=[microphone, webcam, media_chatbot, system_prompt_textbox, voice_choice, temperature, top_p, top_k],
                            outputs=[microphone, webcam, media_chatbot, submit_btn_online, stop_btn_online]
                        )
                        stop_btn_online.click(fn=lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[submit_btn_online, stop_btn_online], cancels=[submit_event_online], queue=False)
                        clear_btn_online.click(fn=clear_history_online, outputs=[media_chatbot, microphone, webcam])

                    with gr.TabItem("Offline"):
                        # FIX: Re-added type="messages"
                        chatbot = gr.Chatbot(label="Chat History (对话历史)", type="messages", height=550, layout="panel", bubble_full_width=False, allow_tags=["think"], render=False)
                        chatbot.render()
                        
                        with gr.Accordion("📎 Click to upload multimodal files (点击上传多模态文件)", open=False):
                            with gr.Row():
                                audio_input = gr.Audio(sources=["upload", 'microphone'], type="filepath", label="Audio", elem_classes="media-upload")
                                image_input = gr.Image(sources=["upload", 'webcam'], type="filepath", label="Image", elem_classes="media-upload")
                                video_input = gr.Video(sources=["upload", 'webcam'], label="Video", elem_classes="media-upload")

                        with gr.Row():
                            text_input = gr.Textbox(show_label=False, placeholder="Enter text or upload files and press Submit... (输入文本或者上传文件并点击提交)", scale=7)
                            submit_btn_offline = gr.Button("Submit (提交)", variant="primary", scale=1)
                            stop_btn_offline = gr.Button("Stop (停止)", visible=False, scale=1)
                            clear_btn_offline = gr.Button("Clear (清空) ", scale=1)
                        
                        def clear_history_offline():
                            return [], None, None, None, None

                        submit_event_offline = gr.on(
                            triggers=[submit_btn_offline.click, text_input.submit],
                            fn=chat_predict,
                            inputs=[text_input, audio_input, image_input, video_input, chatbot, system_prompt_textbox, voice_choice, temperature, top_p, top_k],
                            outputs=[text_input, audio_input, image_input, video_input, chatbot]
                        )
                        stop_btn_offline.click(fn=lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[submit_btn_offline, stop_btn_offline], cancels=[submit_event_offline], queue=False)
                        clear_btn_offline.click(fn=clear_history_offline, outputs=[chatbot, text_input, audio_input, image_input, video_input])

        gr.HTML("""
            <style>
                .media-upload { min-height: 160px; border: 2px dashed #ccc; border-radius: 8px; display: flex; align-items: center; justify-content: center; }
                .media-upload:hover { border-color: #666; }
            </style>
        """)

    demo.queue(default_concurrency_limit=1 if use_transformers else 100, max_size=100).launch(max_threads=100, ssr_mode=False, share=args.share, inbrowser=args.inbrowser, server_port=args.server_port, server_name=args.server_name)



DEFAULT_CKPT_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

def _get_args():
    parser = ArgumentParser()

    parser.add_argument('-c',
                        '--checkpoint-path',
                        type=str,
                        default=DEFAULT_CKPT_PATH,
                        help='Checkpoint name or path, default to %(default)r')
    parser.add_argument('--processor-path',
                        type=str,
                        default=None,
                        help='Path to load the processor from. Defaults to --checkpoint-path. '
                             'Needed when the checkpoint directory lacks processor files '
                             '(e.g. full model saved without processor, or talker-only checkpoint).')
    parser.add_argument('--generate-audio',
                        action='store_true',
                        default=False,
                        help='Enable audio generation.')
    parser.add_argument('--flash-attn2',
                        action='store_true',
                        default=False,
                        help='Enable flash_attention_2 when loading the model for transformers.')
    parser.add_argument('--use-transformers',
                        action='store_true',
                        default=False,
                        help='Use transformers for inference.')
    parser.add_argument('--share',
                        action='store_true',
                        default=False,
                        help='Create a publicly shareable link for the interface.')
    parser.add_argument('--inbrowser',
                        action='store_true',
                        default=False,
                        help='Automatically launch the interface in a new tab on the default browser.')
    parser.add_argument('--server-port', type=int, default=8901, help='Demo server port.')
    parser.add_argument('--server-name', type=str, default='127.0.0.1', help='Demo server name.')
    parser.add_argument(
        '--ref-audio',
        type=str,
        default=None,
        help=(
            'Path to a reference WAV file for voice cloning. '
            'Extracts a speaker embedding and uses generate_audio_talker_direct() '
            'instead of the built-in model.generate() Talker, which produces cleaner audio. '
            'Requires the model-training scripts to be reachable.'
        ),
    )
    parser.add_argument(
        '--layer0-only',
        action='store_true',
        default=False,
        help=(
            'Use only layer0 (token embeddings) for all talker input positions. '
            'Matches training (no layer24 / hidden_projection for audio tokens). '
            'When omitted, the integrated model.generate() talker runs with layer24 '
            'for audio token positions (original behaviour).'
        ),
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = _get_args()
    model, processor = _load_model_processor(args)

    _speaker_embedding = None
    if args.ref_audio:
        print(f"Extracting speaker embedding from {args.ref_audio} ...")
        _speaker_embedding = _load_speaker_embedding(
            os.path.expanduser(args.ref_audio), device="cuda:0"
        )
        print(f"  Speaker embedding ready: {_speaker_embedding.shape}")

    _launch_demo(args, model, processor)
