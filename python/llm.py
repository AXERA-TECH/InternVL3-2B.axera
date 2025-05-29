import cv2
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from ml_dtypes import bfloat16
from axengine import InferenceSession
from tqdm import tqdm
from decord import VideoReader

def img_preprocess(img, input_size):
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_size, input_size))
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1).reshape(1, 3, input_size, input_size)
    return img

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image:np.array, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_height, orig_width,  = image.shape[:2]
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    # resized_img = image.resize((target_width, target_height))
    resized_img = cv2.resize(image, (target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        # split_img = resized_img.crop(box)
        split_img = resized_img[box[1]:box[3], box[0]:box[2]]
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        # thumbnail_img = image.resize((image_size, image_size))
        thumbnail_img = cv2.resize(image, (image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def pre_process(image, input_size=448, max_num=12):
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [img_preprocess(image, input_size) for image in images]
    pixel_values = np.concatenate(pixel_values, axis=0)
    return pixel_values

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


def load_video(video_path, bound=None, num_segments=32):
    vr = VideoReader(video_path, num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    images_list = []
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = vr[frame_index].asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images_list.append(img)
    return images_list

def is_video_file(path):
    return str(path).lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))

def is_image_file(path):
    return str(path).lower().endswith((".jpg", ".png", ".jpeg", ".webp"))

def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Image {path} not found or cannot be read.")
    return image

def post_process(data, topk=1, topp=0.9, temperature=0.6):
    def top_p(l: np.ndarray, p: float) -> np.ndarray:
        index = np.argsort(l)
        res = l.copy()
        sum_p = 0
        for i in index[::-1]:
            if sum_p >= p:
                res[i] = 0
            sum_p += res[i]
        return res / sum_p

    def softmax(l: np.ndarray) -> np.ndarray:
        l_max = l - l.max()
        l_exp = np.exp(l_max)
        res = l_exp / np.sum(l_exp)
        return res.astype(np.float64)

    r = data.astype(np.float32)
    r = r.flatten()
    # topk
    candidate_index = np.argpartition(r, -topk)[-topk:]
    candidate_value = r[candidate_index]
    # temperature
    candidate_value /= temperature
    # softmax
    candidate_soft = softmax(candidate_value)
    # topp
    candidate_soft = top_p(candidate_soft, topp)
    candidate_soft = candidate_soft.astype(np.float64) / candidate_soft.sum()
    pos = np.random.multinomial(1, candidate_soft).argmax()
    next_token = candidate_index[pos]
    return next_token, candidate_index, candidate_soft


class LLM:
  
    def __init__(self, hf_model_path, axmodel_path, vit_axmodel_path ):
        self.hf_model_path = hf_model_path
        
        
        config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True, use_fast=False)
        self.cfg = config.llm_config
        
        self.prefill_slice_len=128
        self.kv_cache_len=2559
        
        self.prefill_decoder_sessins = []
        for i in tqdm(range(self.cfg.num_hidden_layers), desc="Init InferenceSession"):
            session = InferenceSession(
                f"{axmodel_path}/qwen2_p128_l{i}_together.axmodel"
            )
            self.prefill_decoder_sessins.append(session)

        self.post_process_session = InferenceSession(
            f"{axmodel_path}/qwen2_post.axmodel"
        )
        print("model load done!")
        
        self.kv_dim = self.cfg.hidden_size // self.cfg.num_attention_heads * self.cfg.num_key_value_heads
    
            
        self.vit_session = InferenceSession(vit_axmodel_path)
        
        self.embeds = np.load(f"{axmodel_path}/model.embed_tokens.weight.npy")
        
        self.stop = False
    
    def stop_generate(self):
        self.stop = True

    def image_encode(self, images_list):
        pixel_values_list = []
        vit_output_list = []
        if images_list is not None:
            for img in images_list:
                pixel_values = pre_process(img, input_size=448, max_num=1)
                pixel_values_list.append(pixel_values)
            print(f"输入图像数: {len(pixel_values_list)}")
            print("preprocess image done!")

            # extract img feature by vit
            
            
            for idx, pixel_values in enumerate(pixel_values_list):
                vit_output = self.vit_session.run(None, {"image": pixel_values})[0]
                vit_output_list.append(vit_output.copy()) # 避免 vit 输出结果使用同一块内存

            print(f"vit_output.shape is {vit_output_list[0].shape}, vit feature extract done!")
        
        return vit_output_list

    def prompt_encode(self, question, num_of_images) -> list:
        prompt = "<|im_start|>system\n你是书生·万象, 英文名是InternVL, 是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型.<|im_end|>\n"
        # question = args.question
        prompt += "<|im_start|>user\n" + question

        if num_of_images > 0:
            for idx in range(num_of_images):
                prompt += "\n<img>" + "<IMG_CONTEXT>" * 256 + "</img>\n"
        
        prompt += "<|im_end|>\n<|im_start|>assistant"
        # print(f"prompt is {prompt}")
        token_ids = self.tokenizer.encode(prompt)
        print(len(token_ids))
        return token_ids
    

    def generate(self, sources, prompt, video_segments=8):
        self.stop = False
        images_list = []

        # 1. Handle single video path string
        if isinstance(sources, str) and is_video_file(sources):
            images_list = load_video(sources, num_segments=video_segments)

        # 2. Handle [video_path] list
        elif isinstance(sources, list) and len(sources) == 1 and isinstance(sources[0], str) and is_video_file(sources[0]):
            images_list = load_video(sources[0], num_segments=video_segments)

        # 3. Handle single image path
        elif isinstance(sources, str) and is_image_file(sources):
            images_list = [load_image(sources)]

        # 4. Handle single image as np.ndarray
        elif isinstance(sources, np.ndarray):
            images_list = [sources]

        # 5. Handle list of images or paths
        elif isinstance(sources, list):
            for img in sources:
                if isinstance(img, str):
                    images_list.append(load_image(img))
                elif isinstance(img, np.ndarray):
                    images_list.append(img)
                else:
                    raise ValueError(f"Unsupported image type: {type(img)}")
        else:
            raise ValueError("Unsupported input format for 'sources'.")
                    
        vit_output_list = self.image_encode(images_list)
        
        token_ids = self.prompt_encode(prompt, len(vit_output_list))
        
        k_caches = [
            np.zeros((1, self.kv_cache_len, self.kv_dim), dtype=bfloat16)
            for _ in range(self.cfg.num_hidden_layers)
        ]
        v_caches = [
            np.zeros((1, self.kv_cache_len, self.kv_dim), dtype=bfloat16)
            for _ in range(self.cfg.num_hidden_layers)
        ]

        # 图像理解
        image_start_indices = np.where(np.array(token_ids) == 151665)[0].tolist() # <img> tag
        
        prefill_data = np.take(self.embeds, token_ids, axis=0)
        prefill_data = prefill_data.astype(bfloat16)
        token_len = len(token_ids)

        assert token_len < 2048 + 128, f"输入 prompt({token_len}) 超过最大限度!"
        for idx, image_start_index in enumerate(image_start_indices):
            image_insert_index = image_start_index + 1
            prefill_data[image_insert_index : image_insert_index + 256] = vit_output_list[idx][0, :, :]
        ##################################
        print("prefill token_len: ", token_len)
        

        """
            prefill
        """
        prefill_slice_len = self.prefill_slice_len
        # slice_indexs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        slice_indexs = [
            e for e in range(token_len // prefill_slice_len + 1)
        ]
        # print(f"slice_indexs is {slice_indexs}")
        prefill_len = prefill_slice_len * slice_indexs[-1] if slice_indexs[-1] != 0 else prefill_slice_len # 这里的 128 就是 prefill_slice_len

        if prefill_len > 0:
            for slice_index in tqdm(slice_indexs, desc="prefill"):
                indices = np.array(
                    list(
                        range(
                            slice_index * prefill_slice_len,
                            (slice_index + 1) * prefill_slice_len,
                        )
                    ),
                    np.uint32,
                ).reshape((1, prefill_slice_len))

                mask = (
                    np.zeros((1, prefill_slice_len, prefill_slice_len * (slice_index + 1)))
                    - 65536
                )
                data = np.zeros((1, prefill_slice_len, self.cfg.hidden_size)).astype(bfloat16)
                for i, t in enumerate(
                    range(
                        slice_index * prefill_slice_len,
                        (slice_index + 1) * prefill_slice_len,
                    )
                ):
                    if t < len(token_ids):
                        mask[:, i, : slice_index * prefill_slice_len + i + 1] = 0
                        data[:, i : i + 1, :] = (
                            prefill_data[t]
                            .reshape((1, 1, self.cfg.hidden_size))
                            .astype(bfloat16)
                        )

                if slice_index == slice_indexs[-1]:
                    remain_len = token_len - slice_index * prefill_slice_len
                else:
                    remain_len = prefill_slice_len
                mask = mask.astype(bfloat16)
                for i in range(self.cfg.num_hidden_layers):
                    input_feed = {
                        "K_cache": (
                            k_caches[i][:, 0 : prefill_slice_len * slice_index, :]
                            if slice_index
                            else np.zeros((1, 1, self.cfg.hidden_size), dtype=bfloat16)
                        ),
                        "V_cache": (
                            v_caches[i][:, 0 : prefill_slice_len * slice_index, :]
                            if slice_index
                            else np.zeros((1, 1, self.cfg.hidden_size), dtype=bfloat16)
                        ),
                        "indices": indices,
                        "input": data,
                        "mask": mask,
                    }
                    outputs = self.prefill_decoder_sessins[i].run(None, input_feed, shape_group=slice_index + 1)
                    k_caches[i][
                        :,
                        slice_index
                        * prefill_slice_len : slice_index
                        * prefill_slice_len + remain_len,
                        :,
                    ] = outputs[0][:, :remain_len, :]
                    v_caches[i][
                        :,
                        slice_index
                        * prefill_slice_len : slice_index
                        * prefill_slice_len + remain_len,
                        :,
                    ] = outputs[1][:, :remain_len, :]
                    data = outputs[2]
                    
                    if self.stop:
                        return

                # print("slice prefill done", slice_index)
            post_out = self.post_process_session.run(
                None,
                {
                    "input": data[
                        :, token_len - (len(slice_indexs) - 1) * prefill_slice_len - 1, None, :
                    ]
                }
            )[0]
            next_token, posssible_tokens, possible_soft = post_process(post_out)
            posibles = [self.tokenizer.decode([t]) for t in posssible_tokens]
            posible_soft = [str((t, s)) for t, s in zip(posibles, possible_soft)]
            token_ids.append(next_token)

        # set to decoder
        token_ids_cached = []
        
        mask = np.zeros((1, 1, self.kv_cache_len + 1), dtype=np.float32).astype(bfloat16)
        mask[:, :, :self.kv_cache_len] -= 65536
        if prefill_len > 0:
            mask[:, :, :token_len] = 0
        for start_indice in range(self.kv_cache_len):
            if prefill_len > 0 and start_indice < token_len:
                continue

            next_token = token_ids[start_indice]
            indices = np.array([start_indice], np.uint32).reshape((1, 1))
            data = self.embeds[next_token, :].reshape((1, 1, self.cfg.hidden_size)).astype(bfloat16)
            for i in range(self.cfg.num_hidden_layers):
                input_feed = {
                    "K_cache": k_caches[i],
                    "V_cache": v_caches[i],
                    "indices": indices,
                    "input": data,
                    "mask": mask,
                }
                outputs = self.prefill_decoder_sessins[i].run(None, input_feed, shape_group=0)
                k_caches[i][:, start_indice, :] = outputs[0][:, :, :]
                v_caches[i][:, start_indice, :] = outputs[1][:, :, :]
                data = outputs[2]
            mask[..., start_indice] = 0
            if start_indice < token_len - 1:
                pass
            else:
                post_out = self.post_process_session.run(None, {"input": data})[0]
                next_token, posssible_tokens, possible_soft = post_process(post_out)
                token_ids.append(next_token)
                
                if next_token == self.tokenizer.eos_token_id and next_token > token_len:
                    if len(token_ids_cached) > 0:
                        msg = self.tokenizer.decode(token_ids_cached)
                        token_ids_cached.clear()
                        if "\ufffd" in msg:
                            msg = msg.replace("\ufffd", "")
                        # print(msg, end="", flush=True)
                        yield msg
                    break
                
                token_ids_cached.append(next_token)
                
                if len(token_ids_cached) >= 3:
                    msg = self.tokenizer.decode(token_ids_cached)
                    token_ids_cached.clear()
                    if "\ufffd" in msg:
                        msg = msg.replace("\ufffd", "")
                    # print(msg, end="", flush=True)
                    yield msg
                    
                
            if self.stop:
                return
