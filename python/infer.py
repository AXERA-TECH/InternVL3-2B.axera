# import llm_utils
import dataclasses
import json
from transformers import AutoTokenizer, AutoConfig
import torch
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from ml_dtypes import bfloat16
from axengine import InferenceSession
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image
import argparse


"""
pulsar2 llm_build \
    --input_path ./InternVL3-2B   \
    --output_path ./InternVL3-2B_axmodel \
    --hidden_state_type bf16 \
    --prefill_len 128 \
    --last_kv_cache_len 128 \
    --last_kv_cache_len 256 \
    --last_kv_cache_len 384 \
    --last_kv_cache_len 512 \
    --last_kv_cache_len 640 \
    --last_kv_cache_len 768  \
    --last_kv_cache_len 896 \
    --last_kv_cache_len 1024 \
    --last_kv_cache_len 1536 \
    --last_kv_cache_len 2048 \
    --kv_cache_len 2559 \
    --chip AX650 -c 1 --parallel 28

"""


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

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

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
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
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

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


if __name__ == "__main__":

    prompt = None
    parser = argparse.ArgumentParser(description="Model configuration parameters")
    parser.add_argument("--hf_model", type=str, default="./InternVL3-2B",
                        help="Path to HuggingFace model")
    parser.add_argument("--axmodel_path", type=str, default="./InternVL3-2B_axmodel",
                        help="Path to save compiled axmodel of llama model")
    parser.add_argument("-i", "--image", type=str, default="./examples/image1.jpg",
                        help="Path to the test image.")
    args = parser.parse_args()


    # If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
    hf_model_path = args.hf_model
    axmodel_path = args.axmodel_path
    test_img_path = args.image # './examples/image1.jpg' # image.png, image1.jpg
    vit_axmodel_path = "internvl3_2b_vit_slim.axmodel"

    config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True, use_fast=False)
    # tokenizer = AutoTokenizer.from_pretrained("internlm2_5-7b-chat", trust_remote_code=True, use_fast=False)
    # set the max number of tiles in `max_num`
    pixel_values = load_image(test_img_path, input_size=448, max_num=1)
    print("preprocess image done!")

    # generation_config = dict(max_new_tokens=1024, do_sample=True)
    # extract img feature by vit
    vit_session = InferenceSession(vit_axmodel_path)
    vit_output = vit_session.run(None, {"image": pixel_values.numpy()})[0]
    print(f"vit_output.shape is {vit_output.shape}, vit feature extract done!")

    prompt = "<|im_start|>system\n你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型, 英文名叫 InternVL2_5, 是一个有用无害的人工智能助手.<|im_end|><|im_start|>user\n<img>"
    prompt += "<IMG_CONTEXT>" * 256
    # prompt += "<image>" * 256
    # question = "请告诉我 y = 2x^2 + 3 的导数是多少? 告诉我详细的步骤!"
    question = "请详细描述这幅图像."
    # question = "Please describe the image shortly."
    prompt += "</img>\n<|im_end|>" + question + "<|im_start|>assistant\n"
    question = "请将这些英文翻译成中文: `She has always been there for me, through the good times and the bad. She has taught me so much about love and what it means to be a good person. Growing up, my mother was always my rock. She was the one I would go to for advice.`"
    prompt += "</img>\n<|im_end|>" + question + "<|im_start|>assistant\n"

    # prompt = "<|im_start|>system\n你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型, 英文名叫 InternVL2_5, 是一个有用无害的人工智能助手.<|im_end|><|im_start|>user\n<|im_end|>请告诉我 y = 2x^2 + 3 的导数是多少? 告诉我详细的步骤!.<|im_start|>assistant\n"
    token_ids = tokenizer.encode(prompt)

    # 图像理解
    image_start_index = np.where(np.array(token_ids) == 151665)[0].tolist()[0] # <img> tag
    image_insert_index = image_start_index + 1
    embeds = np.load(f"{axmodel_path}/model.embed_tokens.weight.npy")
    prefill_data = np.take(embeds, token_ids, axis=0)
    prefill_data = prefill_data.astype(bfloat16)
    prefill_data[image_insert_index : image_insert_index + 256] = vit_output[0, :, :]
    token_len = len(token_ids)
    assert token_len > 128 * 3 # TODO: 如果缺少这个条件, 会报错!
    ##################################

    lastN = 2559
    cfg = config.llm_config
    # cfg = config
    # cfg.num_hidden_layers = 24

    kv_dim = cfg.hidden_size // cfg.num_attention_heads * cfg.num_key_value_heads
    k_caches = [
        np.zeros((1, lastN, kv_dim), dtype=bfloat16)
        for _ in range(cfg.num_hidden_layers)
    ]
    v_caches = [
        np.zeros((1, lastN, kv_dim), dtype=bfloat16)
        for _ in range(cfg.num_hidden_layers)
    ]

    prefill_decoder_sessins = []
    for i in tqdm(range(cfg.num_hidden_layers), desc="Init InferenceSession"):
        session = InferenceSession(
            f"{axmodel_path}/qwen2_p128_l{i}_together.axmodel"
        )
        prefill_decoder_sessins.append(session)

    post_process_session = InferenceSession(
        f"{axmodel_path}/qwen2_post.axmodel"
    )
    print("model load done!")
    print("prefill token_len: ", token_len)

    """
        prefill
    """

    slice_indexs = [0, 1, 2, 3]
    prefill_len = 128 * slice_indexs[-1]
    # prefill_len = 2048
    if prefill_len > 0:
        prefill_slice_len = 128
        for slice_index in slice_indexs:

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
            data = np.zeros((1, prefill_slice_len, cfg.hidden_size)).astype(bfloat16)
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
                        .reshape((1, 1, cfg.hidden_size))
                        .astype(bfloat16)
                    )
                    
            if slice_index == slice_indexs[-1]:
                remain_len = token_len - slice_index * prefill_slice_len
            else:
                remain_len = prefill_slice_len
            mask = mask.astype(bfloat16)
            for i in range(cfg.num_hidden_layers):
                input_feed = {
                    "K_cache": (
                        k_caches[i][:, 0 : prefill_slice_len * slice_index, :]
                        if slice_index
                        else np.zeros((1, 1, cfg.hidden_size), dtype=bfloat16)
                    ),
                    "V_cache": (
                        v_caches[i][:, 0 : prefill_slice_len * slice_index, :]
                        if slice_index
                        else np.zeros((1, 1, cfg.hidden_size), dtype=bfloat16)
                    ),
                    "indices": indices,
                    "input": data,
                    "mask": mask,
                }
                outputs = prefill_decoder_sessins[i].run(None, input_feed, shape_group=slice_index + 1)
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

            print("slice prefill done", slice_index)

        post_out = post_process_session.run(
            None,
            {
                "input": data[
                    :, token_len - (len(slice_indexs) - 1) * prefill_slice_len - 1, None, :
                ]
            }
        )[0]
        next_token, posssible_tokens, possible_soft = post_process(post_out)
        print(tokenizer.decode([next_token]))
        posibles = [tokenizer.decode([t]) for t in posssible_tokens]
        posible_soft = [str((t, s)) for t, s in zip(posibles, possible_soft)]
        print(f"posibile {','.join(posible_soft)}")
        token_ids.append(next_token)
        print(tokenizer.decode(token_ids))

    # set to decoder
    kv_cache_len = 2559
    mask = np.zeros((1, 1, kv_cache_len + 1), dtype=np.float32).astype(bfloat16)
    mask[:, :, :kv_cache_len] -= 65536
    if prefill_len > 0:
        mask[:, :, :token_len] = 0
    for start_indice in range(kv_cache_len + 1):
        if prefill_len > 0 and start_indice < token_len:
            continue
        # print(start_indice, "start_indice")
        next_token = token_ids[start_indice]
        indices = np.array([start_indice], np.uint32).reshape((1, 1))
        data = embeds[next_token, :].reshape((1, 1, cfg.hidden_size)).astype(bfloat16)
        for i in range(cfg.num_hidden_layers):
            input_feed = {
                "K_cache": k_caches[i],
                "V_cache": v_caches[i],
                "indices": indices,
                "input": data,
                "mask": mask,
            }
            outputs = prefill_decoder_sessins[i].run(None, input_feed, shape_group=0)
            k_caches[i][:, start_indice, :] = outputs[0][:, :, :]
            v_caches[i][:, start_indice, :] = outputs[1][:, :, :]
            data = outputs[2]
        mask[..., start_indice] = 0
        if start_indice < token_len - 1:
            pass
        else:
            post_out = post_process_session.run(None, {"input": data})[0]
            next_token, posssible_tokens, possible_soft = post_process(post_out)
            token_ids.append(next_token)
            if next_token == tokenizer.eos_token_id and next_token > token_len:
                print("hit eos!")
                break

    # print result
    print(tokenizer.decode(token_ids))
