import torch
import argparse
from loguru import logger
import os
import onnx
import onnxruntime
from onnx import TensorProto
from onnx.shape_inference import infer_shapes
from onnxsim import simplify
import numpy as np
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from collections import OrderedDict


"""
Apr 17, 2025. 需要使用 onnxslim 替换 onnxsim
"""

def hack_fuse(onnx_path, dummy_input):
    logger.warning("Hack fuse: 融合 resize_pos_embeddings 结构.")

    def del_node(graph, node_list):
        rm_node_list = node_list
        if isinstance(rm_node_list[0], str):
            rm_node_list = []
            for node in graph.node:
                if node.name in node_list:
                    rm_node_list.append(node)
        for rnl in rm_node_list:
            graph.node.remove(rnl) # rln, remove node list
        return graph

    model = onnx.load(onnx_path)
    inputs = [node.name for node in model.graph.input]
    input_tensors = {}
    input_tensors[inputs[0]] = dummy_input
    inputs = [node.name for node in model.graph.input]
    # ori_output = copy.deepcopy(model.graph.output)
    for node in model.graph.node:
        for output in node.output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
    ort_session = onnxruntime.InferenceSession(model.SerializeToString())
    outputs = [x.name for x in ort_session.get_outputs()]
    ort_outs = ort_session.run(outputs, input_tensors)
    ort_outs = OrderedDict(zip(outputs, ort_outs))
    add_b_tensor = ort_outs['/vision_model/embeddings/Concat_2_output_0']

    del_list = [ # TODO: pulsar2 目前不支持 Resize cubic mode, 所以手动将 constant 结果融合
        "/vision_model/embeddings/Resize",
        "/vision_model/embeddings/Reshape_1",
        "/vision_model/embeddings/Transpose_1",
        "/vision_model/embeddings/Concat_2"
    ]
    model = onnx.load(onnx_path)
    graph = model.graph

    graph = del_node(graph, del_list)
    # graph.initializer.pop('/vision_model/embeddings/Concat_2_output_0')
    add_b_tensor_init = onnx.helper.make_tensor(
        '/vision_model/embeddings/Concat_2_output_0',
        TensorProto.FLOAT,
        dims=add_b_tensor.shape,
        vals=add_b_tensor
    )
    graph.initializer.append(add_b_tensor_init)
    graph = onnx.helper.make_graph(graph.node, graph.name, graph.input, graph.output, graph.initializer)
    # create & check model
    info_model = onnx.helper.make_model(graph)
    onnx_model = onnx.shape_inference.infer_shapes(info_model)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, onnx_path)
    logger.info(f"onnx hack fuse successed, and model saved in {onnx_path}")


def onnx_sim(onnx_path):
    onnx_model = onnx.load(onnx_path)
    onnx_model = infer_shapes(onnx_model)
    # convert model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_path)
    logger.info(f"onnx simpilfy successed, and model saved in {onnx_path}")


class VisionModelWarpper(nn.Module):

    def __init__(self, model, config):
        super().__init__()

        self.downsample_ratio = config.downsample_ratio
        self.vision_model = model.vision_model
        self.mlp1 = model.mlp1

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, pixel_values):
        vit_embeds = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True).last_hidden_state
        vit_embeds = vit_embeds[:, 1:, :]
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds


if __name__ == '__main__':

    """
    Usage:
        python3 export_onnx.py -m /path/your/hugging_face/models/Janus-Pro-1B/ -o ./vit-models
    """
    parser = argparse.ArgumentParser(prog='main')
    parser.add_argument("-m", "--model", type=str, help="hugging fance model path")
    parser.add_argument("--name", type=str, default=None, help="onnx name")
    parser.add_argument("--imgsize", type=int, default=448, help="onnx input image size")
    parser.add_argument("-o", "--onnx_save_dir", type=str, default='./vit-models', help="vit onnx model save path")
    args = parser.parse_args()

    model_path = args.model
    onnx_save_dir = args.onnx_save_dir

    if not os.path.exists(onnx_save_dir):
        os.makedirs(onnx_save_dir)

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()

    IMG_SIZE = args.imgsize
    images = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device="cuda", dtype=torch.float32)

    cfg = AutoConfig.from_pretrained(
        model_path, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    internvl_vit_onnx_save_dir = os.path.join(
        onnx_save_dir,
        f'internvl_vit_model_1x3x{IMG_SIZE}x{IMG_SIZE}.onnx' if args.name is None else args.name
    )

    # vision_model.encoder.layers = vision_model.encoder.layers[:1] # debug 24
    vision_model_warpper = VisionModelWarpper(model, cfg).to(device="cuda", dtype=torch.float32)
    torch.onnx.export(
        vision_model_warpper,
        images,
        internvl_vit_onnx_save_dir,
        opset_version=17, # 14
        do_constant_folding=True,
        verbose=False,
        input_names=["image"],
        output_names=["output"],
    )

    logger.info("export internvl_vit_model onnx succee!")
    onnx_sim(internvl_vit_onnx_save_dir)
    # hack some nodes
    hack_fuse(internvl_vit_onnx_save_dir, images.cpu().numpy())
    logger.warning("Use onnxslim to fine-tune the ONNX model. Please ensure onnxslim is installed first.")
    import subprocess
    result = subprocess.run(
        ["onnxsim", internvl_vit_onnx_save_dir, internvl_vit_onnx_save_dir],
        capture_output=True,
        text=True
    )
    logger.info(result.stdout)

