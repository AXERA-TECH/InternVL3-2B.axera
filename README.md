# InternVL3-2B.axera

> Deepseek InternVL3-2B DEMO on Axera.

- 目前支持 `Python` 语言, `C++` 代码在开发中.
- 预编译模型可以从 [百度网盘](https://pan.baidu.com/s/11BM4Zkf5ThwUvA17TeQc8A?pwd=x3my) 下载.
- 如需自行导出编译 `VIT` 模型请参考 [模型转换](/model_convert/README.md).

## 支持平台

- [x] AX650N
- [ ] AX630C

## Git Clone

首先使用如下命令 `clone` 本项目, 然后进入 `python` 文件夹:

```bash
$ git clone git@github.com:AXERA-TECH/InternVL3-2B.axera.git
$ cd InternVL3-2B.axera/python
```

之后在开发板上下载或安装以下支持库:

- 从 `huggingface` 下载 `InternVL3-2B` 模型.

    ```bash
    $ git clone https://huggingface.co/OpenGVLab/InternVL3-2B
    ```

- 在开发板上安装配置 `pyaxengine`, [点击跳转下载链接](https://github.com/AXERA-TECH/pyaxengine/releases). 注意板端 `SDK` 最低版本要求:

    - AX650 SDK >= 2.18
    - AX620E SDK >= 3.12
    - 执行 `pip3 install axengine-x.x.x-py3-none-any.whl` 安装

将下载后的预编译模型解压到当前文件夹[🔔可选], 默认文件夹排布如下:

```bash
.
├── examples
│   ├── image_0.jpg
│   ├── image_1.jpg
│   ├── image_2.png
│   ├── image_3.png
│   └── red-panda.mp4
├── infer.py
├── InternVL3-2B
│   ├── added_tokens.json
│   ├── config.json
│   ├── configuration_intern_vit.py
│   ├── configuration_internvl_chat.py
│   ├── conversation.py
│   ├── examples
│   ├── generation_config.json
│   ├── merges.txt
│   ├── modeling_intern_vit.py
│   ├── modeling_internvl_chat.py
│   ├── model.safetensors
│   ├── preprocessor_config.json
│   ├── README.md
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.json
├── InternVL3-2B_axmodel_chunk_128
│   ├── model.embed_tokens.weight.npy
│   ├── qwen2_p128_l0_together.axmodel
│   ├── qwen2_p128_l10_together.axmodel
│   ├── qwen2_p128_l11_together.axmodel
│   ├── qwen2_p128_l12_together.axmodel
│   ├── qwen2_p128_l13_together.axmodel
│   ├── qwen2_p128_l14_together.axmodel
│   ├── qwen2_p128_l15_together.axmodel
│   ├── qwen2_p128_l16_together.axmodel
│   ├── qwen2_p128_l17_together.axmodel
│   ├── qwen2_p128_l18_together.axmodel
│   ├── qwen2_p128_l19_together.axmodel
│   ├── qwen2_p128_l1_together.axmodel
│   ├── qwen2_p128_l20_together.axmodel
│   ├── qwen2_p128_l21_together.axmodel
│   ├── qwen2_p128_l22_together.axmodel
│   ├── qwen2_p128_l23_together.axmodel
│   ├── qwen2_p128_l24_together.axmodel
│   ├── qwen2_p128_l25_together.axmodel
│   ├── qwen2_p128_l26_together.axmodel
│   ├── qwen2_p128_l27_together.axmodel
│   ├── qwen2_p128_l2_together.axmodel
│   ├── qwen2_p128_l3_together.axmodel
│   ├── qwen2_p128_l4_together.axmodel
│   ├── qwen2_p128_l5_together.axmodel
│   ├── qwen2_p128_l6_together.axmodel
│   ├── qwen2_p128_l7_together.axmodel
│   ├── qwen2_p128_l8_together.axmodel
│   ├── qwen2_p128_l9_together.axmodel
│   └── qwen2_post.axmodel
├── requirements.txt
└── vit_axmodel
    └── internvl3_2b_vit_slim.axmodel

5 directories, 54 files
```

## 模型转换

关于 `onnx` 和 `axmodel` 的导出、编译参见 [模型转换](./model_convert/README.md) 部分内容.

## 上板部署

- `AX650N` 的设备已预装 `Ubuntu 22.04`
- 以 `root` 权限登陆 `AX650N` 的板卡设备
- 接入互联网, 确保 `AX650N` 的设备能正常执行 `apt install`, `pip install` 等指令
- 已验证设备: `AX650N DEMO Board`、`爱芯派Pro(AX650N)`

### Python API 运行

#### Requirements

```bash
$ mkdir /opt/site-packages
$ cd python
$ pip3 install -r requirements.txt --prefix=/opt/site-packages
``` 

#### 添加环境变量

将以下两行添加到 `/root/.bashrc`(实际添加的路径需要自行检查)后, 重新连接终端或者执行 `source ~/.bashrc`

```bash
$ export PYTHONPATH=$PYTHONPATH:/opt/site-packages/local/lib/python3.10/dist-packages  
$ export PATH=$PATH:/opt/site-packages/local/bin
``` 

#### 运行

在 `Axera 开发板` 上运行以下命令开始聊天对话:

```sh
$ cd InternVL3-2B.axera/python
$ python3 infer.py --hf_model InternVL3-2B/ --axmodel_path InternVL3-2B_axmodel_chunk_128/ --question "请计算函数[y=2x^2+2]的导数, 并提供 markdown 格式的推理过程"
```

输出结果如下:

```bash
Init InferenceSession: 100%|██████████████████████████████████████████████████████████| 28/28 [00:21<00:00,  1.32it/s]
model load done!
prefill token_len:  95
slice_indexs is [0]
slice prefill done 0
Decode:   6%|████                                                                  | 150/2559 [00:07<02:27, 16.36it/s]Decode:  23%|████████████████▍                                                     | 600/2559 [01:09<04:33,  7.17it/s]Decode:  26%|█████████████████▉                                                    | 655/2559 [01:17<04:28,  7.10it/s]hit eos!
Decode:  26%|█████████████████▉                                                    | 655/2559 [01:17<03:44,  8.49it/s]
当然可以。我们来计算函数 \( y = 2x^2 + 2 \) 的导数。

首先，我们使用导数的定义，即对函数 \( y \ \) 关于 \( x \ \) 的导数 \( y' \ \) 是 \( y \ \) 关于 \( x \ \) 的变化率。导数的计算公式是：

\[ y' = \\frac{d}{dx}(2x^2 + 2) \ \]

根据导数的加法法则，我们可以将导数拆分为两部分：

\[ y' = \ \\frac{d}{dx}(2x^2) + \ \frac{d}{dx}(2) \ \]

接下来，我们分别计算这两部分的导数：

1. 对于 \( 2x^2 \ \\)，我们使用幂法则 \( \ \\frac{d}{dx}(x^n) = n \x^{n-1} \ \\)：

\[ \ \\frac{d}{dx}(2x^2) = 2 \ \ \frac{d}{dx}(x^2) = 2 \ \ x^{2-1} = 2 \x \ \]

2. 对于常数 \( 2 \ \)，其导数是 \( 0 \ \)：

\[ \ \frac{d}{dx}(2) = 0 \ \]

将这两部分的结果相加，我们得到：

\[ y' = 2x + 0 = 2x \ \]

因此，函数 \( y = 2x^2 + 2 \ \) 的导数是 \( y' = 2x \ \\)。

下面是这个过程的markdown格式：

    ```markdown
    ## 计算函数 \( y = 2x^2 + 2 \) 的导数

    1. **计算 \( \2x^2 \ \\) 的导数**：
    - 使用幂法则：\( \ \\frac{d}{dx}(x^n) = n x^{n-1} \ \\)。
    - \( \2x^2 \ \\) 的导数为 \( 2x \ \)。

    2. **计算常数 \( 2 \ \\) 的导数**：
    - 常数的导数为 \( 0 \ \\)。

    3. **将两部分导数相加**：
    - \( y' = 2x + 0 = 2x \ \\)。

    ```
因此，函数 \( y = 2x^2 + 2 \ \\) 的导数是 \( y' = 2x \ \\)。
```

输入以下命令执行图像理解任务:

```sh
$ cd InternVL3-2B.axera/python
$ python3 infer.py --hf_model InternVL3-2B/ --axmodel_path InternVL3-2B_axmodel_chunk_128/ -q "请分别描述这几幅图像的内容, 并找出它们的异同点" -i examples/image_0.jpg examples/image_1.jpg examples/image_2.png examples/image_3.png
```

此模型最多支持四幅图像作为输入:

![image_0.jpg](python/examples/image_0.jpg)

![image_1.jpg](python/examples/image_1.jpg)

![image_2.png](python/examples/image_2.png)

![image_3.png](python/examples/image_3.png)

模型推理结果如下:

```bash
[INFO] Chip type: ChipType.MC50
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Engine version: 2.11.0a
Init InferenceSession: 100%|██████████████████████████████████████████████████████████| 28/28 [00:19<00:00,  1.46it/s]
model load done!
prefill token_len:  1123
slice_indexs is [0, 1, 2, 3, 4, 5, 6, 7, 8]
slice prefill done 0
slice prefill done 1
slice prefill done 2
slice prefill done 3
slice prefill done 4
slice prefill done 5
slice prefill done 6
slice prefill done 7
slice prefill done 8
Decode:  47%|████████████████████████████████▊                                    | 1215/2559 [00:12<00:19, 68.19it/s]hit eos!
Decode:  52%|████████████████████████████████████▏                                | 1341/2559 [00:30<00:27, 43.74it/s]
这张图片中，第一张图是一只红熊猫，它正趴在一块木头上，背景是绿色的树叶和树干。第二张图是一只大熊猫，它正坐在地上，用前爪抓着竹子，周围是绿色的竹子和地面。第三张图是三个穿着宇航服的人，他们站在一片森林中，背景是树木和植被。第四张图是一位动漫风格的女性角色，她有着银灰色的长发，头上戴着粉色的花朵，背景是海滩和海洋。

异同点：
- **相同点**：四张图片中的人物或动物都处于自然环境中，且都有明显的特征。
- **不同点**：
  - 第一张图是一只红熊猫，第二张图是一只大熊猫，第三张图是三个宇航员，第四张图是一位动漫风格的女性角色。
  - 第一张图的动物是红熊猫，第三张图的宇航员穿着宇航服，第四张图的女性角色是动漫风格，具有不同的艺术风格和特征。
```

#### 图像理解任务·推理耗时统计

Model | Time |
---| ---|
ImageEncoder | 364.870 ms |
Prefill TTFT | 4588.79 ms |
Decoder | 86.969 ms |

128 chunk prefill 推理, decode layer 耗时 2.686 ms * 28, post 耗时 11.455 ms.

该模型 prefill 阶段存在 9 个可用子图, 每个子图耗时如下:

```
g1: 7.483 ms
g2: 10.089 ms
g3: 12.815 ms
g4: 15.235 ms
g5: 18.527 ms
g6: 20.751 ms
g7: 23.520 ms
g8: 25.932 ms
g9: 29.124 ms
```

prefill 阶段最大 TTFT 为: (g1 + ··· + g9) * 28 + 11.455 = 163.476 * 28 + 11.455 = 4588.79 ms.

模型解码速度为: 1000 / 86.969 ms = 11.50 tokens/s.

---

固定 320 prefill 推理, prefill 每一层耗时 28.258 ms, 一共 28 层, decode 耗时 2.510 ms, post 耗时 11.761 ms.

## 技术讨论

- Github issues
- QQ 群: 139953715
