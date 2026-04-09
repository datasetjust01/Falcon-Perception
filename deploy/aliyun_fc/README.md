# Falcon-Perception

我已经按阿里云 `Function Compute GPU + Custom Container + 实时推理` 的思路，把这套项目的部署资产补上了，并顺手处理了一个对 Serverless 很关键的问题：默认高分辨率缓存会预分配大量 pinned host memory，不适合 FC。

已新增/修改：

- `Dockerfile.aliyun-fc-gpu`
- `.dockerignore`
- `deploy/aliyun_fc/entrypoint.sh`
- `deploy/aliyun_fc/prewarm.py`
- `falcon_perception/server/config.py`
- `falcon_perception/server/engine_worker.py`

核心依据如下。

```55:69:falcon_perception/server/__main__.py
def main(config: ServerConfig):
    import uvicorn

    from falcon_perception.server.app import create_app
    ...
    app = create_app(config)
    ...
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")
```

```198:210:falcon_perception/server/app.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    nonlocal pool
    pool = WorkerPool(config)
    pool.start_all()
    logger.info("Waiting for %d engine(s) to initialize ...", len(pool.workers))
    pool.wait_ready(timeout=config.startup_timeout)
    ...
    app.state.pool = pool
```

```248:296:falcon_perception/server/engine_worker.py
def _warmup_engine(engine, log):
    ...
    log.info("Running warmup inference ...")
    ...
    engine.generate([seq], sampling_params=sampling, use_tqdm=False, print_stats=False)
```

```301:334:falcon_perception/server/app.py
@app.get("/v1/health", response_model=HealthResponse)
async def health():
    if pool is None or not pool.any_ready:
        return HealthResponse(status="loading", num_gpus=0)
    ...
    return HealthResponse(
        status="ready",
        num_gpus=len(gpus),
        ...
    )
```

这说明项目本身已经有：
- 标准 API server；
- 启动阶段阻塞等待 worker ready；
- worker 内部 warmup；
- `/v1/health` readiness 探针。

所以在阿里云 FC 上，最合适的不是重写服务，而是做 3 件事：

1. 做 `GPU Custom Container` 镜像；
2. 用入口脚本把 `host/port/GPU/model path` 固定好；
3. 用 Initializer 执行一个“等健康 + 发最小推理请求”的预热脚本。

我已经把这三件事落成文件了。

---

## 1) 推荐的部署方案

### 推荐：镜像不内置模型，模型走挂载目录
这个项目启动时会从 `HF_LOCAL_DIR` 或 HF Hub 加载 `model.safetensors`、`config.json`、tokenizer 文件：

```253:291:falcon_perception/__init__.py
def load_from_hf_export(
    *,
    hf_model_id: str | None = None,
    hf_revision: str = "main",
    hf_local_dir: str | None = None,
) -> tuple[Any, Any, ModelArgs]:
    ...
    export_dir = Path(hf_local_dir) if hf_local_dir else Path(
        snapshot_download(...)
    )
    ...
    state = safetensors_load_file(str(export_dir / "model.safetensors"))
```

结合阿里云文档，针对这个项目我建议：

- 不要把 `Falcon-Perception` 权重直接烘进镜像；
- 用 FC 的存储挂载，把模型目录挂到比如 `/mnt/auto/falcon-perception`；
- 设置环境变量 `HF_LOCAL_DIR=/mnt/auto/falcon-perception`。

原因：
- 模型大，镜像会非常重；
- 更新模型要重打镜像；
- 实时 API 场景里，镜像越大，冷启动越差；
- 这个项目加载的是单个 `safetensors` 导出目录，非常适合走挂载目录。

如果你必须离线、且模型版本极少变化，才考虑把模型一起打进镜像。

---

## 2) 我补好的部署文件

### `Dockerfile`
```1:51:Dockerfile.aliyun-fc-gpu
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
...
RUN python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 "torch>=2.11.0" "torchvision>=0.26.0" \
    && python -m pip install "huggingface_hub>=0.31.0" ".[server]"
...
ENTRYPOINT ["/app/deploy/aliyun_fc/entrypoint.sh"]
```

这个镜像做了几件事：

- 用 `CUDA 12.8` 基础镜像；
- 按项目的 `cu128` 约束装 `torch/torchvision`；
- 装 `server` 依赖；
- 设置健康检查；
- 入口改为我们自己的 `entrypoint.sh`。

### 入口脚本
```1:63:deploy/aliyun_fc/entrypoint.sh
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export ENABLE_HR_CACHE="${ENABLE_HR_CACHE:-0}"
...
cmd=(
  python -m falcon_perception.server
  --config.host "${SERVER_HOST}"
  --config.port "${PORT}"
  --config.num-gpus "${NUM_GPUS}"
  --config.startup-timeout "${STARTUP_TIMEOUT}"
  --config.images-dir "${IMAGES_DIR}"
)
...
exec "${cmd[@]}"
```

它解决了几个 FC 部署问题：

- 强制监听 `0.0.0.0`；
- 固定 `PORT`；
- 支持 `HF_LOCAL_DIR` 或 `HF_MODEL_ID`；
- 默认把 `ENABLE_HR_CACHE=0`，更适合 Serverless；
- `IMAGES_DIR` 指到容器内可写目录。

### 预热脚本
```1:111:deploy/aliyun_fc/prewarm.py
while time.time() - start < timeout_s:
    ...
    if status_code == 200 and health.get("status") == "ready":
        ...
task = pick_task(health.get("supported_tasks", []))
payload = build_payload(task)
...
status_code, resp = http_json(predict_url, payload=payload, ...)
```

它会：

- 先轮询 `/v1/health`；
- ready 后自动判断可用 task；
- 发一个 `64x64` 的最小推理请求；
- 让 FC 的 Initializer 在“实例可接流量前”就完成预热。

---

## 3) 我额外修掉的一个关键坑：HR cache

这个项目的分割引擎默认会为高分辨率特征缓存预分配 pinned memory：

```399:410:falcon_perception/paged_inference.py
if enable_hr_cache and hasattr(model, "itok_upsampler"):
    _feat_dim = model.args.segm_out_dim
    _max_hw = max_image_size
    _bytes_per = model.dtype.itemsize
    _buf_bytes = _feat_dim * _max_hw * _max_hw * _bytes_per
    _pool_mb = max_hr_cache_entries * _buf_bytes / (1024 ** 2)
    ...
    self._hr_pinned_pool: list[Tensor] = [
        torch.empty(_feat_dim, _max_hw, _max_hw, dtype=model.dtype, pin_memory=True)
```

这对常驻大机器没问题，但对 FC GPU Serverless 很危险。  
所以我给 server 加了两个配置项，并把它们接到了引擎构造里：

```37:55:falcon_perception/server/config.py
dtype: Literal["bfloat16", "float32"] = ...
...
enable_hr_cache: bool = field(default_factory=lambda: _env_bool("ENABLE_HR_CACHE", True))
max_hr_cache_entries: int = field(default_factory=lambda: _env_int("MAX_HR_CACHE_ENTRIES", 100))
```

```223:238:falcon_perception/server/engine_worker.py
engine = PagedInferenceEngine(
    ...
    enable_hr_cache=config.enable_hr_cache,
    max_hr_cache_entries=config.max_hr_cache_entries,
    max_image_size=config.max_image_size,
    ...
)
```

然后在 `entrypoint.sh` 里默认：

- `ENABLE_HR_CACHE=0`

这对 FC 更合理，因为实时 API 通常是“一次图像一次请求”，不是对同一张图反复做多轮分割。

---

## 4) 你在阿里云 FC 控制台里应这样配

### 函数类型
- `GPU 函数`
- 运行方式：`Custom Container`

### 镜像
- 上传到阿里云 ACR
- 使用 `Dockerfile.aliyun-fc-gpu` 构建

### 监听端口
- `7860`

### 触发器
- `HTTP Trigger`

### 实例相关
- `最小实例数 >= 1`
- `单实例并发 = 1`
- 如果业务稳定且对延迟敏感，优先配预留/常驻实例
- 如果是弹性 GPU，建议开启文档里的预热/浅休眠相关能力

### Initializer
类型选“执行指令”，命令填：

```bash
python /app/deploy/aliyun_fc/prewarm.py
```

### Initializer timeout
- 至少 `600`
- 我更建议 `900`

### 请求超时
- 至少 `120`
- 如果图片较大或 query 较复杂，可到 `300`

### 环境变量建议
建议至少配这些：

```bash
PORT=7860
SERVER_HOST=0.0.0.0
NUM_GPUS=1
DTYPE=bfloat16
STARTUP_TIMEOUT=900
IMAGES_DIR=/tmp/falcon-perception/images
ENABLE_HR_CACHE=0
HF_LOCAL_DIR=/mnt/auto/falcon-perception
```

如果你暂时不挂载模型，而是在线拉 HF，再加：

```bash
HF_MODEL_ID=tiiuae/Falcon-Perception
HF_REVISION=main
HF_HUB_ENABLE_HF_TRANSFER=1
```

但生产上我不建议在线拉 HF。

---

## 5) 构建与推送

本地构建：

```bash
docker build -f Dockerfile.aliyun-fc-gpu -t <your-acr-registry>/falcon-perception-fc:latest .
```

推送：

```bash
docker push <your-acr-registry>/falcon-perception-fc:latest
```

---

## 6) 两个最重要的结论

### 结论 A：这套项目本身已经有 server warmup，不需要重写 API
因为它在启动时已经：
- 启动 GPU worker；
- `wait_ready(...)`；
- 每个 worker 做 `_warmup_engine(...)`；
- 再通过 `/v1/health` 对外暴露 ready。

所以 FC Initializer 主要是“平台级预热闭环”，不是替代模型 warmup。

### 结论 B：模型存储建议“挂载目录 + `HF_LOCAL_DIR`”，不要直接塞镜像
对这个项目，阿里云文档里最契合的做法是：
- 模型走 OSS/NAS 挂载；
- 镜像只放代码和运行时；
- Initializer 做健康等待 + 最小推理请求。

已继续补好了可直接改的部署模板：

- `deploy/aliyun_fc/s.yaml`
- `deploy/aliyun_fc/provision.json`

关键内容在这里：

```1:18:deploy/aliyun_fc/s.yaml
edition: "3.0.0"
name: falcon-perception-fc
access: default

defaults:
  region: cn-shanghai
  functionName: falcon-perception-api
  image: registry.cn-shanghai.aliyuncs.com/your-namespace/falcon-perception-fc:latest
  gpuType: fc.gpu.tesla.1
  gpuMemorySize: 24576
  ...
  modelMountDir: /mnt/auto/falcon-perception
  nasServerAddr: please-replace-with-your-nas-server:/
  prewarmCommand: python /app/deploy/aliyun_fc/prewarm.py
```

```24:80:deploy/aliyun_fc/s.yaml
region: ${defaults.region}
functionName: ${defaults.functionName}
...
customContainerConfig:
  image: ${defaults.image}
  port: 7860
gpuConfig:
  gpuType: ${defaults.gpuType}
  gpuMemorySize: ${defaults.gpuMemorySize}
scaling:
  minInstances: ${defaults.minInstances}
  maxInstances: ${defaults.maxInstances}
environmentVariables:
  PORT: "7860"
  SERVER_HOST: "0.0.0.0"
  NUM_GPUS: "1"
  DTYPE: "bfloat16"
  STARTUP_TIMEOUT: "900"
  IMAGES_DIR: "/tmp/falcon-perception/images"
  ENABLE_HR_CACHE: "0"
  HF_LOCAL_DIR: ${defaults.modelMountDir}
  ...
nasConfig:
  ...
initializer:
  enable: true
  timeout: 900
  type: RunCommand
  command: ${defaults.prewarmCommand}
```

```1:13:deploy/aliyun_fc/provision.json
{
  "targetTrackingPolicies": [
    {
      "name": "falcon-perception-gpu-scaling",
      ...
      "metricType": "ProvisionedConcurrencyUtilization",
      "metricTarget": 0.3,
      "minCapacity": 1,
      "maxCapacity": 4
    }
  ]
}
```

我还顺手做了结构校验，`s.yaml` 和 `provision.json` 都能正常解析。

你现在只需要改这几个值：

1. `defaults.region`
2. `defaults.image`
3. `defaults.gpuType`
4. `defaults.gpuMemorySize`
5. `defaults.nasServerAddr`
6. 如需 VPC，把注释掉的 `vpcConfig` 打开并填真实值

建议部署顺序：

```bash
docker build -f Dockerfile.aliyun-fc-gpu -t <acr>/falcon-perception-fc:latest .
docker push <acr>/falcon-perception-fc:latest
s deploy -t deploy/aliyun_fc/s.yaml -a default
s provision put --target 1 --targetTrackingPolicies ./deploy/aliyun_fc/provision.json --qualifier LATEST -t deploy/aliyun_fc/s.yaml -a default
```

两个注意点：

1. `initializer` 我已经按阿里云文档思路写进 `s.yaml` 了；如果你当前 `fc3` 版本不接受这个字段，就到 FC 控制台“生命周期”里填同一条命令：
   ```bash
   python /app/deploy/aliyun_fc/prewarm.py
   ```

2. 当前模板默认是 `NAS 挂载模型目录`。如果你最后决定用 `OSS 挂载`，把 `nasConfig` 改成对应挂载方式，同时把 `HF_LOCAL_DIR` 指到 OSS 挂载路径即可。

如果你愿意，我下一步可以直接帮你把 `deploy/aliyun_fc/s.yaml` 按你的真实阿里云信息改成可部署版本。只要你给我这几个值：

- 地域
- ACR 镜像地址
- GPU 规格
- NAS/OSS 挂载方式
- 是否需要 VPC