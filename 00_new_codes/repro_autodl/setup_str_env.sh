#!/usr/bin/env bash
# STReasoner SFT 环境一键配置（AutoDL 数据盘）
# 用法: bash 00_new_codes/repro_autodl/setup_str_env.sh
set -euo pipefail

REPO=/root/autodl-tmp/STReasoner_reproduce
ENV=/root/autodl-tmp/conda/envs/str-py310
CACHE=/root/autodl-tmp/cache
WHEEL="${CACHE}/wheels/flash_attn-2.7.2.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
FLASH_URL="https://ghproxy.com/https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

mkdir -p "${CACHE}/huggingface" "${CACHE}/pip" "${CACHE}/wheels" "${CACHE}/triton" \
  "${CACHE}/torch_extensions" "${CACHE}/huggingface/datasets"

# pip 镜像 + 缓存到数据盘
mkdir -p /root/.pip
cat > /root/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 120
cache-dir = /root/autodl-tmp/cache/pip

[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

if [[ ! -d "${ENV}" ]]; then
  conda create -p "${ENV}" python=3.10 -y
fi

PIP="${ENV}/bin/pip"

if [[ ! -f "${WHEEL}" ]]; then
  wget -O "${WHEEL}" "${FLASH_URL}" || wget -O "${WHEEL}" "${FLASH_URL#https://ghproxy.com/}"
fi

if ! "${ENV}/bin/python" -c "import torch" 2>/dev/null; then
  "${PIP}" install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
fi

if ! "${ENV}/bin/python" -c "import flash_attn" 2>/dev/null; then
  "${PIP}" install "${WHEEL}"
fi

grep -vE '^(torch==|https://github.com/Dao-AILab|#)' "${REPO}/requirements.txt" | grep -v '^$' > /tmp/str_requirements_rest.txt
"${PIP}" install -r /tmp/str_requirements_rest.txt
"${PIP}" install 'bitsandbytes==0.45.2'

echo "== verify =="
export HF_ENDPOINT=https://hf-mirror.com HF_HOME="${CACHE}/huggingface"
"${ENV}/bin/python" - <<'PY'
import torch, flash_attn, transformers, vllm, deepspeed, bitsandbytes
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("flash_attn", flash_attn.__version__)
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
print("deepspeed", deepspeed.__version__)
print("bitsandbytes", bitsandbytes.__version__)
PY

echo "Done. Activate: export PATH=${ENV}/bin:\$PATH"
