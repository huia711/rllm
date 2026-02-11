#!/bin/bash
set -e

##############################################################################
# rllm FrozenLake + UI-TARS-1.5-7B 一键运行脚本
# 在 8xH200 GPU 服务器上执行此脚本
##############################################################################

# ============ 环境配置 ============
CONDA_ENV=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/llm/project/wxy/conda_envs/rllm
RLLM_DIR=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/llm/project/wxy/rllm
MODEL_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/llm/project/wxy/models/UI-TARS-1.5-7B

export PATH="$CONDA_ENV/bin:$PATH"
unset PYTHONPATH

echo "=========================================="
echo "  rllm FrozenLake Smoke Test"
echo "=========================================="

# ============ Step 1: 环境检查 ============
echo ""
echo "[Step 1] 检查环境..."

echo "  Python: $(python3 --version)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU count: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
echo "  vLLM: $(python3 -c 'import vllm; print(vllm.__version__)')"
echo "  flash-attn: $(python3 -c 'import flash_attn; print(flash_attn.__version__)')"
echo "  transformers: $(python3 -c 'import transformers; print(transformers.__version__)')"
echo "  verl: $(python3 -c 'import verl; print(verl.__version__)')"
echo "  rllm: $(python3 -c 'import rllm; print(\"installed\")')"

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "  ERROR: 模型路径不存在: $MODEL_PATH"
    exit 1
fi
echo "  Model path: OK ($MODEL_PATH)"

# 检查 GPU
GPU_COUNT=$(python3 -c 'import torch; print(torch.cuda.device_count())')
if [ "$GPU_COUNT" -lt 1 ]; then
    echo "  ERROR: 没有检测到 GPU！"
    exit 1
fi
echo "  GPU info:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"

echo ""
echo "[Step 1] 环境检查通过！"

# ============ Step 2: 运行训练 ============
echo ""
echo "[Step 2] 启动 FrozenLake 训练..."
echo "  使用模型: UI-TARS-1.5-7B"
echo "  GPU数量: $GPU_COUNT"
echo ""

cd "$RLLM_DIR"

set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

python3 -m examples.frozenlake.train_frozenlake_agent \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=16 \
    data.val_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=2 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger='[console]' \
    trainer.project_name='rllm-agent' \
    trainer.experiment_name='frozenlake-uitars-7B-smoke' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    rllm.rejection_sample.enable=True \
    rllm.rejection_sample.multiplier=2 \
    +rllm.env.env_args.max_steps=8 \
    +rllm.env.env_args.is_slippery=False \
    rllm.agent.max_steps=10 \
    rllm.stepwise_advantage.enable=False \
    rllm.disable_thinking=False \
    +rllm.agent.agent_args.max_steps=10 \
    +rllm.agent.agent_args.use_accumulate_history=True \
    trainer.total_epochs=1
