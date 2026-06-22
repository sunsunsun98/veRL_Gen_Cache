#!/usr/bin/env bash
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

project_name='GRPO-openPangu-Embedded-7B-V1.1-gsm8k-envtest'
exp_name='0203-train-bf16-qatw8a16-rollout-w8a16-gencache'

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HYDRA_FULL_ERROR=1

export VLLM_USE_V1=1

# export VLLM_ENABLE_V1_MULTIPROCESSING=0
# export RAY_PDB_IMPORT_PATH=pdb
# export RAY_DEBUG=legacy


# # Importance Sampling (IS) weights configuration
# rollout_is="sequence"                     # Self-normalized sequence-level IS
# rollout_is_threshold=2.0                  # Upper threshold for IS weights
# rollout_is_batch_normalize="true"        # Self-normalization (mean=1.0)

# # Rejection Sampling (RS) configuration
# rollout_rs="seq_mean_k2"                         # No rejection sampling for basic RLOO  "null"
# rollout_rs_threshold=2.0               # RS threshold spec (string or float)

# Importance Sampling (IS) weights configuration
rollout_is="null"                     # Self-normalized sequence-level IS
rollout_is_threshold="null"                  # Upper threshold for IS weights
rollout_is_batch_normalize="null"        # Self-normalization (mean=1.0)

# Rejection Sampling (RS) configuration
rollout_rs="null"                         # No rejection sampling for basic RLOO  "null"
rollout_rs_threshold="null"               # RS threshold spec (string or float)

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Generation cache
use_gen_cache=true
gen_cache_save_path="./his_data/${exp_name}"
gen_cache_reuse_factor=0.0
gen_cache_reuse_max_len=1024
gen_cache_chunk_size=1000
gen_cache_num_write_workers=1
gen_cache_num_query_workers=1
gen_cache_max_pending_batches=8
gen_cache_drop_when_full=true
gen_cache_overwrite=true

# actor_rollout_ref.rollout.quantization=ascend \
python3 -m recipe.generate_cache.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/data/l50044498/datasets/gsm8k_rl/train.parquet \
    data.val_files=/data/l50044498/datasets/gsm8k_rl/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=32 \
    data.truncation='left' \
    data.trust_remote_code=True \
    +model.trust_remote_code=True \
    actor_rollout_ref.model.path=/data/l50044498/models/openPangu-Embedded-7B-V1.1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_scheduler_type='constant' \
    actor_rollout_ref.actor.optim.lr_warmup_steps=3 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.qat=True \
    actor_rollout_ref.actor.qat_w_bit=8 \
    actor_rollout_ref.actor.scale_source="learned" \
    actor_rollout_ref.actor.fsdp_config.dtype="bfloat16" \
    actor_rollout_ref.actor.fsdp_config.model_dtype="bfloat16" \
    actor_rollout_ref.ref.fsdp_config.dtype="bfloat16" \
    actor_rollout_ref.ref.fsdp_config.model_dtype="bfloat16" \
    actor_rollout_ref.rollout.dtype="bfloat16" \
    actor_rollout_ref.rollout.quantization=ascend \
    actor_rollout_ref.rollout.model_path=/data/l50044498/models/openPangu-Embedded-7B-V1.1-W8A16 \
    actor_rollout_ref.rollout.load_format="auto" \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.n=8 \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_correction.rollout_is_batch_normalize=${rollout_is_batch_normalize} \
    algorithm.rollout_correction.rollout_rs=${rollout_rs} \
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    +trainer.gen_cache.use_gen_cache=${use_gen_cache} \
    +trainer.gen_cache.save_path="${gen_cache_save_path}" \
    +trainer.gen_cache.reuse_factor=${gen_cache_reuse_factor} \
    +trainer.gen_cache.reuse_max_len=${gen_cache_reuse_max_len} \
    +trainer.gen_cache.chunk_size=${gen_cache_chunk_size} \
    +trainer.gen_cache.num_write_workers=${gen_cache_num_write_workers} \
    +trainer.gen_cache.num_query_workers=${gen_cache_num_query_workers} \
    +trainer.gen_cache.max_pending_batches=${gen_cache_max_pending_batches} \
    +trainer.gen_cache.drop_when_full=${gen_cache_drop_when_full} \
    +trainer.gen_cache.overwrite=${gen_cache_overwrite} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=2 \
    trainer.device=npu $@


#     actor_rollout_ref.rollout.quantization=ascend \
#     actor_rollout_ref.rollout.model_path=/data/l50044498/models/qwen3-1.7b-W8A16-C \
#     actor_rollout_ref.actor.optim.lr_scheduler_type='cosine' \
# algorithm.rollout_correction.rollout_is=${rollout_is} \
#     algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
#     algorithm.rollout_correction.rollout_is_batch_normalize=${rollout_is_batch_normalize} \
#     algorithm.rollout_correction.rollout_rs=${rollout_rs} \
#     algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold} \
