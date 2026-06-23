# Generate Cache Patch

This directory implements generate_cache in the same style as
`verl-recipe/specRL/histoSpec`: keep the upstream trainer intact and add a thin
recipe subclass.

## Files

- `main_ppo.py`
  - Thin entrypoint for `python3 -m generate_cache_patch.main_ppo`.
  - Reuses `verl.trainer.main_ppo` and injects `GenerateCacheRayPPOTrainer`
    through a small `GenerateCacheTaskRunner`.

- `ray_trainer.py`
  - Defines `class GenerateCacheRayPPOTrainer(RayPPOTrainer)`.
  - Does not copy the full upstream trainer file.
  - Wraps rollout generation to query/rebuild historical generations.
  - Overrides `_compute_old_log_prob` to save newly generated rollouts after
    old log probabilities are computed.

- `run_generate_cache_patch.sh`
  - Minimal launcher. Pass the usual Hydra overrides after the script.

## Usage

From the repo root:

```bash
bash generate_cache_patch/run_generate_cache_patch.sh \
  algorithm.adv_estimator=grpo \
  data.train_files=/path/to/train.parquet \
  data.val_files=/path/to/test.parquet \
  actor_rollout_ref.model.path=/path/to/model \
  actor_rollout_ref.rollout.n=8 \
  +trainer.gen_cache.save_path=./his_data/gencache_run \
  trainer.device=npu
```

You can also call the module directly:

```bash
python3 -m generate_cache_patch.main_ppo \
  +trainer.gen_cache.use_gen_cache=true \
  ...
```

## Config

The generate_cache implementation reads:

- `trainer.gen_cache.use_gen_cache`
- `trainer.gen_cache.save_path`
- `trainer.gen_cache.reuse_factor`
- `trainer.gen_cache.reuse_max_len`
- `trainer.gen_cache.chunk_size`
- `trainer.gen_cache.num_write_workers`
- `trainer.gen_cache.num_query_workers`
- `trainer.gen_cache.max_pending_batches`
- `trainer.gen_cache.drop_when_full`
- `trainer.gen_cache.overwrite`

The cache implementation itself is reused from
`recipe.generate_cache.generation_reuse.GenCacheManager`.

## Limitation

`trainer.gen_cache.use_gen_cache=True` currently raises an explicit error with
`algorithm.adv_estimator=remax`, because the latest upstream REMAX path combines
sampled and greedy rollout requests in a way that needs a dedicated cache policy.
