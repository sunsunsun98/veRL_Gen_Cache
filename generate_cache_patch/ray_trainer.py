"""Generate-cache extension for the upstream RayPPOTrainer.

This module intentionally keeps the recipe thin: it subclasses the latest
``verl.trainer.ppo.ray_trainer.RayPPOTrainer`` and wraps only the rollout and
old-log-prob edges needed by generate_cache.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable

from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class GenerateCacheRayPPOTrainer(RayPPOTrainer):
    """Ray PPO trainer with history generation cache reuse."""

    def fit(self):
        gen_cache_config = self.config.trainer.get("gen_cache", {})
        self._use_gen_cache = gen_cache_config.get("use_gen_cache", False)
        self._gen_cache_pending_metrics: dict[str, Any] = {}
        self._gen_cache_pending_timing: dict[str, Any] = {}

        if not self._use_gen_cache:
            return super().fit()

        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            raise ValueError("trainer.gen_cache.use_gen_cache=True is not supported with REMAX yet.")

        from recipe.generate_cache.generation_reuse import GenCacheManager

        self.gc_mgr = GenCacheManager(
            trainer_config=self.config,
            actor_rollout_wg=self.actor_rollout_wg,
            tokenizer=self.tokenizer,
        )

        try:
            with self._patch_rollout_generation():
                return super().fit()
        finally:
            self.gc_mgr.cache.shutdown()

    @contextmanager
    def _patch_rollout_generation(self):
        original_generate_sequences = self.async_rollout_manager.generate_sequences
        self.async_rollout_manager.generate_sequences = self._wrap_generate_sequences(original_generate_sequences)
        try:
            yield
        finally:
            self.async_rollout_manager.generate_sequences = original_generate_sequences

    def _wrap_generate_sequences(self, generate_sequences: Callable[[DataProto], DataProto]):
        def wrapped(gen_batch: DataProto) -> DataProto:
            if not self._should_use_cache_for_batch(gen_batch):
                return generate_sequences(gen_batch)

            metrics: dict[str, Any] = {}
            timing_raw: dict[str, Any] = {}
            reuse_pre_result = self.gc_mgr.reuse_generation(
                gen_batch=gen_batch,
                async_rollout_manager=self.async_rollout_manager,
                async_rollout_mode=self.async_rollout_mode,
                global_steps=gen_batch.meta_info.get("global_steps", self.global_steps),
                metrics=metrics,
                timing_raw=timing_raw,
            )

            if reuse_pre_result is None or not reuse_pre_result.have_pre_rollouts:
                output = generate_sequences(gen_batch)
                self._stash_cache_metrics(metrics, timing_raw)
                return output

            if len(reuse_pre_result.gen_batch) > 0:
                continued_output = generate_sequences(reuse_pre_result.gen_batch)
            else:
                continued_output = None

            output, timing_raw = self.gc_mgr.rebuild_generate_batch(
                continued_output,
                reuse_pre_result,
                timing_raw,
            )
            output.meta_info.setdefault("timing", {})
            output.meta_info["timing"].update(timing_raw)
            self._stash_cache_metrics(metrics, timing_raw)
            return output

        return wrapped

    def _should_use_cache_for_batch(self, gen_batch: DataProto) -> bool:
        if not getattr(self, "_use_gen_cache", False):
            return False
        if gen_batch.meta_info.get("validate", False):
            return False
        if "__do_sample__" in gen_batch.non_tensor_batch:
            return False
        return "raw_prompt" in gen_batch.non_tensor_batch

    def _stash_cache_metrics(self, metrics: dict[str, Any], timing_raw: dict[str, Any]) -> None:
        self._gen_cache_pending_metrics.update(metrics)
        self._gen_cache_pending_timing.update(timing_raw)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        super()._balance_batch(
            batch=batch,
            metrics=metrics,
            logging_prefix=logging_prefix,
            keep_minibatch=keep_minibatch,
        )
        if getattr(self, "_gen_cache_pending_metrics", None):
            metrics.update(self._gen_cache_pending_metrics)
            self._gen_cache_pending_metrics.clear()

    def _compute_old_log_prob(self, batch: DataProto):
        old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
        self._save_generation_cache(batch, old_log_prob)
        return old_log_prob, old_log_prob_mfu

    def _save_generation_cache(self, batch: DataProto, old_log_prob: DataProto) -> None:
        if not getattr(self, "_use_gen_cache", False):
            return
        if "raw_prompt" not in batch.non_tensor_batch:
            return
        if "responses" not in batch.batch or "prompts" not in batch.batch:
            return
        if "old_log_probs" not in old_log_prob.batch:
            return

        self.gc_mgr.cache.save_batch_async(
            prompts=batch.non_tensor_batch["raw_prompt"],
            responses=batch.batch["responses"],
            input_ids=batch.batch["prompts"],
            old_logps=old_log_prob.batch["old_log_probs"],
            n_repeat=self.config.actor_rollout_ref.rollout.n,
        )
