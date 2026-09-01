"""Run v1 metrics, predictions, and comments audit after training."""

from scripts.common.evaluation.entrypoint import run_post_training_pipeline

if __name__ == "__main__":
    run_post_training_pipeline(
        __doc__ or "Run v1 post-training evaluation",
        "scripts.solutions.v1_ocr.evaluate",
    )
