# v1-OCR

Классификатор категории `Легковоспламеняющиеся` получает первое изображение и
распознанный EasyOCR-текст. В production используется одно изображение и
агрегация `single`. Общий адаптер категории `БАД` не использует OCR.

Последовательность воспроизведения: подготовка `data.csv` внутри training
pipeline → обучение двух LoRA → автоматическая калибровка на validation →
оценка → сборка ZIP.

```bash
uv run python -m scripts.solutions.v1_ocr.train --images-dir /data/images
uv run python -m scripts.solutions.v1_ocr.post_training --images-dir /data/images
uv run python -m scripts.solutions.v1_ocr.build_submission
```

Точные пути артефактов и inference-контракт находятся в `profile.json`.
Production thresholds БАД и ЛВ независимо выбираются на своих validation
holdout по F1 → precision → большему threshold и фиксируются в training manifest.

Post-training pipeline использует тот же production runtime, что и submission:
EasyOCR-текст первого фото входит во вход классификатора ЛВ. Он сохраняет
`predictions.csv`, метрики обоих holdout, таблицу проверки comments v4 и
`evaluation_report.json` в `artifacts/gemma_lora/v1_ocr/evaluation/`.

Пересчитать thresholds готовых адаптеров без обучения:

```bash
uv run python -m scripts.common.training.recalibrate_thresholds
uv run python -m scripts.common.training.recalibrate_thresholds --apply
```
