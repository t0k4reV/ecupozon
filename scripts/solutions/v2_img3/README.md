# v2-img3

Классификатор категории `Легковоспламеняющиеся` обучается без OCR. В production
Gemma независимо оценивает до трёх изображений, после чего используется максимум
вероятностей. EasyOCR участвует только в формировании комментариев.

Последовательность воспроизведения: подготовка `data.csv` внутри training
pipeline → обучение двух LoRA → автоматическая калибровка на validation →
оценка → сборка ZIP.

```bash
uv run python -m scripts.solutions.v2_img3.train --images-dir /data/images
uv run python -m scripts.solutions.v2_img3.post_training --images-dir /data/images
uv run python -m scripts.solutions.v2_img3.audit_comments --help
uv run python -m scripts.solutions.v2_img3.build_submission
```

Точные пути артефактов и inference-контракт находятся в `profile.json`.
Production thresholds БАД и ЛВ независимо выбираются на своих validation
holdout по F1 → precision → большему threshold и фиксируются в training manifest.

Готовые validation probabilities можно проверить или перекалибровать без Gemma:

```bash
uv run python -m scripts.common.training.recalibrate_thresholds
uv run python -m scripts.common.training.recalibrate_thresholds --apply
```

Post-training pipeline повторно запускает production-инференс на обоих holdout,
считает F1/precision/recall и confusion matrix, формирует комментарии с EasyOCR и
проверяет их формат, ссылки на фотографии и согласованность evidence с verdict.
Результаты записываются в `artifacts/gemma_lora/v2_img3/evaluation/`:

- `predictions.csv` — вероятности каждого фото, итоговая вероятность и метка;
- `comments_review.csv` — данные для ручной проверки комментариев и ошибок;
- `metrics.json` и `comments_audit.json` — машинно-читаемые проверки;
- `evaluation_report.json` — итоговая оценка и provenance.

По умолчанию оценивается только validation. Для просмотра комментариев на всём
размеченном датасете используйте `--scope all`; контрольные метрики всё равно
считаются только на validation, поэтому train-строки их не завышают.

`audit_comments` повторно использует готовый `predictions.csv` и не загружает
Gemma. Опциональный полный OCR-кэш предназначен для сравнения с reference
comments v4 snapshot; без него запускается тот же bounded EasyOCR, что и в
production runtime.
