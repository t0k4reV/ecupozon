# v2-img3

Классификатор категории `Легковоспламеняющиеся` обучается без OCR. В production
Gemma независимо оценивает до трёх изображений, после чего используется максимум
вероятностей. EasyOCR участвует только в формировании комментариев.

```bash
uv run python -m scripts.solutions.v2_img3.train --images-dir /data/images
uv run python -m scripts.solutions.v2_img3.post_training --images-dir /data/images
uv run python -m scripts.solutions.v2_img3.audit_comments --help
uv run python -m scripts.solutions.v2_img3.build_submission
```

Точные пути артефактов и inference-контракт находятся в `profile.json`.

Post-training pipeline повторно запускает production-инференс на обоих holdout,
считает F1/precision/recall и confusion matrix, формирует комментарии с EasyOCR и
проверяет их формат, ссылки на фотографии и согласованность evidence с verdict.
Результаты записываются в `artifacts/gemma_lora/v2_img3/evaluation/`:

- `predictions.csv` — вероятности каждого фото, итоговая вероятность и метка;
- `comments_review.csv` — данные для ручной проверки комментариев и ошибок;
- `metrics.json` и `comments_audit.json` — машинно-читаемые проверки;
- `reproduction_report.json` — сравнение с историческим v2 baseline.

По умолчанию оценивается только validation. Для просмотра комментариев на всём
размеченном датасете используйте `--scope all`; baseline всё равно проверяется
только на исходных holdout, поэтому train-строки не попадают в контрольные метрики.

`audit_comments` повторно использует готовый `predictions.csv` и не загружает
Gemma. Опциональный полный OCR-кэш предназначен для сравнения с историческим
comments v4 snapshot; без него запускается тот же bounded EasyOCR, что и в
production runtime.
