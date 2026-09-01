# v1-OCR

Классификатор категории `Легковоспламеняющиеся` получает первое изображение и
распознанный EasyOCR-текст. В production используется одно изображение и
агрегация `single`. Общий адаптер категории `БАД` не использует OCR.

```bash
uv run python -m scripts.solutions.v1_ocr.train --images-dir /data/images
uv run python -m scripts.solutions.v1_ocr.post_training --images-dir /data/images
uv run python -m scripts.solutions.v1_ocr.build_submission
```

Точные пути артефактов и inference-контракт находятся в `profile.json`.
Исторический submission и результаты воспроизводимого обучения проверяются
раздельно; provenance фиксируется в baseline и training manifest.

Post-training pipeline использует тот же production runtime, что и submission:
EasyOCR-текст первого фото входит во вход классификатора ЛВ. Он сохраняет
`predictions.csv`, метрики обоих holdout, таблицу проверки comments v4 и
`reproduction_report.json` в `artifacts/gemma_lora/v1_ocr/evaluation/`.

Для БАД доступен полный исторический baseline. Для ЛВ архив лучшего v1 фиксирует
production threshold `0.005`; его оценка на детерминированном holdout сохранена
отдельно от метрик нового адаптера. Свежий threshold выбирается по собственным
validation probabilities.

Проверить исторический, сохранённый и validation-optimal thresholds без Gemma:

```bash
uv run python -m scripts.solutions.v1_ocr.audit_threshold \
  --predictions artifacts/gemma_lora/v1_ocr/evaluation/predictions.csv \
  --output-dir artifacts/gemma_lora/v1_ocr/threshold_audit
```
