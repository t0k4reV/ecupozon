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
Историческое ограничение воспроизводимости OCR-адаптера описано в корневом
README и фиксируется в training manifest.

Post-training pipeline использует тот же production runtime, что и submission:
EasyOCR-текст первого фото входит во вход классификатора ЛВ. Он сохраняет
`predictions.csv`, метрики обоих holdout, таблицу проверки comments v4 и
`reproduction_report.json` в `artifacts/gemma_lora/v1_ocr/evaluation/`.

Для БАД доступен полный исторический baseline. Для ЛВ архив лучшего v1 фиксирует
production threshold `0.005`, но не содержит validation predictions и метрик.
Поэтому отчёт проверяет доступные факты и явно помечает историческое сравнение
ЛВ как `partial`, не подменяя отсутствующие результаты оценкой новой модели.
