# E-CUP Gemma LoRA solutions

Репозиторий команды `347cc0148476edd1d0e92389c`.

Репозиторий воспроизводит два решения на общей Gemma 4 E4B и общем
LoRA-адаптере категории `БАД`:

| Решение | Классификатор ЛВ | Production-инференс ЛВ | Роль EasyOCR |
|---|---|---|---|
| `v1_ocr` | изображение + OCR-текст | первое фото, `single` | входит во вход Gemma и влияет на метку |
| `v2_img3` | только изображение | до трёх фото отдельно, `max` | используется только в комментариях |

Лучшим итоговым решением команды является `v1_ocr`. Результаты соревнования по
метрике **Macro F1 Averaged**:

| Решение | Public | Private |
|---|---:|---:|
| `v1_ocr` — лучшее итоговое | `0.9114292364990689` | `0.8723967687` |
| `v2_img3` | `0.9229088283358036` | — |

Оба submission используют детерминированные комментарии v4. Правила комментариев
не меняют verdict модели.

Готовые модели, изображения, OCR-кэш и ZIP-архивы в репозиторий не включены.
Исходный датасет соревнования также не публикуется.

## Структура проекта

```text
scripts/
├── common/                   # общая реализация без различий v1/v2
│   ├── training/             # Gemma, данные, БАД и training manifest
│   ├── evaluation/           # общие метрики, predictions и аудит comments v4
│   └── submission/           # EasyOCR, ZIP builder и offline runtime
└── solutions/
    ├── v1_ocr/               # OCR-кэш, OCR-LoRA и entrypoints v1
    └── v2_img3/              # image-only LoRA и entrypoints v2
```

В каталоге каждого решения находятся собственные `README.md`, `train.py`,
`post_training.py`, `build_submission.py`, `profile.json` и исторический
`reproduction_baseline.json`. Профиль является единственным источником путей
артефактов, baseline и production inference-контракта конкретного решения.

- [`v1_ocr`](scripts/solutions/v1_ocr/README.md) — классификация ЛВ с OCR-текстом;
- [`v2_img3`](scripts/solutions/v2_img3/README.md) — image-only классификация ЛВ с max3.

## Требования

- Linux x86_64 и Python 3.12;
- CUDA GPU для OCR, обучения и инференса;
- обучение финальных адаптеров выполнялось на NVIDIA A100 80 GB;
- `uv` и окружение из `uv.lock`;
- доступ к `google/gemma-4-E4B-it` только на стадии загрузки;
- изображения в `<images-dir>/<id>/*.jpg`.

```bash
uv sync
```

Для gated-модели можно задать `HF_TOKEN`.

## Воспроизведение из чистого checkout

Клонируйте репозиторий на Linux-машину с CUDA и установите зафиксированное
окружение:

```bash
git clone https://github.com/t0k4reV/ecupozon.git
cd ecupozon
uv sync --frozen
```

Получите train CSV и изображения из официального датасета соревнования. Положите
исходный CSV в корень репозитория под именем `data.csv`, а изображения — во
внешний каталог со структурой `<images-dir>/<id>/*.jpg`. CSV должен содержать
колонки `id`, `name`, `description`, `category`, `label`.

Для точного воспроизведения использован следующий снимок данных:

```text
data.csv SHA-256: 4bc59e640563160fa04572b570606ceb1dd3d31627c6cf7fd1750ae4ea61f510
исходных строк: 12 971
строк после дедупликации: 9 516
изображений: 35 886
```

Проверить CSV можно командой `sha256sum data.csv`. На macOS используйте
`shasum -a 256 data.csv`. `data.csv`, автоматически создаваемый
`data_no_duplicates.csv` и все производные артефакты исключены из Git.

Перед длительным обучением выполните один CUDA forward для каждого решения:

```bash
uv run python -m scripts.solutions.v2_img3.train \
  --images-dir /data/images --check-only

uv run python -m scripts.solutions.v1_ocr.train \
  --images-dir /data/images \
  --ocr-cache /data/easyocr_ru_en.jsonl \
  --check-only
```

Для `v1_ocr` в check-only нужен заранее созданный совместимый OCR-кэш. Полные
pipeline запускаются независимо:

```bash
uv run python -m scripts.solutions.v2_img3.train \
  --images-dir /data/images

uv run python -m scripts.solutions.v1_ocr.train \
  --images-dir /data/images
```

`v1_ocr` самостоятельно скачает EasyOCR-веса и построит OCR-кэш, если
`--ocr-cache` не передан. Оба pipeline загружают Gemma только при отсутствии
проверенной локальной копии. После обучения выполните оценку и соберите архивы:

```bash
uv run python -m scripts.solutions.v2_img3.post_training \
  --images-dir /data/images
uv run python -m scripts.solutions.v1_ocr.post_training \
  --images-dir /data/images

uv run python -m scripts.solutions.v2_img3.build_submission
uv run python -m scripts.solutions.v1_ocr.build_submission
```

`v1_ocr` передаёт OCR-текст во вход классификатора ЛВ. `v2_img3` классифицирует
только изображения, а EasyOCR использует исключительно для comments v4.
Исторический submission v1 и воспроизводимый training pipeline проверяются
раздельно; их метрики приведены для соответствующих адаптеров.

## Общие стадии обучения

Оба pipeline:

1. проверяют локальную Gemma в `models/google/gemma-4-E4B-it` и скачивают её
   только при отсутствии;
2. читают предоставленный пользователем `data.csv`, удаляют дубли, записывают
   локальный `data_no_duplicates.csv` и валидируют категории, метки, `id` и
   изображения;
3. создают dataset manifest в `artifacts/gemma_lora/data/`;
4. обучают общий адаптер `БАД`;
5. обучают собственный адаптер ЛВ и создают отдельный `training_manifest.json`.

Общие параметры двух LoRA-профилей:

| Параметр | Значение |
|---|---:|
| seed | `2026` |
| epochs | `2` |
| batch size | `1` |
| gradient accumulation | `8` |
| learning rate | `1e-4` |
| scheduler | cosine, warmup `30` steps |
| LoRA | `r=32`, `alpha=64`, dropout `0.05`, `all-linear` |
| oversampling | меньший класс до `30%` только в train |

`БАД` использует split `90/10`, одно случайное фото из всех доступных и RGB без
EXIF transpose или resize. Production использует первое фото и threshold `0.7`.

Обе воспроизводимые ветки ЛВ используют split `80/20`: 3 160 train-строк до
oversampling, 4 352 после него и 791 validation. На каждом train-примере подаётся
ровно одно фото: первое с вероятностью `0.5`, иначе случайное из первых трёх.
Применяются EXIF transpose, RGB и ограничение стороны 1024.

## Решение v2-img3

```bash
uv run python -m scripts.solutions.v2_img3.train --images-dir /data/images
```

ЛВ обучается без OCR. На validation первые три фотографии оцениваются отдельно;
threshold выбирается по `F1`, затем `precision`, затем большему значению threshold.
Production использует максимум трёх вероятностей.

После обучения отдельный pipeline оценивает обе LoRA на сохранённых holdout и
прогоняет тот же production runtime комментариев:

```bash
uv run python -m scripts.solutions.v2_img3.post_training \
  --images-dir /data/images
```

Для аудита комментариев по всему размеченному датасету добавьте `--scope all`.
Контрольные метрики при этом по-прежнему считаются только на validation, чтобы
train-примеры не завышали результат. Pipeline создаёт:

```text
artifacts/gemma_lora/v2_img3/evaluation/
├── predictions.csv
├── comments_review.csv
├── metrics.json
├── comments_audit.json
└── reproduction_report.json
```

`reproduction_report.json` сравнивает F1, precision, recall, confusion matrix,
размеры holdout и production thresholds с зафиксированным историческим baseline.
Для стохастического CUDA-обучения отдельно показываются точные отклонения и
результат сравнения с явно записанными допусками.

Комментарии можно пересчитать по уже сохранённым predictions без повторной
загрузки Gemma. Без `--ocr-cache` используется ограниченный production-режим
EasyOCR. С полным историческим кэшем выполняется точный comments v4 audit:

```bash
uv run python -m scripts.solutions.v2_img3.audit_comments \
  --images-dir /data/images \
  --predictions artifacts/gemma_lora/v2_img3/evaluation_all/predictions.csv \
  --ocr-cache /data/easyocr_ru_en.jsonl \
  --golden-comments /data/comments_all_cards.csv \
  --output-dir artifacts/gemma_lora/v2_img3/comments_v4_reproduction
```

Golden CSV и dataset-specific OCR-кэш нужны только для проверки исторического
результата и не включаются в submission. Pipeline сохраняет новый
`comments_review.csv`, машинно-читаемый отчёт сравнения и только отличающиеся
карточки в `comment_mismatches.csv`.

Артефакты:

```text
artifacts/gemma_lora/shared/adapters/gemma_e4b_supplements/
artifacts/gemma_lora/v2_img3/adapters/gemma_e4b_flammable/
artifacts/gemma_lora/v2_img3/training_manifest.json
```

## Решение v1-OCR

Сначала EasyOCR строит JSONL-кэш для первых трёх фотографий каждой карточки ЛВ.
Можно передать ранее созданный совместимый кэш через `--ocr-cache`.

```bash
uv run python -m scripts.solutions.v1_ocr.train \
  --images-dir /data/images

uv run python -m scripts.solutions.v1_ocr.train \
  --images-dir /data/images \
  --ocr-cache /data/easyocr_ru_en.jsonl
```

Для выбранного train-изображения в Gemma передаются два текстовых блока:

```text
OCR изображения N: <распознанный текст>
Категория + правила + название + описание + предупреждение об ошибках OCR
```

Production использует первое фото и OCR этого же фото. Threshold выбирается на
validation для режима `single`.

После обучения v1 запускается тем же post-training pipeline, что и v2:

```bash
uv run python -m scripts.solutions.v1_ocr.post_training \
  --images-dir /data/images
```

Он повторяет production-инференс с OCR во входе классификатора ЛВ, считает
метрики обоих holdout и создаёт аудит comments v4. Результаты записываются в
`artifacts/gemma_lora/v1_ocr/evaluation/` с тем же набором файлов, что у v2.

Артефакты:

```text
artifacts/gemma_lora/v1_ocr/easyocr_training_cache.jsonl
artifacts/gemma_lora/v1_ocr/adapters/gemma_e4b_flammable_ocr/
artifacts/gemma_lora/v1_ocr/training_manifest.json
```

### Проверка threshold v1

Из исторического ZIP v1 восстановлены production thresholds (`0.7` для БАД и
`0.005` для ЛВ), режим `single/1` и SHA-256 OCR-адаптера. Повторная оценка
архивного адаптера на детерминированном holdout из 791 карточки при threshold
`0.005` дала F1, precision и recall `1.0` (`TN=762`, `FP=0`, `FN=0`, `TP=29`).
Эти значения помечены как реконструированная оценка, а не исходный training log.

Свежий адаптер использует threshold, выбранный по собственным validation
probabilities. Selector проверяет все пороги, меняющие набор predictions, и
выбирает максимум по F1, затем precision, затем большему threshold. Проверить
чувствительность без загрузки Gemma можно по готовому CSV:

```bash
uv run python -m scripts.solutions.v1_ocr.audit_threshold \
  --predictions artifacts/gemma_lora/v1_ocr/evaluation/predictions.csv \
  --output-dir artifacts/gemma_lora/v1_ocr/threshold_audit
```

Результат содержит `threshold_audit.json` и `threshold_curve.csv` с метриками
исторического, сохранённого и validation-optimal thresholds.

## Проверка без обучения

`--check-only` загружает Gemma, создаёт split и выполняет один forward без
backward, optimizer и записи адаптера. Для `v1_ocr` нужен существующий OCR-кэш.

```bash
uv run python -m scripts.solutions.v2_img3.train \
  --images-dir /data/images --check-only

uv run python -m scripts.solutions.v1_ocr.train \
  --images-dir /data/images \
  --ocr-cache /data/easyocr_ru_en.jsonl \
  --check-only
```

## Сборка submission

Training и submission pipeline независимы. Builder проверяет training manifest,
SHA-256 адаптеров и EasyOCR-весов, allowlist runtime-файлов, версии vendored
пакетов, безопасные ZIP-пути, дубли, CRC и финальные хеши.

```bash
uv run python -m scripts.solutions.v1_ocr.build_submission
uv run python -m scripts.solutions.v2_img3.build_submission
```

Результаты:

```text
artifacts/submissions/e-cup-gemma-lora-ocr-v1-comments-v4.zip
artifacts/submissions/e-cup-gemma-lora-v2-img3-comments-v4.zip
```

В контейнере Gemma должна быть доступна офлайн по пути
`$SHARED_MODELS_PATH/google/gemma-4-E4B-it`. Точка входа обоих архивов:

```bash
python -u run.py \
  --test_data_path /data/test.csv \
  --output_path /output/submission.csv
```

Изображения читаются из `/data/images/<id>/`; сеть во время инференса не нужна.

## Статическая проверка

```bash
uv run ruff check scripts tests
uv run ruff format --check scripts tests
uv run python -m compileall -q scripts tests
uv run python -m unittest discover -v
```

Те же CPU-проверки и smoke-check публичных CLI автоматически выполняются в
GitHub Actions на Python 3.12 без скачивания Gemma, изображений и CUDA-весов.
