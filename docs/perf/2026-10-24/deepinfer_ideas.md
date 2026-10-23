# DeepInfer → llm: идеи и план переноса (2026-10-24)

Источник: `~/Git/DeepInfer` (deepseek-ai/DeepInfer @ 9bd9cf2, 2026-10-21; форк vLLM под
DeepSeek-V4.1-Flash, SM100/Ascend; `vendor/*` — пустые сабмодули). Пять read-only разборов
(планировщик, спекуляция, KV/кэш, детерминизм/сэмплер, ядра). Пути DeepInfer ниже — от корня
клона. Цифры «у нас» — из CLAUDE.md и `docs/perf/2026-10-1{2,6,7}/`.

## 0. Аномалия «~200 мс на шаг» и метрика шага

`stats_6proc_steady.txt`: rows=6 prefilling=n → 5.08 шага/с (~197 мс), микробенч B=8 34–52 мс.
Причина, скорее всего, в логе: `rows`/`prefilling` в строке `batch:` — снимок в момент записи,
`steps/s` — за 10-секундное окно, в котором мог пройти 4–9-секундный слэб префилла. Плюс своп
5–6.7 GB в том прогоне. Сделано: строка `batch:` получила `step_ms` (стена декод-тиков на шаг),
`gbps` (байты, которые шаг обязан прочитать: веса + KV + 2× GDN, `_step_bytes`, / стена),
`decode_s` / `prefill_s` за окно; `analyze_worker_log.py` их печатает. Аналог DeepInfer:
`csrc/engine/flops_estimator.hpp`, `deepinfer/engine/stats_logger.py:198-205,274-288,315-337`.
Проверка — под нагрузкой (replay AppWorld), GPU-фаза.

## 1. Batch-invariant режим (детерминизм без MAX_BATCH=1/2)

DeepInfer: `DEEPINFER_BATCH_INVARIANT` (`environ.py:197`, `worker/runtime_config.py:67-69`,
`layers/attention_backend.py:85,141`); тест `tests/models/test_batch_invariant.py:52-108`
(64 промпта батчем vs по одному, принудительные токены `golden_token_ids`
(`sampling_params.py:60,83`, `layers/sampler.py:482-490,650-683`), logprobs и top-5 через `==`).
У нас: `mrqmm._config` — KS=8 при ≤ 8 строках (MB=1), KS=4 при 9..16 (MB=2) → порядок
редукции по K зависит от числа строк; B ≤ 2 — MLX qmv; cascade — своя композиция внимания.
Внимание (per-row B=1 вызовы), GDN в стеке, нормы, сэмплер уже построчно инвариантны.
Идея: под knob — один фиксированный KS на (K, N) для 1..16 строк, mrqmm и при B ≤ 2 (включая
GDN-fuse), cascade off → строка в батче из N == та же строка при B=1 бит-в-бит (9B; 27B/35B —
модули, которые mrqmm не берёт, остаются qmv/qmm: для них отдельный вопрос, см. mlx-vlm
`quantized_verifier.py` `optimized_affine_linear` / `exact_quantized_switch_linear`).
Цена ~+2 мс на шаг при B=1 (mrqmm 32 vs qmv 30 мс линейных). Выигрыш: replay B=2 сейчас −11 %
req/min и p50 22 vs 15 с (`docs/perf/2026-10-17/replay_batch1.md`). Строгая воспроизводимость
по-прежнему требует `QWEN_PROMPT_CACHE_REPLY=0` (reply KV из декод-ядер ≠ cold prefill).

## 2. Дисковый уровень prompt cache

DeepInfer (`doc/inference_deployment_qa.md:129-191`; хуки `engine/llm_engine.py:787-846,930-937`,
сам kvstore не в релизе): блочный KV на SSD, TTL 3 дня для агентного трафика, KV декода тоже
пишется, хит по цепочке хэшей, подкачка асинхронно, перекрыта с forward.
У нас: вытеснение = удаление. 17.10: 8 процессов — 8-я история вытесняется каждый круг, MISS
52–82 с; хвост — MISS 54.6K/58.4K = 181/203 с; своп 5–6.7 GB. `_Seg` неизменяемы, fp16 из байтов
бит-в-бит. Запись 1–2.5 GB ~0.3–0.8 с, чтение ~0.2–0.5 с (SSD M1 Max ~5 GB/s, не мерено).
Писать при вытеснении (не при каждом store), одним файлом на запись, лимит + LRU на диске,
подгружать на потоке-обработчике (константный mx.array, как `to_mlx` для картинок) пока запрос
в очереди; загруженная запись проходит обычные `_fits`/budget; инвалидировать при старте
(или ключ = модель + флаги, влияющие на числа + версия кода).

## 3. DFlash verify через mrqmm

DeepInfer: verify-токены — обычные строки плоского батча (`layers/sampler.py:430`).
У нас verify — `(1, T, K)`, точные multi-row qmv до `QWEN_VERIFY_EXACT_MAX_T=8` (63–73 мс
линейных при T=8), выше `quantized_matmul` (80–90 мс при 8–16). mrqmm: 44 мс при 8 строках,
67–71 при 16 (`docs/perf/2026-10-12/microbench.md` 1b). `(1,T,K)` → `(T,1,K)` в mrqmm: раунд
~85 → ~65 мс, solo-декод 48 → ~63–68 ток/с, блок 16 становится выгоден; lm_head драфтера через
mrqmm −4 мс. Цена: DFlash ≠ plain greedy побайтно (класс батч-декода, top-1 ≥ 99.7 %) → knob.
Новый аргумент к dead end (3) («≤ 1.3× запаса» — до mrqmm).

## 4. Полоса для коротких префиллов

DeepInfer: уровни по длине контекста с квотами (QA «Prefill Load Balancer»); на GPU — строгий
FIFO (`csrc/engine/scheduler.hpp:369-371,386-388`). У нас FIFO без обгона, один `task`;
в патологии 110K короткие (10–27K) запросы получали таймаут 600 с в очереди (19 cancel за 72 мин).
Идея: второй, приоритетный task для текстовых hit-запросов с суффиксом ≤ 1 слэба, исполняемый
между слэбами длинного префилла (память под два частичных кэша; `_rope_deltas` — только текст).

## 5. Вытеснение строки в кэш вместо обрезки `length`

DeepInfer: `_preempt` → `reset_for_recompute` + `waiting.push_front`, жертва — строка с самым
длинным выводом (`scheduler.hpp:622-643,766-790`). У нас растущая сверх бюджета строка
заканчивается `length` (`_regrow`) — сломанный tool call. Идея: сохранить prompt + выданные
токены через механизм reply-KV / partial store, вернуть запрос в голову очереди, продолжить с
хита; клиент видит один непрерывный поток.

## Дешёвые мелочи

- **forbid-sync guard** (DeepInfer `utils.py:119`, `worker/model_runner.py:324,359`): debug-env,
  на время сборки шага (`_step`) запрещающий `.item()` / `.tolist()` / `mx.eval` / `np.asarray`
  → регрессия вроде старого `.tolist()` ловится тестом.
- **`QWEN_PROMPT_CACHE_CHAIN_TAIL=0`**: DeepInfer хранит одно состояние на конец запроса
  (QA:165). Прогнать `docs/perf/2026-10-16/chain_ckpts_sim.py`: байты и позиции resume.
- **top_p / min_p по k выжившим** при top_k > 0 вместо argsort по 248K (DeepInfer
  `tilelang_ops/cuda/fused_sample.py:54-161` — radix-select). Только T > 0.
- **Инъекция спецтокенов**: DeepSeek токенизирует сегментами, чтобы `<|im_start|>` и др. из текста
  tool output не стали спецтокенами. Проверить наш `_render` + `encode`.

## Cascade: ядро с MMA по fp16-префиксу

DeepInfer/FlashMLA: Q-головы одной KV-головы набиваются до M MMA-тайла
(`layers/attention_backend.py:5-24`), два источника KV с общим softmax (`:125-142`), план
разбиения один на шаг для всех слоёв (`v4_modules/attn.py:241,663`, `input_metadata.py:388`).
У нас `cascade.py`: fp32-префикс (64 KiB/токен), `mx.matmul`/`logsumexp`/`softmax` чанками по
1024 с материализованными scores (4, c, 64, 1024) fp32 на слой, суффиксы отдельно, merge.
Идея: `metal_kernel` — префикс читается в fp16 прямо из буфера строки-источника / сегмента,
simdgroup MMA с fp32-аккумулятором, online softmax, scores в регистрах; элементы работы = чанки
префикса на всю группу (16 строк × G4 = 64 строки запроса на KV-голову) + чанки суффикса на
строку; merge как pass 2 `rattn.py`. Прикидка 32K × 16: ~69 GFLOP → 9–11 мс, чтение префикса
1.07 GB → ~3 мс (сейчас 18–19 мс в микробенче); −1 GiB на группу 32K, нет транзиента +2.2 GB.
Открытый вопрос: 92 vs ~62 мс на шаг при 32K × 16 (`docs/perf/2026-10-12/cascade_gpu.md`).

## Позже / не берём

Позже: KV в q8 через свой вариант `rattn` (`rattn.py` пока не интегрирован), батчевая varlen-
спекуляция (DSpark: `csrc/engine/mtp_varlen.hpp:150-267`; при 4–6 строках вырождается в «добить
до 8 строк mrqmm»), paged KV pool, тёплый контекст драфтера после хита.
Неприменимо: SWA bounded replay / YOCO (GDN-состояние не оконное), index/sparse attention,
MLA/FP8/FP4, EP/EPLB, CUDA graphs, P/D-дезагрегация.
