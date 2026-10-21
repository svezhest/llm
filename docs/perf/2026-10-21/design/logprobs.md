# Logprobs: teacher forcing по промпту + logprobs сгенерированных токенов

Спецификация (2026-10-21). Заказчик: атрибуция частей промпта по записанным
трассам агента (AppWorld): берём трассу (messages system/user/assistant/tool,
10–30K токенов), меняем кусок промпта и одним прямым проходом считаем, как
изменилась вероятность уже записанных ответов модели — без генерации, тысячи
запросов, большинство — варианты одного начала (кэш префикса обязателен).

## Инварианты

1. **Генерация без флагов побайтно прежняя.** При `logprobs=false` и
   `prompt_logprobs=false` граф префилла и декода не меняется ни одной операцией
   (`_bounds`, слэбы, `skip_logits=True`, семплер). Повтор журнала даёт тот же ответ.
2. **Teacher forcing == генерация.** Для ответа, сгенерированного при T=0
   (presence 0), сумма log p его токенов в режиме prompt_logprobs (messages +
   этот ответ как последнее assistant-сообщение) равна сумме `logprobs` при
   генерации с точностью до численной погрешности (префилл-ядра vs декод-ядра:
   pfqmm/qmm с M строками против qmv; на CPU-моделях тестов — ~1e-3 на сумму).
   EOS (`<|im_end|>`) входит в обе суммы.
3. **Logprobs — «сырые»**: от логитов модели до penalties / temperature / масок
   (vLLM `logprobs_mode=raw_logprobs`), fp32 `logit − logsumexp`.
4. **Кэш префикса работает как обычно** (lookup по чекпоинтам, store, chain,
   partial store, eviction). Запрос с `prompt_cache=false` пропускает lookup
   (холодный префилл), но сохраняет запись.
5. Токенизация и чат-шаблон — те же, что при генерации (`_render`, тот же
   `apply_chat_template`, тот же `enable_thinking` default). История
   assistant-сообщения рендерится шаблоном Ornith как
   `<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}<|im_end|>\n` —
   при пустом reasoning это ровно строка промпта генерации с `enable_thinking=false`
   (`<think>\n\n</think>\n\n`) + content + `<|im_end|>`, при reasoning — строка
   промпта с thinking (`<think>\n`) + то, что модель сгенерировала.

## Позиции

Промпт `ids[0..n)`. Префилл обрабатывает `ids[:n-1]` слэбами от `start`
(позиция возобновления из кэша, 0 при холодном) — последний токен `ids[n-1]` —
первый вход декода. Логиты позиции j дают log p токена j+1. Значит:

- scored позиции: `first = start + 1` … `n − 1` включительно (последний токен
  промпта оценён логитами последнего слэба — ВСЕГДА, и при max_tokens 0, и при ≥ 1).
- позиция 0 — никогда; позиции ≤ start при хите кэша — нет (логиты позиции
  start − 1 не хранятся). Клиент: базовый запрос с `prompt_cache=false`,
  варианты — с кэшем; до точки расхождения значения совпадают по каузальности.
- `max_tokens: 0` (только вместе с `prompt_logprobs`): префилл как обычно,
  store записи как обычно (ids[:n-1] + end state), ряд завершается `length`
  до первого декод-шага (в `_iterate` проверка `r.n == max_tokens` стоит до
  `_step`), gen_tokens 0, Token-событий нет; reply-KV ничего не добавляет.

## Префилл с логитами (worker, `_prefill`)

После `lm(..., skip_logits=True, return_hidden=True, ...)` (mlx-vlm 0.7.6
`LanguageModel.__call__`: `return_hidden` → `out.hidden_states[-1]` = финальный
`out` после norm, то, что ест `lm_head`; проверить, что это так и для
`tie_word_embeddings`), после `mx.eval([c.state for c in cache])` (порядок и
граф кэша не меняются) — чанками по `QWEN_LOGPROBS_CHUNK` (256) строк:
`lg = lm_head(h[:, a:b])` → `lse = logsumexp(lg.astype(fp32))`,
`tgt = take_along_axis(lg, targets)` → `lp = tgt.astype(fp32) − lse`, eval
чанка (транзиент 256 × 248K × (2+4) байт ≈ 0.38 GB, 8 синков на слэб — ничто
против секунд слэба). Цели строки j слэба с offset pos: `input_ids[0, j+1]`
(для последней строки слэба — первый токен следующего слэба / последний токен
промпта: он всегда есть, цикл обрабатывает `shape[1] − 1`). lm_head над 2048
строками ≈ +10–12 % FLOPs слэба — только для запросов с prompt_logprobs.
Vision-префилл: `prompt_logprobs` с картинками → 400 в Check (text-only).
DFlash (`_spec`): запрос с logprobs / prompt_logprobs не идёт в DFlash (как
grammar). Partial store / cancel: как обычно, logprobs отбрасываются.

## Декод (worker, `_step` / `_iterate`)

Для рядов с `logprobs`: в `_step` на сырых `x` (до `_logits`): `lp =
(x.astype(fp32) − logsumexp)`, `row.lp = take(lp, y)` (lazy, в тот же
`async_eval`, что и `y`), top-k: `idx = argpartition(−lp, k)[:k]` → отсортировать
по значению, `row.top = (idx, lp[idx])` lazy, тот же async_eval. Чтение после
`r.y.item()` — `r.lp.item()`, `r.top` → `tolist()` — ТОЛЬКО у рядов с logprobs
(иначе ни одной новой операции в графе). Никаких новых op над выходами шага до
постановки следующего шага (step-ahead, см. CLAUDE.md «Decode step profile»):
lp/top входят в граф шага, читаются после. Токен EOS, на котором ряд остановился
(`why == "stop"`), тоже получает lp и уходит в событие (клиенту нужна сумма с
EOS). В `req.out` кладём ("tok", tok, shown, lp, top) или отдельную запись —
хэндлер набирает `pb.TokenLogprob` и прикрепляет их к ближайшему Token-событию
(`lp` не выровнен с `text`; допустим Token с пустым text и непустым lp — в
конце потока, на EOS, на придержанном stop-хвосте). Строка токена —
`tokenizer.decode([id])`.

## Spans (assistant-сообщения)

Worker, после токенизации: ищем блоки `<|im_start|>` + токен `assistant` в ids;
k-й блок ↔ k-е сообщение с role assistant в `req.messages` (по порядку; при
несовпадении числа блоков и сообщений — warning в лог, zip по короткому).
`end` = позиция `<|im_end|>` блока + 1. `start` (начало сгенерированного):
после `<|im_start|>`, `assistant`, `\n`; затем если следуют ровно
[`<think>`, `\n\n`, `</think>`, `\n\n`] (пустой reasoning = промпт генерации без
thinking) — после них; иначе если [`<think>`, `\n`] — после них (thinking:
reasoning + `\n</think>\n\n` + content сгенерированы моделью); иначе сразу.
Проверить фактические id через tokenizer на реальном шаблоне (
`<think>\n\n</think>\n\n` → [`<think>`, `\n\n`, `</think>`, `\n\n`]?) — тест на
Ornith-токенизаторе (`weights/ornith15-9b-mlx-8bit/tokenizer.json` есть локально,
без GPU: tokenizers-only тест, skip если нет файла).

## Wire (proto, уже добавлено)

`Params.logprobs / top_logprobs / prompt_logprobs / prompt_cache`;
`Token.lp: repeated TokenLogprob{token_id, token, logprob, top[]}`;
`Event.prompt_logprobs: PromptLogprobs{first, tokens[] (id+строка на каждую
позицию), logprobs[] (для позиций first..n−1), spans[]}` — одно событие перед
Done (и при max_tokens 0, и при ≥ 1; при cancel/error — нет).

## HTTP (gateway)

Запрос (`/v1/chat/completions`): `logprobs: true`, `top_logprobs: 0..20`
(требует logprobs, как OpenAI/vLLM); `prompt_logprobs: true` (bool; vLLM
принимает int — число top-k по промпту; у нас только bool/0/1, int > 1 → 400
«only true/false»); `max_tokens: 0` допустим только с `prompt_logprobs: true`
(иначе прежняя ошибка vLLM); `prompt_cache: false` (расширение, default true);
`prompt_logprobs` + images → 400 (worker Check). `echo` остаётся unsupported.

Ответ (JSON): `choices[0].logprobs = {"content": [{token, logprob, bytes,
top_logprobs:[{token, logprob, bytes}]}]}` по OpenAI (все сгенерированные
токены по порядку: reasoning, ответ, EOS; `bytes` = UTF-8 строки токена);
без `logprobs` — `"logprobs": null` (как OpenAI). Top-level (как у vLLM
`prompt_logprobs`):
```
"prompt_logprobs": [null, null, {"token": "…", "token_id": 123, "logprob": -2.3}, …]   // n элементов
"prompt_logprobs_messages": [{"message": 3, "role": "assistant", "start": 1200, "end": 1500,
                               "tokens": 300, "scored": 300, "logprob": -45.2}]
```
`tokens = end − start`, `scored` — сколько из них оценено (null — не оценены:
хит кэша глубже начала), `logprob` — сумма по оценённым. SSE: `logprobs` в
чанках — токены, прочитанные с прошлого чанка (НЕ выровнены с delta текста;
чанк может нести logprobs при пустой delta); `prompt_logprobs` +
`prompt_logprobs_messages` — отдельный чанк с `choices: []` перед финальным
(finish) чанком. `usage.prompt_tokens_details.cached_tokens` как раньше (= first − 1
при хите; 0 при холодном).

## Тесты

Worker (CPU, tiny модели, `tests/test_logprobs.py`): (a) генерация T=0 с
logprobs → токены + lp; teacher forcing тех же messages + ответ (как assistant,
`add_generation_prompt=false`) с `max_tokens=0` → сумма по span == сумма при
генерации (atol ~1e-3·N на CPU fp16; сравнить также первые токены по одному);
(b) при выключенных флагах выход побайтно равен прежнему (сравнить ids
генерации с флагами и без: равны; и что при флагах off `row.lp is None`, в
графе нет op — через отсутствие атрибутов/счётчик вызовов lm_head);
(c) хит кэша: второй такой же запрос — `first` = глубина хита + 1, значения на
scored позициях равны холодным (bit-exact на CPU — проверить, иначе atol);
`prompt_cache=false` → first = 1 и те же значения; (d) spans: thinking-off и
thinking-on формы заголовка, несколько assistant-сообщений, tool-сообщения
между ними; (e) max_tokens 0: finish length, gen_tokens 0, store записи как у
генерации (та же длина ids[:n−1]); (f) B=2 батч: два ряда с logprobs и без —
lp есть только у одного, token draws как без флагов; (g) top_logprobs: k
значений, убывают, содержат выбранный токен при T=0 на первом месте.
Gateway (Rust unit tests в request.rs/api.rs): валидация (max_tokens 0 с/без
prompt_logprobs, top_logprobs без logprobs, prompt_logprobs int), сборка JSON из
pb-событий (Acc), SSE-чанки.

Реальная модель (позже, GPU по согласованию, скрипт
`scripts/perf/logprobs/check_teacher_forcing.py` через HTTP): генерация T=0 с
logprobs → teacher forcing → |Δ суммы|, и побайтное равенство ответа с флагами
и без (один и тот же воркер, B=1).
