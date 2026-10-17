# SP5: окружение узла (uv + PyTorch, установка, curl, WSL2, ядра, macOS, Docker/K8s)

Исследование для плана SP5: как присоединить новую машину (Linux, macOS, Windows через WSL2) к хабу одной командой.
Дата: 2026-10-18. Всё проверялось на машине владельца (Linux 6.18 в WSL2 `WSL 3.0.3.0` поверх Windows 10 build 19045, 20 логических ядер, RTX 4070 Ti, драйвер 610.88), без установки чего-либо в систему: временные файлы в `/tmp/claude-1000/-home-viv-dev-repos-Colosseum/b41cd787-f9ea-471b-9664-8578a7f7b4b1/scratchpad/research/` (там же исходники, тестовые серверы и скрипты из этого документа).

Метки достоверности в тексте:

- **[В]** проверено локально командой, которая приведена рядом (версии: uv 0.13.1, curl 7.81.0/OpenSSL 3.0.2, Python 3.12.15 и 3.13.7, grpcio 1.84.0, kubeconform v0.8.0);
- **[И]** проверено по первоисточнику (исходный код на заданном теге, официальная документация, changelog, API индекса), URL рядом;
- **[Вывод]** моё заключение, прямо в источнике не сказано или не проверено на целевой платформе (особенно macOS: Mac под рукой нет).

## Сводная таблица

| № | Вопрос | Ответ | Достоверность | Источник |
|---|---|---|---|---|
| 1 | `uv pip install --torch-backend auto` / `UV_TORCH_BACKEND` | Есть. Введено в uv 0.6.9 (2025-03-20) как preview; метка preview снята в 0.7.14 (2025-06-23). В 0.13.1 работает без `--preview`. Только в `uv pip` (для `uv sync`/`add` есть `[tool.uv] torch-backend`, с 0.9.18). Работает с точной версией `torch==X.Y.Z` | В + И | changelogs uv, `crates/uv-torch/src/{backend,accelerator}.rs@0.13.1` |
| 1 | Как `auto` находит драйвер | По порядку: `UV_CUDA_DRIVER_VERSION` -> `/sys/module/nvidia/version` -> `/proc/driver/nvidia/version` -> `nvidia-smi --query-gpu=driver_version --format=csv,noheader` -> `rocm_agent_enumerator` -> PCI Intel GPU. В WSL2 работает только последний путь через `nvidia-smi`, а он не в `PATH` чистого окружения | В + И | `accelerator.rs`; тест с `env -i` |
| 1 | Выбор индекса | Опрашивает ВСЕ совместимые с драйвером индексы (newest CUDA ... cpu) и берёт лучшую версию; пин `torch==2.14.1` при драйвере 535 даёт `+cu126`, при 610 даёт `+cu132`, при 470 молча даёт `+cpu` | В | dry-run, см. §1.4 |
| 1 | Подводные камни | `auto` не смотрит compute capability (GTX 1080 Ti получает cu13x-колесо, которое не запустится): uv issue #14742, открыт; нужно правило на стороне loader | И + В | https://github.com/astral-sh/uv/issues/14742 |
| 2 | Установка uv | `curl -LsSf https://astral.sh/uv/0.13.1/install.sh \| sh`; нужны `curl` или `wget`, `tar`, `uname mktemp chmod mkdir rm grep cat`; `UV_UNMANAGED_INSTALL=<dir>` ставит без правки профилей и без receipt-файла; `UV_INSTALL_DIR` пишет `~/.config/uv/uv-receipt.json` (!); на macOS нет `sha256sum`, проверка контрольной суммы тихо пропускается | В + И | скрипт `install.sh` 0.13.1 |
| 3 | Управляемый Python | `uv venv --python 3.12` скачивает python-build-standalone (29 MiB, 8.8 с), по умолчанию uv ПРЕДПОЧИТАЕТ managed-Python системному; offline или `UV_PYTHON_DOWNLOADS=never` даёт понятную ошибку; venv ссылается на `UV_PYTHON_INSTALL_DIR` (его нельзя удалять) | В | локальные запуски |
| 3 | Окружение из списка | `uv pip install --python ENV/bin/python -r requirements.txt -c constraints.txt --torch-backend auto --exact`; `uv pip sync` НЕ подходит (не ставит зависимости торча: нужен полный замороженный список, а `nvidia-*`/`triton` зависят от бэкенда) | В | dry-run |
| 4 | `--pinnedpubkey sha256//` | OpenSSL/GnuTLS/wolfSSL/mbedTLS/Schannel; sha256 с 7.44.0 (в OpenSSL-коде есть и для LibreSSL). Secure Transport поддерживал, но удалён из curl 8.15. macOS `/usr/bin/curl` 8.7.1: LibreSSL по умолчанию + SecureTransport, фича `HTTP2` есть; сам `--pinnedpubkey` на Mac НЕ проверен | В (Linux) / И + Вывод (macOS) | https://curl.se/libcurl/c/CURLOPT_PINNEDPUBLICKEY.html; исходники curl 8.7.1; https://curl.se/mail/lib-2025-01/0048.html |
| 4 | gRPC unary через curl | Работает: `curl --http2 -k --pinnedpubkey ... -X POST -H 'content-type: application/grpc' -H 'te: trailers' --data-binary @-` с 5 нулевыми байтами; БЕЗ `te: trailers` grpc-сервер отвечает `INTERNAL_ERROR` (curl exit 92); ответ идёт с 5-байтовым префиксом; ошибки приходят как HTTP 200 + `grpc-status` в заголовках/трейлерах, так что `curl -f` ошибок не ловит | В | эксперимент с grpcio 1.84.0 |
| 4 | Stock curl с HTTP/2 | Ubuntu 22.04 (7.81) [В]; Debian 11 (7.74.0) и 12 (7.88.1): `libcurl4` зависит от `libnghttp2-14` [И]; Ubuntu 20.04 (7.68.0) версия подтверждена Launchpad, HTTP/2 [Вывод] | В / И / Вывод | packages.debian.org, Launchpad API |
| 5 | Режим сети WSL2 | `wslinfo --networking-mode` (WSL 2.0.4, 2023-10-04); у нас вернул `nat`, хотя маршрут по умолчанию идёт через `wsltap` (Consomme-подобный режим); значит на `wslinfo` одном полагаться нельзя. Хост владельца - Windows 10 (19045): зеркальный режим и Hyper-V firewall там недоступны | В + И | MS Learn, GitHub releases WSL 2.0.4 |
| 5 | Входящие в NAT-режиме | `netsh interface portproxy add v4tov4 ...` + `New-NetFirewallRule ...`, оба с повышенными правами (проверено: без админа `netsh` отвечает "требуется повышение прав") | В + И | MS Learn `wsl/networking` |
| 5 | Mirrored | `.wslconfig` `[wsl2] networkingMode=mirrored` (Windows 11 22H2+), плюс `Set-NetFirewallHyperVVMSetting`/`New-NetFirewallHyperVRule` с VMCreatorId `{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}` | И | MS Learn |
| 5 | Tailscale в WSL2 | Рабочая альтернатива, но НЕ ставить одновременно на Windows и в WSL2; MTU 1280 в WSL2 (tailscaled поднимает до 1340) | И (через поиск, страницу tailscale.com открыть не удалось) | https://tailscale.com/docs/install/windows/wsl2 |
| 6 | Ядра без psutil | `os.sched_getaffinity(0)` + уникальные `thread_siblings_list` + потолок cgroup `cpu.max`/v1 quota; `os.process_cpu_count()` (3.13) учитывает только affinity и `PYTHON_CPU_COUNT`, но НЕ cgroup. В WSL2 топология виртуальная (10 "физических" у 12700KF вместо 12) | В + И | код в §6 |
| 7a | Смерть ребёнка с родителем без `PR_SET_PDEATHSIG` | Переносимо: pipe-lifeline или `multiprocessing.parent_process().sentinel` (оба проверены на Linux); `select.kqueue` `KQ_FILTER_PROC`+`KQ_NOTE_EXIT` есть в документации, на Mac не запускался | В (Linux) / И | §7.1 |
| 7b | Лимит пути Unix-сокета | Linux 107 символов + NUL (проверено), macOS 104 (по справочникам); `multiprocessing.connection` с `family="AF_UNIX"` и адресом `"\0name"` на Linux работает (3.12 и 3.13) | В / И | §7.2 |
| 7c | `/dev/shm` на macOS | Нет; обычная замена - mmap файла в `$TMPDIR` (диск, не RAM). `multiprocessing.shared_memory` на macOS идёт через POSIX `shm_open` (имя до ~31 байта) | Вывод + И | §7.3 |
| 7d | Resource tracker в 3.12 | Подтверждено: отдельный Python-процесс, открывший сегмент по имени, удаляет его при выходе (3.12.15 и 3.13.7 без `track=False`). Лечение: `track=False` (3.13+, gh-82300) или `resource_tracker.unregister(shm._name, "shared_memory")` на 3.12 (проверено) | В + И | §7.4 |
| 8 | Docker/K8s | `--shm-size` по умолчанию 64m; `--gpus` требует NVIDIA Container Toolkit; в WSL без интеграции `docker info` даёт rc=1 и сообщение про Docker Desktop; `kubeconform` - один бинарь (14 MB), офлайн-валидация проверена в сетевом namespace без сети; `emptyDir: {medium: Memory}` даёт tmpfs, учитывается в memory limit | В + И | §8 |

Два результата, которые меняют дизайн (подробно в «Выводах»):

1. **Один порт + mutual TLS + `curl`-установщик несовместимы в grpcio** [В]: `grpc.ssl_server_credentials(..., require_client_auth=False)` вообще не запрашивает клиентский сертификат (`openssl s_client`: "No client certificate CA names sent"), а `True` требует его от всех, в том числе от `curl` без сертификата. Опционального режима в Python API нет (`cygrpc.server_credentials_ssl(pem_root_certs, pem_key_cert_pairs, force_client_auth)` принимает только bool).
2. **Старые GPU**: `--torch-backend auto` выбирает cu13x по драйверу, а cu13x не поддерживает compute capability ниже 7.5 (Maxwell/Pascal/Volta). Нужно правило в loader (§1.6).

---

## 1. uv и сборки PyTorch

### 1.1 Что есть и с какой версии

- `--torch-backend` / `UV_TORCH_BACKEND` появился как preview в uv 0.6.9 (релиз 2025-03-20, "Add experimental `--torch-backend` to the PyTorch guide"), `auto` добавлен в 0.6.9 (PR #12070), "Check all compatible torch indexes when `--torch-backend` is enabled" - 0.6.10 (2025-03-26, PR #12385). Источник: `changelogs/0.6.x.md` в https://github.com/astral-sh/uv, даты по GitHub Releases API. [И]
- Метка preview снята в 0.7.14 (2025-06-23, PR #14119; в том же релизе добавлены ROCm и XPU). [И] В uv 0.13.1 команда работает без `--preview` и без предупреждений. [В]
- `--torch-backend` в `[tool.uv]` принимается с 0.9.18 (2025-12-16, PR #17116); в `uv tool` - с 0.9.19. [И] Документация PyTorch-гайда: "At present, `--torch-backend` is only available in the `uv pip` interface." (https://docs.astral.sh/uv/guides/integration/pytorch/) [И]. В `uv sync --help` и `uv run --help` 0.13.1 флага нет, в `uv pip install --help` и `uv pip sync --help` есть. [В]
- Допустимые значения в 0.13.1 (`uv pip install --help`): `auto, cpu, cu132, cu130, cu129, cu128, cu126, cu125, ... cu80, rocm7.2 ... rocm4.0.1, xpu`. [В]
- Применяется к пакетам экосистемы: `torch, torchvision, torchaudio, triton, xformers, torchao, torchcodec, ...` (`TorchStrategy::applies_to`); остальные идут с обычного индекса (PyPI). [И] `backend.rs@0.13.1`

### 1.2 Как `auto` определяет драйвер

Из `crates/uv-torch/src/accelerator.rs` (тег 0.13.1) [И], по порядку:

1. переменная `UV_CUDA_DRIVER_VERSION` (переопределение);
2. `UV_AMD_GPU_ARCHITECTURE`;
3. файл `/sys/module/nvidia/version`;
4. файл `/proc/driver/nvidia/version`;
5. `nvidia-smi --query-gpu=driver_version --format=csv,noheader` (первая строка);
6. `rocm_agent_enumerator` (AMD);
7. `/sys/bus/pci/devices` - Intel GPU (класс display, vendor 0x8086) -> `xpu`;
8. Windows: дерево устройств (Intel).

Нет ни одного GPU - индекс `cpu`.

**WSL2** [В]: у нас нет ни `/sys/module/nvidia/version`, ни `/proc/driver/nvidia/version` (драйвер живёт в Windows), uv находит драйвер только через `nvidia-smi`. Лог: `DEBUG Detected CUDA driver version from nvidia-smi: 610.88`. Но `nvidia-smi` лежит в `/usr/lib/wsl/lib/`, и этот каталог попадает в `PATH` только из окружения, которое WSL-init формирует для сессии; чистое окружение его не видит:

```sh
env -i /bin/bash -lc 'which nvidia-smi'   # пусто
env -i /bin/sh -c 'command -v nvidia-smi' # пусто
# а с урезанным PATH uv молча выбирает CPU:
env PATH=$HOME/.local/bin:/usr/bin:/bin uv pip install --dry-run --torch-backend auto torch==2.9.0   # -> torch==2.9.0+cpu
```

Это случай `ssh` в WSL, `wsl.exe -e`, cron, systemd-юнит. Лечение: loader сам добавляет `/usr/lib/wsl/lib` в `PATH`, если там есть `nvidia-smi`, и/или экспортирует `UV_CUDA_DRIVER_VERSION`, полученный собственным зондом (§6.4). Тогда решение от `PATH` не зависит.

**macOS** [И]: для `Accelerator::Cuda` на macOS (и FreeBSD и др.) список индексов - только `cpu`; при отсутствии ускорителя тоже `cpu`. На macOS индекс `https://download.pytorch.org/whl/cpu/torch/` содержит колёса для macOS: у torch 2.14.1 и 2.13.0 это только `macosx_14_0_arm64` (Apple Silicon, macOS 14+; MPS входит в обычное arm64-колесо, [Вывод]). У 2.3.0 и новее колёс для Intel Mac нет вообще (последнее `macosx_10_9_x86_64` у 2.2.2). [В] (разбор HTML индекса). Вывод: Intel-Mac и macOS < 14 не смогут поставить торч хаба; loader должен это проверять заранее и писать человекочитаемую ошибку.

### 1.3 Таблица "драйвер -> CUDA" в uv

`LINUX_CUDA_DRIVERS` (`backend.rs@0.13.1`) [И]: `cu132`, `cu130` - драйвер >= 580; `cu129` ... `cu120` - >= 525.60.13; `cu118` ... `cu111` - >= 450.80.2; `cu110` - >= 450.36.6; дальше старые. Для Windows-драйверов (`WINDOWS_CUDA_VERSIONS`) пороги 528.33 и 452.39. Для WSL uv использует Linux-таблицу (ОС в смысле колёс - manylinux), а версия драйвера - Windows-версия из `nvidia-smi` [Вывод]; расхождение возможно только в узком окне 525.60-528.32 (практически неважно).

Первоисточник порогов: NVIDIA CUDA Toolkit Release Notes, Table 3 "CUDA Toolkit and Minimum Required Driver Version for CUDA Minor Version Compatibility": "13.x >= 580; 12.x >= 525 < 580; 11.x >= 450 < 525" (https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html). [И] Колёса cu12x работают на драйверах 525-579 благодаря minor-version compatibility, но часть возможностей (в том числе архитектура Blackwell) требует нового драйвера; для cu128 нужен >= 570.26.

### 1.4 Как uv выбирает индекс при пине версии (эксперимент)

В режиме `auto` uv запрашивает страницы `torch/` у всех индексов от нового CUDA до `cpu` (28 запросов при драйвере 610) и выбирает лучшую подходящую версию; если пин есть только в одном индексе, берётся он. Всё это `--dry-run`, torch не скачивался (кэш 7 MB):

```sh
export UV_CACHE_DIR=$PWD/uvcache
uv venv -p 3.12 venv312
uv pip install -p venv312/bin/python --dry-run --torch-backend auto "torch==2.9.0"          # драйвер 610.88 (реальный)
#   Selecting: torch==2.9.0+cu130   (в cu132 торча 2.9.0 нет; "Adding system dependency cuda@13.0")
UV_CUDA_DRIVER_VERSION=535.54.03 uv pip install -p venv312/bin/python --dry-run --torch-backend auto "torch==2.9.0"   # + torch==2.9.0+cu129
UV_CUDA_DRIVER_VERSION=470.1     uv pip install -p venv312/bin/python --dry-run --torch-backend auto "torch==2.9.0"   # + torch==2.9.0+cpu   (молча!)
uv pip install -p venv312/bin/python --dry-run --torch-backend auto "torch==2.14.1"                                     # + torch==2.14.1+cu132 + triton==3.8.0
UV_CUDA_DRIVER_VERSION=535.54.03 ... "torch==2.14.1"                                                                    # + torch==2.14.1+cu126
UV_CUDA_DRIVER_VERSION=575.0     ... "torch==2.14.1"                                                                    # + torch==2.14.1+cu126 (у 2.14.1 нет cu128/cu129)
uv pip install ... --torch-backend cu128 "torch==2.14.1"        # error: there is no version of torch==2.14.1 (фиксированный бэкенд без нужной сборки - жёсткая ошибка)
```

Вывод [В]:

- Пин вида `torch==2.14.1` (публичная версия, без локального `+cuXXX`) подходит: локальные теги выбираются бэкендом. Нельзя пересылать узлу `torch==2.14.1+cpu` из `pip freeze` хаба: `+cpu` зафиксирует CPU-колесо даже при `auto` (проверено: `req2.txt` с `torch==2.9.0+cpu` дал `+cpu` на машине с GPU).
- Если драйвер старый (< 525.60.13 для cu12x) или в подходящих индексах нет нужной версии, `auto` молча падает на `+cpu`. Loader после установки обязан проверить результат (`torch.__version__`, `torch.version.cuda`, `torch.cuda.is_available()`) и громко предупредить, если GPU есть, а торч CPU.
- Сборки, которые существуют для данной версии торча, различаются от релиза к релизу [В] (разбор индексов `download.pytorch.org/whl/<tag>/torch/` на 2026-10-18, колёса cp312):

| torch | индексы с колёсами |
|---|---|
| 2.14.1, 2.14.0 | cpu, cu126, cu130, cu132, xpu |
| 2.13.0 | cpu, cu126, cu129, cu130, cu132, rocm7.1, xpu |
| 2.12.1 | cpu, cu126, cu129, cu130, cu132, rocm7.1, xpu |
| 2.12.0 | cpu, cu126, cu130, cu132, rocm7.1, xpu |
| 2.11.0 / 2.10.0 | cpu, cu126, cu128, cu129, cu130, rocm7.1, xpu |

Linux x86_64 и aarch64 есть у всех CUDA-индексов; Windows (win_amd64) у cu126, cu130, cu132 (и cu128 у старых).

### 1.5 URL индексов и как узнать, какие сборки есть

Шаблон: `https://download.pytorch.org/whl/<tag>/` (`cpu`, `cu118`, `cu126`, `cu128`, `cu129`, `cu130`, `cu132`, `rocm7.1`/`rocm7.2`, `xpu`); страница пакета - `.../whl/<tag>/torch/` (отдаёт HTML, `Accept: application/vnd.pypi.simple.v1+json` проигнорирован, [В]). Список индексов с примерами - в гайде uv. [И]

Скрипт (stdlib, проверен на 2.14.1): `research` каталог scratchpad, файл `code/scan_torch_indexes.py`; суть:

```python
import re, urllib.parse, urllib.request
TAGS = ["cpu", "cu118", "cu126", "cu128", "cu129", "cu130", "cu132", "rocm7.1", "xpu"]
def builds(version, py="cp312"):
    out = {}
    for tag in TAGS:
        html = urllib.request.urlopen(f"https://download.pytorch.org/whl/{tag}/torch/", timeout=60).read().decode()
        names = (urllib.parse.unquote(n) for n in re.findall(r">(torch-[^<]+\.whl)<", html))
        plats = {m.group(1) for n in names
                 if (m := re.match(rf"torch-{re.escape(version)}(?:\+[^-]+)?-{py}-\w+-(.+)\.whl$", n))}
        if plats:
            out[tag] = plats
    return out
```

Дешевле и точнее для loader: `uv pip install --dry-run --torch-backend auto torch==X` и разобрать строку ` + torch==X+cuNNN` (около 8 с на холодном кэше, затем секунды). Официальная матрица "что публикуется": https://pytorch.org/get-started/locally/ и RFC-тикеты в `pytorch/pytorch` (на них ссылаются объявления на dev-discuss.pytorch.org).

### 1.6 Старые GPU: `auto` не смотрит compute capability

- uv issue #14742 "`--torch-backend=auto` ignores CUDA compute capability when installing `torch` for GTX 1080 Ti" (создан 2025-07-18, на 2026-10-18 открыт): после установки `RuntimeError: CUDA error: no kernel image is available`; текущая сборка поддерживает sm_75 sm_80 sm_86 sm_90 sm_100 sm_120. [И + В: поведение следует из таблицы драйверов, у нас Ada sm_89, воспроизвести нечем]
- PyTorch dev-discuss, "Notice: CUDA 12.6 wheels will no longer be published from PyTorch 2.15 (drops Maxwell / Pascal / Volta)" (2026-09-04, https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432): "CUDA 13.x does not support compute capabilities below Turing (sm_75), and CUDA 12.6 was the only remaining build covering them"; "PyTorch 2.14 is the last release to ship any CUDA 12.x wheels"; Maxwell sm_50, Pascal sm_60, Volta sm_70 - dropped after 2.15. [И]
- Правило для loader [Вывод]: если `min(compute_cap)` среди GPU < 7.5 и торч хаба <= 2.14 - передать `--torch-backend cu126` (если драйвер >= 525.60.13), иначе CPU с предупреждением. Торч хаба >= 2.15 для таких GPU готовых колёс не имеет. `compute_cap` есть в `nvidia-smi --query-gpu` (проверено: `--help-query-gpu` перечисляет `compute_cap`); на очень старых драйверах поля может не быть - тогда запрос повторить без него.

---

## 2. Неинтерактивная установка uv

Официальный однострочник: `curl -LsSf https://astral.sh/uv/install.sh | sh`, без curl - `wget -qO- https://astral.sh/uv/install.sh | sh`; закрепление версии: `https://astral.sh/uv/0.13.1/install.sh` (https://docs.astral.sh/uv/getting-started/installation/). [И]

Что делает скрипт (читал `https://astral.sh/uv/0.13.1/install.sh`, 2191 строка) [В/И]:

- требует: `uname mktemp chmod mkdir rm tar grep cat` и (curl или wget; snap-версия curl отвергается, если есть wget); `head`, `tail`, `base64` - в ветках для receipt/Windows;
- качает `uv-<triple>.tar.gz` с `https://releases.astral.sh/github/uv/releases/download/0.13.1` (запасной - `https://github.com/astral-sh/uv/releases/download/0.13.1`); переопределения: `UV_DOWNLOAD_URL`, `INSTALLER_DOWNLOAD_URL`, `UV_INSTALLER_GITHUB_BASE_URL`, `UV_INSTALLER_GHE_BASE_URL`, `UV_GITHUB_TOKEN`;
- контрольные суммы SHA-256 зашиты в скрипт для каждого артефакта; **проверка требует команды `sha256sum`, при её отсутствии пишет "skipping sha256 checksum verification" и продолжает** (на macOS `sha256sum` по умолчанию нет [Вывод]; `shasum -a 256` есть). Хаб, который сам отдаёт установщик, должен проверять сумму сам;
- каталог установки: `UV_INSTALL_DIR` (плоский), иначе `UV_UNMANAGED_INSTALL`, иначе `$XDG_BIN_HOME`, `$XDG_DATA_HOME/../bin`, `$HOME/.local/bin`;
- `UV_NO_MODIFY_PATH=1` (старый алиас `INSTALLER_NO_MODIFY_PATH`) - не трогать `PATH`/профили; `UV_UNMANAGED_INSTALL=<dir>` - "для эфемерных окружений": ставит в каталог и ничего не меняет в профилях; `-q`/`UV_PRINT_QUIET` - тише; `UV_DISABLE_UPDATE`.

Проверено локально в scratch (не затрагивая профили):

```sh
UV_UNMANAGED_INSTALL=$PWD/uvinst sh uv-install.sh   # -> uvinst/{uv,uvx} (37.8 MB), ~/.bashrc и ~/.profile не изменились, receipt не создан
UV_INSTALL_DIR=$PWD/uvinst2 UV_NO_MODIFY_PATH=1 sh uv-install.sh -q   # PATH не тронут, НО перезаписан ~/.config/uv/uv-receipt.json
```

**Побочный эффект, важный для узла** [В]: с `UV_INSTALL_DIR` установщик пишет receipt `~/.config/uv/uv-receipt.json` (`install_prefix` указывает на ваш каталог), что ломает `uv self update` у пользователя, у которого uv уже стоит в другом месте. Для приватного uv узла использовать `UV_UNMANAGED_INSTALL` (receipt не пишется). (При этом исследовании я перезаписал receipt владельца этой машины и вернул его: `install_prefix=/home/viv/.local/bin`, `modify_path=true`; остальные поля прежние.)

Альтернативы без доступа к `astral.sh` [И]:

- GitHub Releases, без скрипта: `https://github.com/astral-sh/uv/releases/download/0.13.1/uv-x86_64-unknown-linux-gnu.tar.gz` (+ `.sha256`, проверено: `eadd9d27...` для linux-gnu x86_64); артефакты: `uv-{x86_64,aarch64}-unknown-linux-{gnu,musl}`, `uv-{aarch64,x86_64}-apple-darwin`, `uv-*-pc-windows-msvc.zip` и др.; зеркало `releases.astral.sh/github/uv/...` отвечает 200 [В];
- `pip install uv` / `pipx install uv` (PyPI: `https://pypi.org/pypi/uv/0.13.1/json` отвечает 200 [В]; колёса содержат бинарь);
- хаб сам раздаёт `uv` для платформы узла (он может положить tar.gz в свою gRPC-раздачу и проверить SHA-256 из `dist-manifest.json`): это единственный вариант без интернета на узле; Python в этом случае тоже нужно привезти (§3, `UV_PYTHON_INSTALL_MIRROR=file://...`).

---

## 3. Управляемый Python и сборка окружения

### 3.1 Поведение (все пункты [В], uv 0.13.1)

- `uv venv --python 3.12 ENV`: если подходящего Python нет, uv скачивает python-build-standalone: `Downloading cpython-3.11.17-linux-x86_64-gnu (download) (29.4MiB)`, 8.8 с до готового venv, на диске 90 MB. Установка идёт в `UV_PYTHON_INSTALL_DIR` (по умолчанию `~/.local/share/uv/python`), архивы кэшируются в `UV_PYTHON_CACHE_DIR`, пакеты - в `UV_CACHE_DIR`.
- По умолчанию uv ПРЕДПОЧИТАЕТ управляемый Python системному: при системном 3.11.16 и пустом каталоге управляемых он выбрал скачанный 3.11.17. Это даёт воспроизводимость, но также скачивание. Флаги: `--managed-python`, `--no-managed-python`, `--python-preference only-managed|managed|system|only-system`, `UV_PYTHON_DOWNLOADS=never` / `--no-python-downloads`.
- Нет интернета и нет подходящего Python: `error: No interpreter found for Python 3.14 in managed installations or search path` + `hint: A managed Python download is available for Python 3.14, but uv is set to offline mode` (`UV_OFFLINE=1`) или `... but Python downloads are set to 'never'`.
- venv ссылается на конкретный каталог управляемого Python (`pyvenv.cfg: home = .../pyinst/cpython-3.11-linux-x86_64-gnu/bin`, `bin/python` - симлинк). Каталог `UV_PYTHON_INSTALL_DIR` должен жить столько же, сколько venv: держать его внутри `COLOSSEUM_HOME`.
- Зеркало/офлайн-источник дистрибутивов: `UV_PYTHON_INSTALL_MIRROR` (в т.ч. `file://`), `uv python install --mirror` (https://docs.astral.sh/uv/reference/environment/). [И]
- Кэш uv и целевое окружение должны быть на одной ФС, иначе вместо hardlink'ов идёт копирование (`UV_LINK_MODE=copy` убирает предупреждение); в WSL не класть окружение на `/mnt/c`. [Вывод, из документации `UV_LINK_MODE`]
- Окружение процесса: на машине владельца (ROS) задан `PYTHONPATH=/opt/ros/iron/lib/python3.10/site-packages`, и он **попадает в `sys.path` нового venv** [В]. Python с `-I` его игнорирует, но Colosseum рассчитывает на cwd в `sys.path` у дочерних процессов, поэтому надёжнее чистить окружение: `unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE VIRTUAL_ENV; export PYTHONNOUSERSITE=1`. (Это же причина префикса `env -u PYTHONPATH` в CLAUDE.md.)

### 3.2 `uv pip install -r` против `uv pip sync`

- `uv pip sync FILE`: ставит ровно перечисленное и НЕ разрешает зависимостей. [В]: `req1.txt` с `torch==2.9.0` и `numpy==2.0.2` дал "Would install 2 packages" (торч без sympy, networkx, nvidia-*). Значит нужен полный замороженный список, а у торча набор `nvidia-*`, `triton` зависит от бэкенда (CUDA 12 против 13, CPU вообще без них). Непереносимо.
- `uv pip install -r requirements.txt -c constraints.txt --exact`: разрешает зависимости, `-c` фиксирует версии всего остального, `--exact` удаляет лишнее (проверено: `Uninstalled 2 packages`). Рекомендация [Вывод]:
  - `requirements.txt`: прямые зависимости Colosseum + `torch==<публичная версия хаба>` (без `+cpu`/`+cuNNN`);
  - `constraints.txt`: `name==version` из `pip freeze` хаба для ВСЕХ пакетов, кроме `torch`, `triton`, `nvidia-*`, `cuda-*`, `pytorch-triton*` (зависят от бэкенда); локальные версии `+...` срезать; редактируемые/VCS-пакеты (сам `colosseum`) исключить (код привозит хаб).

### 3.3 Готовая последовательность (проверенные фрагменты + предложение)

```sh
export COLO_HOME="${COLO_HOME:-$HOME/.colosseum}"
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE VIRTUAL_ENV; export PYTHONNOUSERSITE=1
export UV_CACHE_DIR="$COLO_HOME/uv-cache" UV_PYTHON_INSTALL_DIR="$COLO_HOME/python" UV_NO_PROGRESS=1 UV_NO_MODIFY_PATH=1
[ -x /usr/lib/wsl/lib/nvidia-smi ] && PATH="$PATH:/usr/lib/wsl/lib"          # WSL2: драйвер в Windows
UV="$COLO_HOME/uv-bin/uv"
[ -x "$UV" ] || curl -LsSf https://astral.sh/uv/0.13.1/install.sh | UV_UNMANAGED_INSTALL="$COLO_HOME/uv-bin" sh
"$UV" venv --python 3.12 "$COLO_HOME/env/<hub-id>"                            # major.minor хаба
"$UV" pip install --python "$COLO_HOME/env/<hub-id>/bin/python" \
      --torch-backend "${BACKEND:-auto}" -r requirements.txt -c constraints.txt --exact
```

Проверка без скачивания перед реальной установкой: добавить `--dry-run` и разобрать строки ` + torch==...`. Окружение кладётся по фиксированному пути; смена версии хаба = новый каталог `env/<hub-id>` (старый удалить после переключения). Windows-нативный uv/PowerShell в SP5 не нужен (цель - WSL2), но однострочник Windows существует: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.13.1/install.ps1 | iex"`. [И]

---

## 4. curl: `--pinnedpubkey`, `--http2`, gRPC

### 4.1 Поддержка `--pinnedpubkey sha256//`

- libcurl `CURLOPT_PINNEDPUBLICKEY`: добавлено в 7.39.0, "TLS-backend: OpenSSL, GnuTLS, wolfSSL, mbedTLS, Schannel" (https://curl.se/libcurl/c/CURLOPT_PINNEDPUBLICKEY.html; исходник `docs/libcurl/opts/CURLOPT_PINNEDPUBLICKEY.md`, master). [И] В истории страницы: PEM/DER - 7.39.0 (OpenSSL, GnuTLS), 7.43.0 (wolfSSL), 7.47.0 (mbedTLS); хеши sha256: OpenSSL/GnuTLS/wolfSSL с 7.44.0 (по сводке инструмента чтения; в самом md-файле раздел про sha256 я не видел, [Вывод]: для надёжности считать 7.44.0 нижней границей), Schannel 7.58.1.
- Хеш - это SHA-256 от DER-кодировки SubjectPublicKeyInfo, в base64. Получение (проверено, на сервере с самоподписанным EC-сертификатом):

```sh
openssl x509 -in cert.pem -pubkey -noout | openssl pkey -pubin -outform der | openssl dgst -sha256 -binary | openssl base64
# -> h+rL8WV1o773zGe/QCGKpiHuLjSJoUUURPjb9Ee8MNA=      curl ... --pinnedpubkey 'sha256//h+rL8WV1o773zGe/QCGKpiHuLjSJoUUURPjb9Ee8MNA='
```

- Поведение [В, curl 7.81.0]: верный пин + `-k` - успех (rc 0); неверный пин - `curl: (90) SSL: public key does not match pinned public key!` (rc 90), соединение рвётся до передачи данных; без пина и без `-k` - rc 60. Документация: "This option is independent of option CURLOPT_SSL_VERIFYPEER" [И], поэтому `-k` + пин = "доверяем только этому ключу".
- Редиректы на другие origin пином не проверяются [И] (в нашем случае редиректов быть не должно: `-L` не использовать).
- **Secure Transport**: в коде curl 7.79.1 и 8.7.1 `lib/vtls/sectransp.c` есть поддержка пина (`SECTRANSP_PINNEDPUBKEY`), но backend удалён из curl 8.15.0 (`docs/DEPRECATE.md`: "Secure Transport (removed in 8.15.0)"). [И] Современная документация его уже не перечисляет.
- **macOS `/usr/bin/curl`**: Apple 8.7.1 (Release-Date 2024-03-27), в Sonoma 14.7.2 `Features: ... HTTP2 ...`, `libcurl/8.7.1 (SecureTransport) LibreSSL/3.3.6 ... nghttp2/1.61.0`, по словам Daniel Stenberg "supports both LibreSSL and SecureTransport, but ... uses LibreSSL by default" (https://curl.se/mail/lib-2025-01/0048.html, 0062); та же сборка на Sequoia 15.2 (nghttp2/1.63.0) и в macOS 26 по сообщению Apple Community. [И] Пин в OpenSSL-коде curl (`ossl_pkp_pin_peer_pubkey` в `lib/vtls/openssl.c`) не завязан на LibreSSL-условия, а SecureTransport тоже его поддерживает, поэтому ожидается рабочим [Вывод]; **на настоящем Mac я не запускал** - это обязательный пункт приёмки на Mac (одна команда: `curl --http2 -k --pinnedpubkey sha256//... https://hub:port/...`). Прямых отчётов "не работает" найдено не было.
- Windows в WSL2 - это Linux-curl, схема Schannel не нужна.

### 4.2 HTTP/2

- `--http2` (7.33.0): "For HTTPS, this means curl negotiates HTTP/2 in the TLS handshake. curl does this by default." [И, master `docs/cmdline-opts/http2.md`]. Для HTTPS ALPN `h2` предлагается и без флага: curl 7.81 без `--http2` против h2-only сервера дал `HTTP/2 200` (`%{http_version}` = 2). [В] Флаг стоит оставить ради читаемости и чтобы curl без HTTP/2 падал явно.
- curl без HTTP/2 (собран без nghttp2): `--http2` -> сообщение "the installed libcurl version doesn't support this" (проверено на `--http3` для 7.81: `curl: option --http3: the installed libcurl version doesn't support this`); без флага curl тихо пойдёт по HTTP/1.1 и gRPC-сервер ответит обрывом. [В/Вывод]
- `--http1.1` против grpc-сервера: `curl: (56) OpenSSL SSL_read: ... unexpected eof` (ALPN сервера только `h2`). [В]
- Stock curl:
  - Ubuntu 22.04: 7.81.0, OpenSSL/3.0.2, nghttp2/1.43.0, `HTTP2` в Features. [В]
  - Ubuntu 20.04: `curl 7.68.0-1ubuntu2.25` (Launchpad API `getPublishedSources`). [В] HTTP/2 в нём: [Вывод] (Ubuntu собирает с nghttp2, но страницу пакета открыть не удалось: `packages.ubuntu.com` вернул ошибку разбора).
  - Debian 11: `libcurl4 7.74.0-1.3+deb11u16`, `Depends: libnghttp2-14 (>= 1.12.0), libssl1.1`. [И] Debian 12: `7.88.1-10+deb12u15`, `Depends: libnghttp2-14 (>= 1.50.0), libssl3`. [И] (https://packages.debian.org/bullseye/libcurl4, .../bookworm/libcurl4)
  - Старые curl используют OpenSSL 1.1.1 (focal, bullseye): пин поддерживается так же.

### 4.3 gRPC unary через curl: что реально нужно

Стенд: сервер на `grpcio 1.84.0` (generic handler, TLS с самоподписанным EC-сертификатом), клиент curl 7.81.0. [В]

```sh
printf '\x00\x00\x00\x00\x00' | curl -sS --http2 -k --pinnedpubkey "sha256//$PIN" \
   -X POST -H 'content-type: application/grpc' -H 'te: trailers' --data-binary @- \
   -D hdr.txt -o out.bin https://127.0.0.1:$PORT/colosseum.Hub/Installer
```

Результаты:

- Тело запроса - gRPC-кадр: 1 байт флага сжатия + 4 байта длины + сообщение; пустое сообщение = 5 нулевых байт. Сообщения в protobuf-обёртке не нужны, если у метода пустой/произвольный запрос.
- **`te: trailers` обязателен для grpc-core**: без него `curl: (92) HTTP/2 stream 0 was not closed cleanly: INTERNAL_ERROR (err 2)`. (Спецификация gRPC требует `TE: trailers` в заголовках запроса: https://github.com/grpc/grpc/blob/master/doc/PROTOCOL-HTTP2.md; другие реализации сервера могут не требовать.)
- Ответ: `HTTP/2 200`, `content-type: application/grpc`; тело = 5 байт префикса (`00 00 00 00 1e`: флаг, длина) + сообщение; поэтому скрипт получают как `curl ... | tail -c +6 | sh` (проверено: `hello-from-hub`).
- Трейлеры (`grpc-status: 0`, `grpc-message:`) curl 7.81 пишет в файл `-D` вместе с заголовками. Для старых curl (7.68, 7.74) это не проверено [Вывод: не опираться на трейлеры].
- **Ошибки идут как HTTP 200**: вызов с `abort(NOT_FOUND)` -> `HTTP/2 200 ... grpc-status: 5 grpc-message: nope` (при ошибке "trailers-only" статус в заголовках), неизвестный метод - `grpc-status: 12`. `curl -f` код ошибки не вернёт (rc 0), а `curl ... | tail -c +6 | sh` при пустом теле молча выполнит ничего и вернёт 0. Нужен явный контроль: скачать в файл, проверить непустоту и первую строку (`#!/bin/sh`), хеш, затем выполнить.
- Механизм ретраев/таймаутов: `--connect-timeout 10 --max-time 120 --retry 3 --retry-connrefused`.

### 4.4 Один порт + mutual TLS + curl: несовместимость в grpcio (важно)

Проверка на grpcio 1.84.0 [В]:

```python
creds = grpc.ssl_server_credentials([(key, cert)], root_certificates=ca_pem, require_client_auth=False)
# openssl s_client -alpn h2 -cert node.pem -key node.key -> "No client certificate CA names sent"
# auth_context() обработчика: только transport_security_type/security_level/..., ни x509_common_name, ни x509_pem_cert,
# и для клиента с сертификатом нашего CA, и без сертификата, и с чужим.
```

Сигнатура низкого уровня: `cygrpc.server_credentials_ssl(pem_root_certs, pem_key_cert_pairs, force_client_auth)`; никакого "запросить, но не требовать". Следствия для спецификации SP5 (один внешний порт, mutual TLS, установка по `curl`):

- с `require_client_auth=True` `curl` без сертификата не пройдёт даже `-k`;
- с `False` сервер не запросит сертификат и у зарегистрированных узлов, то есть идентичность узла нужно проверять в самом приложении (подписанный токен/подпись запроса/одноразовый join-token в metadata), а TLS даёт только шифрование и пин;
- варианты, если mTLS обязателен: (a) второй порт только для bootstrap (противоречит решению "один порт"), (b) TLS-терминатор/мультиплексор перед grpcio (по SNI/ALPN или по наличию сертификата), (c) собственный h2-сервер на `ssl` + `h2` вместо grpc-core. Решение за владельцем; здесь только факт.

---

## 5. WSL2: сеть

### 5.1 Определить режим

- `wslinfo --networking-mode` - бинарь `/usr/bin/wslinfo`, появился в WSL **2.0.4** (релиз 2023-10-04: "Implement /usr/bin/wslinfo binary for querying various state, added `--networking-mode` argument", GitHub Releases `microsoft/WSL` тег 2.0.4). [И + В] На этой машине (WSL 3.0.3.0): `wslinfo --help` показывает `--networking-mode --msal-proxy-path --vm-id --version -n`.
- Допустимые значения `networkingMode` в `.wslconfig` (https://learn.microsoft.com/windows/wsl/wsl-config, raw `MicrosoftDocs/WSL/WSL/wsl-config.md`): `none`, `nat` (по умолчанию), `bridged` (deprecated с WSL 2.4.5), `mirrored`, `consomme` (раньше `virtioproxy`); "starting from WSL 2.3.25, if NAT network mode fails, it falls back to using Consomme network mode". [И]
- Наблюдение [В]: `wslinfo --networking-mode` -> `nat`, но `ip route` -> `default via 192.168.127.1 dev wsltap`, а адреса: `eth0 172.31.101.51/20` и `wsltap 192.168.127.2/24`. То есть реальный маршрут не классический NAT. Вывод [Вывод]: для проверки "достижим ли сервис из LAN" не полагаться на строку режима; различать по сочетанию (`wslinfo` или чтение `.wslconfig`, имя устройства маршрута по умолчанию, `ip -br addr`), а достижимость проверять фактическим подключением снаружи (кнопка "проверить" в мастере присоединения).
- Фолбэк, если `wslinfo` нет (WSL < 2.0.4): прочитать `%UserProfile%\.wslconfig` из WSL: `P=$(wslpath "$(cmd.exe /c 'echo %USERPROFILE%' | tr -d '\r')"); cat "$P/.wslconfig"` (проверено: файл читается, содержимое `[wsl2] memory=26GB swap=32GB`, `networkingMode` не задан). Интероп `cmd.exe`/`powershell.exe` из WSL работает [В] (если не выключен `[interop] enabled=false`).
- Версия ОС Windows: `powershell.exe -NoProfile -Command "(Get-ComputerInfo -Property OsName,OsVersion | Format-List | Out-String)"` -> у владельца Windows 10 Pro 10.0.19045. **Следствие**: зеркальный режим и Hyper-V firewall требуют Windows 11 22H2+ (https://learn.microsoft.com/windows/wsl/networking: "On machines running Windows 11 22H2 and higher you can set networkingMode=mirrored"); на этом хосте `Get-NetFirewallHyperVVMSetting` не существует (`CommandNotFoundException`). [В + И] Для Windows 10 остаётся только NAT/Consomme + portproxy.

### 5.2 Входящие подключения из LAN

**NAT (по умолчанию, и Windows 10)** [И, https://learn.microsoft.com/windows/wsl/networking]: порты WSL доступны Windows по `localhost` (`localhostForwarding=true`), но не из LAN ("This isn't the default case in WSL 2"). Нужен portproxy на Windows (PowerShell от администратора):

```powershell
$ip = (wsl.exe hostname -I).Trim().Split(' ')[0]          # адрес WSL VM (строчная -i даёт не то)
netsh interface portproxy add v4tov4 listenport=8443 listenaddress=0.0.0.0 connectport=8443 connectaddress=$ip
New-NetFirewallRule -DisplayName "Colosseum hub 8443" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8443
# просмотр / удаление:
netsh interface portproxy show v4tov4
netsh interface portproxy delete v4tov4 listenport=8443 listenaddress=0.0.0.0
Remove-NetFirewallRule -DisplayName "Colosseum hub 8443"
```

Проверено на хосте владельца [В]: `New-NetFirewallRule`, `Get-NetFirewallRule`, `Remove-NetFirewallRule` есть; служба `iphlpsvc` (IP Helper, нужна portproxy) `Running/Automatic`; `netsh interface portproxy show v4tov4` без прав выполняется и пуст; `... add v4tov4 ...` без прав отвечает "Для запрошенной операции требуется повышение прав" - то есть из WSL без UAC правило не поставить; можно предложить `powershell.exe -Command "Start-Process -Verb RunAs ..."` (UAC-диалог) [Вывод]. Адрес WSL меняется при перезапуске VM, правило нужно обновлять (хук на старт) [Вывод]. Приложение внутри WSL должно слушать `0.0.0.0` (документация: "you may need to bind your application to 0.0.0.0"). [И]

**Mirrored (Windows 11 22H2+)** [И]: в `%UserProfile%\.wslconfig`

```ini
[wsl2]
networkingMode=mirrored
```

затем `wsl --shutdown`. Из LAN можно подключаться к WSL напрямую, но Hyper-V firewall по умолчанию блокирует входящие (WSL 2.0.9+, Windows 11 22H2+: "the Hyper-V firewall feature will be turned on by default"). От администратора:

```powershell
Get-NetFirewallHyperVVMCreator                       # {40E0AC32-46A5-438A-A0B2-2B479E8F2E90}  WSL
Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow
# либо точечно:
New-NetFirewallHyperVRule -Name "ColosseumHub" -DisplayName "Colosseum hub" -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -Protocol TCP -LocalPorts 8443
Get-NetFirewallHyperVRule -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}'
```

Источники: https://learn.microsoft.com/windows/security/operating-system-security/network-security/windows-firewall/hyper-v-firewall (тот же GUID: "The WSL VMCreatorId is `{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}`") и https://learn.microsoft.com/windows/wsl/networking. [И] Настройка `.wslconfig` `firewall=true` (по умолчанию) определяет, фильтруют ли WSL-трафик правила Windows/Hyper-V. [И] В mirrored `localhostForwarding` игнорируется, `localhost` Windows и WSL общий. [И]

### 5.3 Tailscale внутри WSL2

Рабочая альтернатива (когда не хочется ни portproxy, ни mirrored): узел получает адрес `100.x` и достижим откуда угодно. Из официальной страницы "Install Tailscale on Windows with WSL 2" (по пересказу поиска; прямой fetch `tailscale.com` из этой среды не удался: таймаут/закрытый сокет): ставить в дистрибутив скриптом `tailscale.com/install.sh`, затем `sudo tailscale up`; **не запускать Tailscale одновременно на Windows и в WSL2** (трафик WSL2 через клиент Windows ломается; рекомендуется только клиент на Windows хосте, если нужен весь хост); MTU интерфейса WSL2 по умолчанию 1280, tailscaled поднимает его до 1340. [И - косвенно] Требуется systemd (`/etc/wsl.conf`: `[boot] systemd=true`; у владельца уже включён - [В]). Для gRPC на одном TLS-порту Tailscale удобен тем, что не нужны ни портпрокси, ни правила Windows firewall (правила Windows для WSL-интерфейса не применяются к tailnet-трафику) [Вывод].

---

## 6. Ядра, память, GPU без psutil и без torch

### 6.1 Что вернули источники на машине владельца [В]

- `os.cpu_count()` = `len(os.sched_getaffinity(0))` = 20; `os.process_cpu_count()` (Python 3.13) существует.
- `/sys/devices/system/cpu/cpu0/topology/`: `core_id=0, physical_package_id=0, thread_siblings_list=0-1, core_cpus_list=0-1`; `lscpu`: "Thread(s) per core: 2, Core(s) per socket: 10, Socket(s): 1" при том, что i7-12700KF физически имеет 8 P-ядер (2 потока) + 4 E-ядра (12 ядер, 20 потоков). **В WSL2 топология виртуальная**, число "физических" ядер приблизительное (10). На нативном Linux/macOS числа точные.
- cgroup v2 (`stat -fc %T /sys/fs/cgroup` = `cgroup2fs`), `/sys/fs/cgroup/cpu.max` = `max 100000`, `memory.max` = `max`; `/proc/self/cgroup` = `0::/non-systemd`.

### 6.2 Правила

- **Affinity** (Linux): `os.sched_getaffinity(0)` (учитывает `taskset`, cpuset контейнера). `os.process_cpu_count()` (3.13) = то же самое плюс переопределение `PYTHON_CPU_COUNT`/`-X cpu_count`; **cgroup-квоту он не учитывает** (документация: "can be less than cpu_count() depending on the CPU affinity"). [И, Doc/library/os.rst@3.13]
- **Квота контейнера**: cgroup v2 `cpu.max` (`"$MAX $PERIOD"`, `max` = без лимита; файл "exists on non-root cgroups"; https://docs.kernel.org/admin-guide/cgroup-v2.html) [И]; cgroup v1: `/sys/fs/cgroup/cpu/cpu.cfs_quota_us` и `cpu.cfs_period_us` (`-1` = без лимита) [Вывод - из документации ядра v1, не перечитывалось]. Docker `--cpus` и K8s `resources.limits.cpu` превращаются именно в эту квоту [Вывод]. Лимит надо брать как минимум по цепочке cgroup от текущей к корню.
- **Физические ядра**: число уникальных `topology/thread_siblings_list` среди CPU из affinity; если файла нет, использовать логические.
- **macOS**: `sysctl -n hw.physicalcpu`, `hw.logicalcpu` (на Apple Silicon равны), `hw.perflevel0.*` / `hw.perflevel1.*` - P/E-кластеры (`.name` говорит, какой это кластер: порядок не фиксирован: на одном чипе level0 = Performance, на другом "Super"), `hw.ncpu` суммарно. [И, https://developer.apple.com/forums/thread/664774, https://eclecticlight.co/sysctl-information/]. Память: `sysctl -n hw.memsize` (байты) [Вывод: источник не нашёл, команда стандартная].
- **Память Linux**: `/proc/meminfo` `MemTotal`, `MemAvailable` (кБ); контейнер: `memory.max` (v2, `max` = нет лимита) / `memory/memory.limit_in_bytes` (v1, огромное число = нет лимита), брать минимум. [И - формат `memory.max` по доке ядра]
- **GPU без torch**: `nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap,uuid --format=csv,noheader,nounits`, пример вывода [В]: `0, NVIDIA GeForce RTX 4070 Ti, 12282, 610.88, 8.9, GPU-4e69...`. AMD: `rocm_agent_enumerator`; Intel: PCI (class display, vendor 0x8086) (как делает uv). macOS: `uname -m` (`arm64` = Apple Silicon, MPS), `sysctl -n machdep.cpu.brand_string`.

### 6.3 Функции (stdlib, Python >= 3.9, проверено на 3.12.15 и 3.13.7; юнит-проверка разбора вложенного `cpu.max` на фиктивном дереве дала 1.5 / 4.0 / None)

```python
"""Stdlib-only hardware probe (Linux, macOS)."""
from __future__ import annotations
import math, os, subprocess, sys
from pathlib import Path

def _read(p):
    try:
        return Path(p).read_text().strip()
    except OSError:
        return None

def _sysctl(name):
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip()) if out.returncode == 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None

def _cg2_walk(fname, root="/sys/fs/cgroup", rel=None):
    """Values of `fname` from the process's cgroup v2 up to the mount root."""
    rootp = Path(root)
    if not (rootp / "cgroup.controllers").exists():
        return []
    if rel is None:
        rel = ""
        for line in (_read("/proc/self/cgroup") or "").splitlines():
            if line.startswith("0::"):
                rel = line[3:]
    d, out = rootp / rel.lstrip("/"), []
    while True:
        v = _read(d / fname)
        if v is not None:
            out.append(v)
        if d == rootp or rootp not in d.parents:
            break
        d = d.parent
    return out

def _cgroup_cpu_limit(root="/sys/fs/cgroup", rel=None):
    limits = []
    for v in _cg2_walk("cpu.max", root, rel):
        q, _, per = v.partition(" ")
        if q != "max":
            limits.append(int(q) / int(per or 100000))
    if not limits and not (Path(root) / "cgroup.controllers").exists():            # cgroup v1
        q = _read(f"{root}/cpu/cpu.cfs_quota_us") or _read(f"{root}/cpu,cpuacct/cpu.cfs_quota_us")
        per = _read(f"{root}/cpu/cpu.cfs_period_us") or _read(f"{root}/cpu,cpuacct/cpu.cfs_period_us")
        if q and per and int(q) > 0:
            limits.append(int(q) / int(per))
    return min(limits) if limits else None

def cpu_counts():
    if sys.platform == "darwin":
        lg = _sysctl("hw.logicalcpu") or os.cpu_count() or 1
        return {"logical": lg, "physical": _sysctl("hw.physicalcpu") or lg, "quota": None}
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    logical = len(cpus) or os.cpu_count() or 1
    sib = set()
    for c in cpus:
        s = _read(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")
        if s is None:
            sib = set(); break
        sib.add(s)
    physical = len(sib) or logical
    quota = _cgroup_cpu_limit()
    if quota is not None:
        q = max(1, math.floor(quota))
        logical, physical = min(logical, q), min(physical, q)
    return {"logical": logical, "physical": physical, "quota": quota}

def mem_bytes():
    if sys.platform == "darwin":
        return {"total": _sysctl("hw.memsize"), "limit": None}
    total = avail = None
    for line in (_read("/proc/meminfo") or "").splitlines():
        if line.startswith("MemTotal:"): total = int(line.split()[1]) * 1024
        elif line.startswith("MemAvailable:"): avail = int(line.split()[1]) * 1024
    vals = [int(v) for v in _cg2_walk("memory.max") if v != "max"]
    v1 = _read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if v1 and int(v1) < (1 << 60):
        vals.append(int(v1))
    return {"total": total, "available": avail, "limit": min(vals) if vals else None}

def nvidia_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,compute_cap,uuid",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    rows = []
    for line in out.stdout.strip().splitlines():
        i, name, mem, drv, cc, uuid = [x.strip() for x in line.split(",")]
        rows.append({"index": int(i), "name": name, "mem_mib": int(mem), "driver": drv, "cc": cc, "uuid": uuid})
    return rows
```

Результат на этой машине: `{'logical': 20, 'physical': 10, 'quota': None}`, память `total 27313180672`, GPU с `cc 8.9`. Полная версия файла: scratchpad `research/code/hw.py`. Рекомендованное число рабочих процессов - от `physical` (минус резерв под learner/агента), а не от `os.cpu_count()`; на WSL2 `physical` приблизительно, это нормально.

### 6.4 Что отдавать uv

`export UV_CUDA_DRIVER_VERSION=<driver_version из nvidia_gpus()>` - тогда выбор индекса не зависит от `PATH` и совпадает с тем, что показал зонд (в логах узла и в мастере присоединения видно одно и то же).

---

## 7. Переносимость на macOS

Mac под рукой нет: пункты помечены, что именно проверено на Linux.

### 7.1 Смерть ребёнка вместе с родителем без `prctl(PR_SET_PDEATHSIG)` (в macOS нет)

- **Lifeline-pipe (рекомендуется, переносимо)** [В, Linux; механизм POSIX]: родитель создаёт `os.pipe()`, передаёт ребёнку read-конец (`pass_fds`), сам закрывает у себя read-конец и держит write-конец; ребёнок в потоке делает блокирующий `os.read(fd, 1)`; EOF (все write-концы закрыты, в том числе при `kill -9` родителя) -> `os._exit(143)`. Проверено: ребёнок завершился сразу после `os._exit(0)` родителя.

```python
def watch_parent_lifeline(fd):          # в ребёнке
    def run():
        try:
            while os.read(fd, 1): pass
        except OSError: pass
        os._exit(143)
    threading.Thread(target=run, daemon=True).start()
```

- **`multiprocessing.parent_process().sentinel`** (3.8+) [В, Linux]: для процессов, порождённых `multiprocessing`; `multiprocessing.connection.wait([sentinel])` в потоке -> `os._exit`. Проверено со `spawn`-контекстом: ребёнок умер после выхода родителя. Это тот же механизм lifeline-pipe внутри мультипроцессинга и он переносим на macOS [Вывод].
- **kqueue** [И по документации, не запускалось]: `select.kqueue()`, `select.kevent(ppid, filter=select.KQ_FILTER_PROC, flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT, fflags=select.KQ_NOTE_EXIT)`; `KQ_FILTER_PROC`, `KQ_NOTE_EXIT` ("the process has exited") перечислены в https://docs.python.org/3/library/select.html (`Doc/library/select.rst@3.12`). Подходит и когда pipe использовать нельзя, но гонка: если родитель умер до регистрации события, kevent вернёт `ESRCH` - проверять `os.getppid()` сразу после регистрации.
- **Поллинг `os.getppid()`** [Вывод]: при смерти родителя ppid меняется на 1 (launchd) или на ближайшего subreaper; поллинг раз в 1-2 с проще всего, но с задержкой и ложными срабатываниями в контейнерах, где родитель - PID 1.
- На Linux дополнительно `prctl(PR_SET_PDEATHSIG)` через `ctypes` ничего не стоит, но lifeline-pipe покрывает обе платформы одним кодом.

### 7.2 Unix-сокеты

- Лимит `sun_path`: Linux 108 байт, macOS 104 (man `unix(4)`; сводка поиска: GnuPG-тест на macOS Sierra `sizeof addr.sun_path: 104`, Racket docs: 108 для Linux и 104 для macOS) [И - вторичные источники]. Проверка на Linux [В]: путь длиной 100 и 107 байт - OK, 108 и 109 - `OSError: AF_UNIX path too long` (Python сравнивает с `sizeof(sun_path)`; для не-абстрактных путей нужен ещё завершающий NUL). Итого безопасный бюджет для обеих ОС: **путь <= 100 символов**. Важно: macOS `$TMPDIR` вида `/var/folders/xx/yyyyyyyyyyyyyyyyyyyyyyyyyyyyyy/T/` уже ~49 символов; короткий путь строить как `/tmp/colo-<uid>/<8hex>.sock` или `chdir` + относительный путь.
- Абстрактные сокеты (адрес с ведущим `\0`) есть **только на Linux**. `multiprocessing.connection.Listener("\0name", family="AF_UNIX", authkey=b"k")` работает, `Client(...)` подключается, round-trip прошёл, закрытие без ошибки (Python 3.12.15 и 3.13.7) [В]; `Listener.address` возвращает `bytes`. Unlink для абстрактного адреса не вызывается (в CPython есть `util.is_abstract_socket_namespace`) [Вывод]. На macOS - только файловые пути.

### 7.3 Разделяемая память на macOS

- `/dev/shm` на macOS нет [Вывод, общеизвестно]. Замена - файл в `$TMPDIR` (или `~/.colosseum/run/`) + `mmap.mmap(fd, size)`; он лежит на APFS (не tmpfs): страницы остаются в кэше ОС, но возможен сброс на диск, а запись на SSD при большом потоке весов - лишний износ. Для весов (несколько сотен МБ на агента, обновление раз в секунды) это приемлемо. [Вывод]
- `shm_open` из Python: `multiprocessing.shared_memory.SharedMemory` реализован через `_posixshmem.shm_open` на любой POSIX-платформе (`_USE_POSIX = os.name == "posix"`; в коде `Lib/multiprocessing/shared_memory.py@3.12` для BSD есть `_SHM_SAFE_NAME_LENGTH = 14`: "FreeBSD (and perhaps other BSDs) limit names to 14 characters") [И]; имя вида `/psm_xxxxxxxx`. У macOS лимит имени POSIX shm около 31 байта (сводки: 30/31/32 в разных источниках, MIDAS: "31 bytes (+NUL), см. PSHMNAMLEN в posix_shm.h") [И - вторичные источники]: держать имя <= 30 символов. Также известная особенность macOS: повторный `ftruncate` на shm-объекте падает [Вывод - не проверялось]: размер задавать один раз при создании.
- На Linux: `/dev/shm` (tmpfs, по умолчанию 50% RAM; в Docker 64 MB, §8) + `mmap` по пути `/dev/shm/colosseum-<hub>-<agent>`; файл-подход без `multiprocessing.shared_memory` вообще обходит resource tracker (§7.4). На Windows-нативной ОС (вне SP5) нужен именованный mapping.

### 7.4 Resource tracker (`multiprocessing.shared_memory`)

- В Python <= 3.12 каждый `SharedMemory(name=..., create=False)` регистрируется в resource tracker и при выходе процесса сегмент **удаляется**, даже если его создал другой процесс. Документация 3.13: "Python processes created in any other way [not via multiprocessing] will receive their own resource tracker when accessing shared memory with *track* enabled. This will cause the shared memory to be deleted by the resource tracker of the first process that terminates. To avoid this issue, users of subprocess or standalone Python processes should set *track* to False" (`Doc/library/multiprocessing.shared_memory.rst@3.13`, `versionchanged:: 3.13 Added the track parameter`). [И] Коммит-источник `gh-82300: Add track parameter to multiprocessing.shared_memory (#110778)`; предыстория - bpo-38119 (https://bugs.python.org/issue38119, обсуждение автоудаления tracker'ом). [И по результатам поиска; сам текст issue не открывал]
- Воспроизведено [В] (создатель - один Python-процесс, потребитель - отдельный `python -I`, запущенный через `subprocess`):

| Python | потребитель | сегмент после выхода потребителя |
|---|---|---|
| 3.12.15 | `SharedMemory(name=n)` | **удалён** (+ предупреждения tracker'а "leaked shared_memory objects") |
| 3.13.7 | `SharedMemory(name=n)` | **удалён** |
| 3.13.7 | `SharedMemory(name=n, track=False)` | **цел** |
| 3.12.15 | `SharedMemory(name=n)` + `resource_tracker.unregister(s._name, "shared_memory")` | **цел** (обход для 3.8-3.12) |

- Для SP5 [Вывод]: узел на Python 3.12 (версия хаба) -> либо `resource_tracker.unregister(...)` сразу после открытия сегмента, либо (чище и без вариативности по версиям) собственный `mmap` файла в `/dev/shm` / `$TMPDIR`: ни tracker'а, ни имён shm, одинаково на Linux и macOS. Процессы, порождённые через `multiprocessing` от общего предка, делят один tracker и проблемы не имеют.

---

## 8. Docker / K8s

### 8.1 Docker

- `--shm-size`: "Size of `/dev/shm`. The format is `<number><unit>`. `number` must be greater than 0. Unit is optional and can be b, k, m, or g. If you omit the unit, the system uses bytes. If you omit the size entirely, the system uses 64m." (docker/cli `docs/reference/commandline/run.md`). [И] Для весов/буферов трейджекторий в контейнере по умолчанию мало: `--shm-size=2g` или `--ipc=host`. Опция `--ipc`: `private`, `shareable`, `container:<id>`, `host`. [И]
- `--gpus all` / `--gpus device=0` / `--gpus '"device=0,2"'`: требуется NVIDIA Container Toolkit на хосте ("Install NVIDIA Container Toolkit ... Include the `--gpus` flag when you start a container to access GPU resources"; `docker run -it --rm --gpus all ubuntu nvidia-smi`) [И, docker/docs `engine/containers/gpu.md`]. Docker Desktop на Windows: `--gpus` только с WSL 2 backend, нужны свежие драйверы NVIDIA с поддержкой WSL2 GPU-PV и `wsl --update` [И, docs `desktop/features/gpu.md`].
- **Docker Desktop и WSL integration**: включается в Docker Desktop: **Settings > Resources > WSL Integration** (если пункта нет - Docker в режиме Windows-контейнеров, переключить на Linux containers); "enabling WSL integration for a distribution gives you direct access to docker commands from that distribution's terminal"; минимум WSL 2.1.5 [И, docs `desktop/features/wsl/_index.md`].
- Как понять изнутри WSL, работает ли `docker` [В, на машине владельца интеграция выключена]:

```sh
type -a docker           # docker is /mnt/c/Program Files/Docker/Docker/resources/bin/docker   (скрипт-заглушка Docker Desktop), /usr/bin/docker -> /mnt/wsl/docker-desktop/cli-tools/usr/bin/docker (битая ссылка)
docker info >/dev/null 2>&1; echo $?     # 1
docker info 2>&1 | head -3
# The command 'docker' could not be found in this WSL 2 distro.
# We recommend to activate the WSL integration in Docker Desktop settings.
```

Критерий для мастера: `command -v docker` недостаточно (заглушка есть), нужен `docker info` с кодом 0 (`docker version --format '{{.Server.Version}}'`); при провале показать подсказку "включите WSL Integration для дистрибутива". `nvidia-smi` внутри контейнера: `docker run --rm --gpus all <image> nvidia-smi`.

### 8.2 kubeconform

- Релиз v0.8.0 (2026-06-04), один статический бинарь (14 MB) в архиве: `kubeconform-{linux,darwin}-{amd64,arm64}.tar.gz` (+`armv6`), `kubeconform-windows-{amd64,arm64}.zip`, файл `CHECKSUMS`; также Homebrew, winget, `go install github.com/yannh/kubeconform/cmd/kubeconform@latest`. [И + В: `GET .../releases/download/v0.8.0/kubeconform-linux-amd64.tar.gz`, `./kubeconform -v` -> `v0.8.0`]
- Флаги: `-strict`, `-summary`, `-kubernetes-version`, `-schema-location`, `-ignore-missing-schemas`, `-skip`, `-n <workers>`, `-output json`, `-exit-on-error`. По умолчанию схемы качаются с `https://raw.githubusercontent.com/yannh/kubernetes-json-schema/master/{{.NormalizedKubernetesVersion}}-standalone{{.StrictSuffix}}/{{.ResourceKind}}{{.KindSuffix}}.json`; `-schema-location` принимает шаблон с локальным путём. [И, Readme]
- Проверено на манифестах репозитория `deployment/k8s/` (3 Deployment, HPA, Namespace, PVC, 2 Service) [В]:

```sh
kubeconform -strict -summary -kubernetes-version 1.30.0 deployment/k8s
# Summary: 8 resources found in 4 files - Valid: 8, Invalid: 0, Errors: 0, Skipped: 0
# офлайн: схемы положены в schemas/v1.30.0-standalone-strict/{deployment-apps-v1,horizontalpodautoscaler-autoscaling-v2,namespace-v1,persistentvolumeclaim-v1,service-v1,_definitions}.json
unshare -rn kubeconform -strict -summary -kubernetes-version 1.30.0 \
  -schema-location 'schemas/{{.NormalizedKubernetesVersion}}-standalone{{.StrictSuffix}}/{{.ResourceKind}}{{.KindSuffix}}.json' deployment/k8s
# запуск в сетевом namespace без сети (unshare -rn): тот же результат Valid: 8
# опечатка 'sizeLimitt' в emptyDir: ... additional properties 'sizeLimitt' not allowed -> Invalid: 1, rc=1
```

Для HPA нужны оба файла (`autoscaling-v1` и `v2`) смотря на `apiVersion`; список файлов получить из `kind` и `apiVersion` манифестов. Для CI офлайн: взять каталог схем нужной версии из репозитория `yannh/kubernetes-json-schema` (ветка master, каталог `v<ver>-standalone-strict`) и закоммитить/кэшировать нужные файлы. CRD (если появятся) - `openapi2jsonschema` из того же репозитория kubeconform. kubeconform проверяет схему, а не семантику (не поймает неверный `image`, нехватку RBAC, неподходящий `medium`).

### 8.3 `/dev/shm` в подах

`emptyDir.medium: Memory` монтирует tmpfs: "If you set the `emptyDir.medium` field to "Memory", Kubernetes mounts a tmpfs (RAM-backed filesystem) for you instead. ... files you write count against the memory limit of the container that wrote them. ... If no size is specified, memory-backed volumes are sized to node allocatable memory." (kubernetes/website `concepts/storage/volumes.md`). [И] Рецепт (`/dev/shm` в поде по умолчанию 64 MB, как в Docker [Вывод]):

```yaml
spec:
  containers:
  - name: node
    volumeMounts: [{name: dshm, mountPath: /dev/shm}]
    resources: {limits: {memory: 8Gi}}   # tmpfs входит в этот лимит
  volumes:
  - name: dshm
    emptyDir: {medium: Memory, sizeLimit: 2Gi}
```

`sizeLimit` обязателен для предсказуемости (иначе размер = allocatable узла, а использование всё равно считается в memory limit контейнера). Для GPU-подов в K8s - расширение `nvidia.com/gpu` (NVIDIA device plugin) вместо `--gpus` [Вывод, вне исследованных документов].

---

## Выводы для SP5

Что принять в план как решения (для владельца и для pre-flight).

**uv и PyTorch**

1. Опираться на `uv pip install --torch-backend auto` (стабильный с uv 0.7.14, в 0.13.1 без `--preview`); закреплять версию uv (`https://astral.sh/uv/0.13.1/install.sh`) и версию Python (`--python 3.12`, major.minor хаба). Использовать только интерфейс `uv pip` (в `uv sync`/`add` флага нет).
2. Окружение: `uv venv` по фиксированному пути + `uv pip install --python ENV/bin/python -r requirements.txt -c constraints.txt --exact --torch-backend $BACKEND`. `uv pip sync` не использовать. В `requirements.txt` пин `torch==<публичная версия хаба>`; в `constraints.txt` - `pip freeze` хаба без `torch`, `triton`, `nvidia-*`, `cuda-*`, без локальных тегов `+...` и без самого пакета (§3.2).
3. Перед запуском `UV_CUDA_DRIVER_VERSION` из собственного зонда (`nvidia-smi --query-gpu=...`) и `PATH` с `/usr/lib/wsl/lib`: иначе в WSL2 из ssh/systemd выбор молча уходит на CPU (§1.2).
4. Правило старых GPU: `min(compute_cap) < 7.5` -> `--torch-backend cu126` для torch <= 2.14 (иначе CPU + предупреждение): `auto` этого не знает (uv #14742; PyTorch notice 2026-09-04: после 2.14 нет колёс для Maxwell/Pascal/Volta).
5. После установки обязательная самопроверка в дочернем интерпретаторе: `torch.__version__`, `torch.version.cuda`, `torch.cuda.is_available()`, `torch.backends.mps.is_available()` на macOS; расхождение "GPU есть, колесо `+cpu`" - ошибка уровня WARNING в мастере, не молчание. Причины молчаливого CPU: драйвер ниже порога для нужных cu12x/cu13x, у торча хаба нет сборки под драйвер (пример 2.14.1: только cu126/cu130/cu132), PATH без nvidia-smi.
6. macOS: торч хаба (>= 2.13) ставится только на Apple Silicon (arm64) и macOS >= 14; Intel Mac и старые macOS отклонять заранее понятным сообщением (§1.2).
7. Окружение процесса: чистить `PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE VIRTUAL_ENV`, ставить `PYTHONNOUSERSITE=1` (на машине владельца ROS `PYTHONPATH` попадает в новый venv). `UV_PYTHON_INSTALL_DIR` и `UV_CACHE_DIR` держать внутри `COLOSSEUM_HOME` на той же ФС, что и окружение; в WSL не использовать `/mnt/c`.
8. Установка uv: `UV_UNMANAGED_INSTALL=$COLO_HOME/uv-bin` (не трогает профили и receipt); `UV_INSTALL_DIR` перезаписывает `~/.config/uv/uv-receipt.json` пользователя. Проверку SHA-256 делать на стороне loader (`shasum -a 256` на macOS), потому что `install.sh` на машине без `sha256sum` тихо её пропускает. Режим без интернета на узле: хаб раздаёт `uv-<triple>.tar.gz` и управляемый Python (`UV_PYTHON_INSTALL_MIRROR=file://...`, `UV_OFFLINE=1`); этот вариант можно отнести к SP5b.

**curl-установщик**

9. Команда: `curl --http2 -k --pinnedpubkey 'sha256//<b64>' -X POST -H 'content-type: application/grpc' -H 'te: trailers' --data-binary @<5 нулей> https://hub:port/colosseum.Hub/Installer -o file`; тело ответа начинается с 5-байтового gRPC-префикса (`tail -c +6`). `te: trailers` обязателен для grpc-core; `-L` не использовать (пин не распространяется на редиректы).
10. Ошибки gRPC приходят как HTTP 200: `-f` и `curl | sh` их не ловят. Установщик скачивать в файл, проверять непустоту, первую строку и хеш, затем выполнять; статус смотреть в `-D` (но на трейлеры старых curl не опираться).
11. Совместимость: Ubuntu 22.04, Debian 11/12 - HTTP/2 и пин проверены/подтверждены; Ubuntu 20.04 (curl 7.68) - ожидается (приёмка на ней нужна); macOS `/usr/bin/curl` 8.7.1 - HTTP2 подтверждён, пин в LibreSSL/SecureTransport ожидаем рабочим, но не проверен: обязательный шаг приёмки на Mac. Если пин на какой-то сборке не сработает, запасной вариант - `--cacert` с самоподписанным сертификатом хаба вместо `-k` + пин.
12. **Противоречие требований**: "один внешний порт + mutual TLS" и "установка по `curl` без клиентского сертификата" не совмещаются в grpcio (§4.4): `require_client_auth=False` сертификат не запрашивает вообще. В спецификации нужно выбрать: проверка личности узла на прикладном уровне (join-token/подпись) поверх серверного TLS с пином, либо второй порт для bootstrap, либо TLS-мультиплексор перед grpc.

**WSL2**

13. Рецепт: определить режим (`wslinfo --networking-mode`, при её отсутствии чтение `.wslconfig` через `cmd.exe /c echo %USERPROFILE%` и `wslpath`); проверить версию Windows (на хосте владельца Windows 10 build 19045: mirrored и Hyper-V firewall недоступны). Для NAT/Consomme - портпрокси + правило файрвола с правами администратора (команды в §5.2; из WSL без UAC их не выполнить: проверено), приложение слушает `0.0.0.0`. Для mirrored (Win 11 22H2+) - `Set-NetFirewallHyperVVMSetting` либо `New-NetFirewallHyperVRule` с VMCreatorId `{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}`. Не доверять строке режима: проверять достижимость реальным подключением снаружи.
14. Tailscale внутри WSL2 - рабочая альтернатива без портпрокси и правил (не ставить одновременно на Windows и в WSL2); оформить как документированный вариант, не как автоматический.

**Ядра, память, GPU**

15. Функции из §6.3 (stdlib; affinity + `thread_siblings_list` + потолок cgroup v2/v1; macOS через `sysctl`; GPU через `nvidia-smi`). Число рабочих процессов считать от `physical`. В WSL2 `physical` приблизительное. `os.process_cpu_count()` не использовать как единственный источник: cgroup он не видит.

**macOS-прокладки**

16. Смерть ребёнка с родителем: lifeline-pipe (или `multiprocessing.parent_process().sentinel`) вместо `PR_SET_PDEATHSIG`; одинаково на Linux и macOS; kqueue `EVFILT_PROC`/`NOTE_EXIT` - запасной вариант.
17. Unix-сокеты: путь <= 100 символов, строить короткий путь в `/tmp/colo-<uid>/`; абстрактные сокеты только как оптимизация Linux (работают через `multiprocessing.connection`, проверено), на macOS - файловые.
18. Разделяемая память: собственный `mmap` файла (`/dev/shm` на Linux, `$TMPDIR`/каталог запуска на macOS), без `multiprocessing.shared_memory`: так не нужны ни `track=False` (только 3.13+), ни ручной `unregister` (3.12), ни ограничение длины имени shm (~30 байт на macOS). Если всё же нужен `SharedMemory` на 3.12, сразу после открытия вызывать `resource_tracker.unregister(shm._name, "shared_memory")` (проверено).

**Docker/K8s (в первую очередь SP5b)**

19. В контейнерах `--shm-size` (по умолчанию 64m) и `emptyDir: {medium: Memory, sizeLimit: ...}` с учётом memory limit; `--gpus` только при установленном NVIDIA Container Toolkit; в WSL проверять `docker info`, а не `command -v docker`. Статическая проверка манифестов: `kubeconform -strict -summary -kubernetes-version X` (один бинарь, офлайн со схемами в репозитории проверено без сети); это валидация схем, а не поведения.

**Что осталось непроверенным (в план приёмки)**

- всё на реальном macOS (`curl --pinnedpubkey` с Apple 8.7.1, `select.kqueue`, лимит 104 байта, `shm_open`, `sysctl`);
- Ubuntu 20.04 (curl 7.68): `--http2`, трейлеры;
- cgroup v1 и реальный контейнер с `--cpus`/limits (разбор проверен на фиктивном дереве);
- `compute_cap` < 7.5 на реальном железе (правило следует из issue и уведомления PyTorch);
- полная установка torch (скачивание колёс) - умышленно не выполнялась; проверялись только `--dry-run` и индексы.

Файлы эксперимента (не коммитятся): `/tmp/claude-1000/-home-viv-dev-repos-Colosseum/b41cd787-f9ea-471b-9664-8578a7f7b4b1/scratchpad/research/` (`code/hw.py`, `code/scan_torch_indexes.py`, `code/uds.py`, `code/shm_main*.py`, `code/pdeath.py`, `code/pp.py`, `grpc_srv.py`, `grpc_srv2.py`, `kc/` со схемами, `src/` с копиями исходников uv, curl, документов).
