# curated_set → GenEx raw ERP（Vast 操作入口）

本分支新增 `initializer_batch.py`，**一次載入 initializer，依序處理每張圖**，不是同時多張 GPU batching；不載 Explorer、不為每張另開 subprocess。原生 `initializer.py` 不改動。預設 2048×1024、guidance 3.5、50 steps、90° cube front；輸出直接保存 native raw ERP，沒有貼回來源／seam 修補。原始長方圖會中心裁成正方形，再縮成 512²；先 EXIF transpose、轉 RGB。不要把保留區 conditioning 當逐像素保真。

## 0. 驗證範圍／授權

已完成 CPU fixture 測試與真實 metadata 的 plan；**未下載權重、未跑 CUDA／GPU inference，也未量 VRAM／速度／品質**。沒有可保證的卡型或最快 ETA。`requirements-initializer.txt` 是依 source 選定的隔離基線，不是 GPU 實測 lock。首次 smoke test 與全批執行需使用者另行授權，以下推論指令是供未來已授權的 Vast 主機操作；本次未執行。請先確認來源圖權利及 GenEx／FLUX.1-Fill-dev 的非商業與下游用途條款；HF CC-BY 標籤不等於全套授權已清除。

固定 upstream：`eda604588d0a205c57e2b2599eec566d97162c2e`。建立分支時 fork 與 upstream 沒有 delta。
模型：`genex-world/World-Initializer-image-to-panorama@bdf9a02257c4c2728bcd90404d53687a9641e205`。

## 1. 取得分支、建立獨立環境

```bash
git clone --branch feat/curated-initializer-batch --single-branch https://github.com/Pierce-Su/genex.git
cd genex
# 選 Python 3.11 的 Vast/PyTorch image；不要改系統環境。
python3.11 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r code/inference/requirements-initializer.txt
```

先確認主機原有 CUDA PyTorch 能用：

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")'
```

`--system-site-packages` 會沿用 image 既有 torch。若 image 沒有 PyTorch，改在 venv 依 [PyTorch 官方安裝選擇器](https://pytorch.org/get-started/locally/) 安裝符合 GPU／driver 的 CUDA wheel，不要先裝 CPU wheel；例如新架構 GPU 不能任意用舊 CUDA。此文件不宣稱任何 Vast image 已實機驗證。不要安裝合併的 Explorer `requirements.txt` 來跑 initializer。

只要 CPU plan，可用普通 Python >=3.11 的獨立 venv，只裝 `Pillow==11.3.0`；不需 torch、diffusers、numpy、GPU 或 HF token。CPU tests 另外需要 numpy（可用 requirements 中版本）。

## 2. 放置輸入（實際遠端位置由操作者指定）

以下 `/ABS/PATH/...` 都是**佔位符，不是已確認的 Vast 路徑**。上傳／掛載你的 curated_set，保留：

```text
curated_set/
  metadata.json
  photorealistic/0000.jpg ...
  stylized/0005.jpg ...
```

CLI 讀 `samples[]` 下每個 `photorealistic`／`stylized` variant 的 `filename` 和**該 variant 的 exact prompt**；同 index 的兩個版本會各自輸出，不會被頂層 style 或相同檔名蓋掉。依 index、variant、key 穩定排序。不是每個資料夾共用一個 prompt，也不是寫死 140。

本機提供的 metadata 實際為 133 個 sample rows，70 photo + 70 stylized = 140 個 variant jobs（含 7 個 paired index）；新目錄名稱帶 curated140 的 `original_metadata.json` 仍是相同結構。`original_path` 僅保留 provenance，不拿舊機器路徑猜輸入。若 metadata 在別處，用 `--metadata /ABS/PATH/original_metadata.json`，`--input-root` 仍指含兩個來源圖子目錄的 curated_set，不是舊生成 ERP 目錄。

```bash
export INPUT=/ABS/PATH/curated_set
export OUTPUT=/ABS/PATH/genex-raw-v1
python code/inference/initializer_batch.py --input-root "$INPUT" --output-root "$OUTPUT" --plan > plan.json
# 單張預覽（key 必須存在於 metadata）
python code/inference/initializer_batch.py --input-root "$INPUT" --output-root "$OUTPUT" --select photorealistic_0000 --limit 1 --plan
```

plan 會讀／hash 所選來源圖、檢查解碼與 crop、列出完整 jobs JSON；不建 output-root，不載模型。`--dry-run` 同義。若缺圖／key 不存在會退出，不偷偷少跑。`--limit 1` 依固定排序取一個 variant，不是取整對。

## 3. 先一張，再全批；保留日誌

先確認模型使用條款與下載權限；如 HF 要求登入，使用標準 HF 安全登入方式，不把 token 寫進程式／log。第一次實際執行才會載入固定 revision 權重，可用 HF_HOME 放在持久磁碟。

```bash
# INPUT / OUTPUT 必須已改成真實路徑
set -o pipefail
python -u code/inference/initializer_batch.py \
  --input-root "$INPUT" --output-root "$OUTPUT" \
  --select photorealistic_0000 --limit 1 --seed 0 2>&1 | tee smoke.log
# 檢查 rawERP / front / manifest 後，同一個 OUTPUT 接續全批：
python -u code/inference/initializer_batch.py \
  --input-root "$INPUT" --output-root "$OUTPUT" --seed 0 --resume 2>&1 | tee batch.log
```

smoke 是獨立程序，會載一次；後續全批又載一次，但**全批的每張共用同一個 resident pipeline**。不要以 shell 迴圈每張呼叫 CLI。全批若全部已驗證完成，完全不載模型。

長時間用 tmux：`tmux new -s genex`，在 tmux 內 `cd` 到 repo、`source .venv/bin/activate`、重新 export INPUT／OUTPUT，再執行上面全批命令。`Ctrl-b d` 離開，`tmux attach -t genex` 回來。保留 `set -o pipefail`，避免 tee 掩蓋推論失敗的退出碼。

若整卡載入 OOM，可在**新的 output root**先以 `--cpu-offload` 做單張試跑；此模式仍只建一個 pipeline，模組會在 CPU／GPU 之間移動，可能較慢且吃 RAM，未實測。只支援預設 cuda 的 GPU 0；不是多 GPU scheduler。換 offload／guidance／環境／seed 必須換 output root，不能混進先前完成批次。

## 4. 結果、resume、失敗處理

```text
OUTPUT/
  photorealistic_0000/
    rawERP.png           # 原生 2048x1024 RGB
    front.png            # 實際 pipeline 回傳的 512x512 crop
    source.jpg           # 原始來源 bytes；副檔名依輸入
    manifest.json
  stylized_0005/...
  run-<time>-<id>.json    # completed/skipped/failed_key/error/時間
  .batch.lock            # Linux 檔案鎖，禁止同 output-root 並行 writer
```

manifest 有來源 hash／相對路徑、variant metadata（含原始 provenance）、exact/effective prompt、EXIF/RGB 規則、中心 crop box、每張 seed、model revision、pipeline/driver hash、套件版本、device/offload/settings、三個 artifact hashes 與推論秒數。RNG 用 base seed + 穩定 key 的 SHA256 派生，呼叫前固定 Python／numpy／torch CPU 和全部 CUDA RNG；native call **不傳不支援的 generator 或 steps kwargs**，繼承 50 steps。不同硬體／CUDA／套件不保證 bit-exact。

`--resume` 在載模型前驗證所有所選既有目錄的 identity、artifact hashes、尺寸／RGB；不僅看檔案存在。prompt/source/settings／程式／套件版本不同就拒絕，請另用 output root。沒有 `--resume` 時，既有 sample 直接報錯。輸出先寫 `.partial-*` 再發佈 sample 目錄，未完成檔不會冒充成功；如保留 `.partial-*`，先檢查失敗報告，手動移出封存後再重跑亦可，它不會被當已完成。

任何推論錯誤都立即停整批，不繼續用可能 poisoned 的 CUDA state；修好原因後用**新程序**、相同設定 `--resume`。不要在同 Python process 捕捉 OOM 然後硬跑下一張。run JSON 保存失敗 key 和原因，stdout 有逐張進度與總結。SIGKILL／機器斷電可能來不及寫 run JSON；已完成且 hash 驗證通過者可 resume。不是完整斷電 durability 保證，也不支持惡意 writer 或跨機網路磁碟鎖語義。

## 5. CPU 驗證

```bash
python -m unittest discover -s tests -v
```

fixture 以假的生成輸出驗證多張單一 loader、distinct variant prompts、原生 kwargs 邊界／trigger／50 steps、固定排序／seed、plan 不載模型、resume 全驗證、source/prompt/output 改動拒絕、鎖與 fail-fast。這些藍色測試圖不是模型結果，CPU 綠燈不證明 CUDA、顯存、全景品質或速度。
