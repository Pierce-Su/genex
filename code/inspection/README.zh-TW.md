# GenEx CPU 球面檢查

這是既有 panorama-spherical-check v0.2.0 的獨立副本及 GenEx manifest 批次 adapter；不載模型、不使用 GPU、不更動生成資料夾。依賴 Python、NumPy、Pillow，優先重用已有環境，不要為這個工具更動常駐推論環境。

先測試：

    CPU_PYTHON -m unittest discover -s code/inspection -p 'test_*.py' -v

完成且已核對轉存的全140資料：

    CPU_PYTHON code/inspection/batch_spherical.py \
      --input-root /local/genex-raw-v1 \
      --output-root /local/genex-spherical-checks \
      --expected 140 --exit-marker /local/genex-batch.exit --size 512

只驗證指定已完成樣本（不等生成完成）：

    CPU_PYTHON code/inspection/batch_spherical.py \
      --input-root /workspace/genex-raw-v1 \
      --output-root /workspace/genex-spherical-checks/sample \
      --expected 1 --key photorealistic_0000 --size 512

`--exit-marker` 必須是0；不等待、不輪詢。全批次之前必須由執行 owner 另核對140個 identity 與原始 plan、70+70類別、來源/config/log/exit marker保存完整；這個 adapter 的數量與 hash 驗證不取代原始 plan 一致性驗證。

每次在 output-root 下建立新的 batch_spherical_check[_timestamp_suffix]，原有結果不覆寫。先檢查所有 sample manifest/identity/source hash、所有 artifact hash 與 decode/native尺寸；保留 inventory.json。每圖產生0,45,...315度水平視角（yaw180跨原ERP接縫）加zenith/nadir，各512方形90度FoV，同一相機中心。保留 per-image manifest、14個view/preview影像、review.md，以及可本機開啟的index.html。rendered.jsonl逐圖落盤，summary.json只在全部成功且原始資料再次驗證後產生。失敗時保留診斷結果；重跑建立新folder。

所有自動產物維持pending_manual_review；成功render不代表品質通過。需實際看contact sheet、ERP及source對照後，寫review.md、更新該manifest review_status=reviewed_2d與review_sha256。未看過的圖不可標已審核。raw/front/source都不修改；複製raw/source至derived tree以便獨立保留。manifest記錄的是生成時絕對路徑，離線搬移後以index相對連結及hash為準。

這是2D同心重投影，不是3D重建、幾何有效性或物理全景保證。yaw0為ERP中央、yaw正向右、pitch正向上；不是dataset校準pose。連續接縫也不證明語義一致。Source/front不是registered pixel comparison。

本次測量：Vast既有/venv/main（NumPy2.5.4/Pillow13.0.0），18項CPU tests通過，真實photorealistic_0000生成10視圖+14輸出影像，總2.7183秒；140線性外推380.56秒（不含transfer/人工審核，非全批實測且非minipc測量）。建議先轉存至minipc，不要為離線CPU檢查延長付費GPU閒置。此單樣本實看yaw180硬接縫、天花/地板塗抹，評needs_review，不能外推其他139張。
