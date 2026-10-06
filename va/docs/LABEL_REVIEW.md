# 標註複核報告（Label Review）— v7.2

> **身分聲明**：本文件依目前 `model_report.json` 的 `label_review` 區塊整理。質化查證僅代表目前的複核紀錄，
> 不是具公信力的人工標註結論。畢業專題定稿前，仍應由專題團隊自行確認並對論文內容負責。

## 一、方法

針對 `model_report.json → label_review` 的兩類個案：

- `benign_with_high_proba`：資料標為正常（0），模型 OOF 機率偏高。
- `scam_with_low_proba`：資料標為詐騙（1），模型 OOF 機率偏低。

以下 OOF 機率均已更新為目前模型報告中的數值。

## 二、已查證個案（高信心）

### 1. `gh168-bank-loan-money.tw` — OOF probability = 0.9736

**查證結果**：先前複核紀錄將此案例判定為合法貸款仲介網站，因此原始正常標籤可視為合理，模型高分屬 False Positive。
其域名字面含有 `bank`、`loan`、`money` 等金融詞彙，且 URL 含有廣告追蹤參數，可能同時觸發多個風險訊號。

### 2. `zama.tw` — OOF probability = 0.0120

**查證結果**：先前複核紀錄將此案例判定為 165 通報所涉詐騙平台，因此原始詐騙標籤可視為合理，模型低分屬 False Negative。
此案例 URL 本身相對乾淨，反映 URL-only 模型對社交工程型詐騙的先天限制。

## 三、目前模型中高風險正常樣本（建議人工確認）

| 網域 | OOF probability | 初步判斷 |
|---|---:|---|
| `gh168-bank-loan-money.tw` | 0.9736 | 合法貸款仲介，確認 FP |
| `disneyplus.com` | 0.7673 | 國際知名品牌官方網域，建議確認 |
| `uniqlo.com` | 0.6805 | 國際知名品牌官方網域，建議確認 |
| `subaru.asia` | 0.6797 | 官方／品牌網域，建議確認 |
| `evaair.com` | 0.6721 | 長榮航空官方網域，建議確認 |
| `mercci22.com` | 0.6620 | 合法網站，建議確認 |
| `swarovski.com` | 0.6388 | 國際知名品牌官方網域，建議確認 |
| `rhinoshield.tw` | 0.6351 | 品牌台灣官方網域，建議確認 |
| `oncehuman.game` | 0.5557 | 遊戲官方網域，含追蹤參數，建議確認 |

## 四、目前模型中低風險詐騙樣本（建議人工確認）

| 網域 | OOF probability | 初步判斷 |
|---|---:|---|
| `zama.tw` | 0.0120 | 已查證為模型 False Negative |
| `tyurd.com` | 0.0248 | 低分 FN，建議人工確認 |
| `sents.tw` | 0.0255 | 低分 FN，建議人工確認 |
| `newedin.com` | 0.0305 | 低分 FN，建議人工確認 |
| `decrede.com` | 0.0309 | 低分 FN，建議人工確認 |
| `du96.net` | 0.0321 | 低分 FN，建議人工確認 |
| `fackgo.co` | 0.0338 | 低分 FN，建議人工確認 |
| `twlss.chende.cyoy/h5` | 0.0343 | 低分 FN，建議人工確認 |
| `based.tw` | 0.0360 | 低分 FN，建議人工確認 |
| `heu.kacsu.com` | 0.0369 | 低分 FN，建議人工確認 |
| `rthrd.com` | 0.0383 | 低分 FN，建議人工確認 |
| `ydzq.pages.dev` | 0.0395 | 低分 FN，建議人工確認 |
| `client.bitharvest.io` | 0.0429 | 低分 FN，建議人工確認 |
| `toeies.jp` | 0.0463 | 低分 FN，建議人工確認 |
| `m.guardarianc.com` | 0.0495 | 低分 FN，建議人工確認 |

## 五、給專題團隊的具體建議

1. **不要因為模型誤判就直接修改標籤。** 目前至少兩個高信心案例的紀錄顯示，問題更可能是模型本身的偵測盲點，而非資料標註錯誤。
2. 論文可將「品牌官方網站被誤判」與「URL 乾淨的社交工程詐騙被漏判」列入 Error Analysis 與研究限制。
3. 若時間允許，再人工抽查少量案例並將確認結果納入論文附錄即可，不需要把所有可能仍在運作的網站逐一訪問。
