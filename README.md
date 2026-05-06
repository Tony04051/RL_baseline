# Rank-GRPO
請協助我跑Llama
## 環境

- Python 3.10
- **transformers==4.55.4** (注意: 不要升新版)
- trl[vllm]==0.21.0
- vllm==0.10.0

### for zeroshot
因為Llama是prohibited model，要先到Hugging face 取得授權
1. 在 Hugging Face 網站上同意條款：

登入你的 Hugging Face 帳號。

前往模型頁面：meta-llama/Llama-3.2-3B-Instruct

填寫簡單的資料並點擊 "Acknowledge license" 或 "Agree to terms"。通常申請後會立刻或在幾分鐘內通過。

2. 取得你的 Access Token：

點擊右上角頭像 -> Settings -> 左側選單的 Access Tokens。

點擊 "Create new token"，權限選擇 Read 即可，然後複製這串 Token。

3. 在你的伺服器環境登入：
回到你執行腳本的終端機（Terminal），輸入以下指令來登入 Hugging Face：

```Bash
huggingface-cli login
```
然後跑
```
python evaluate/zeroshot.py \     
--model_name meta-llama/Llama-3.2-3B-Instruct \
--model_root results \     
--dataset_path processed_datasets/sft_dataset \    
--catalog_path processed_datasets/gt_catalog.pkl \     
2>&1 | tee logs/Llama_zeroshot.txt
```
### for sft
```
python evaluate/eval_grpo_test.py \
--model_name meta-llama/Llama-3.2-3B-Instruct \
--model_root ../results/Llama/Llama-3.2-3B-Instruct \
--dataset_path ../processed_datasets/sft_dataset \
--catalog_path ../processed_datasets/gt_catalog.pkl  \
--checkpoint 1500 \
2>&1 | tee logs/Llama_sft.txt
```
### for rank grpo
```
python evaluate/eval_grpo_test.py \
--model_name meta-llama/Llama-3.2-3B-Instruct \
--model_root ../results/grpo/Llama/Llama-3.2-3B-Instruct_lr1e-06_kl0.001 \
--dataset_path ../processed_datasets/sft_dataset \
--catalog_path ../processed_datasets/gt_catalog.pkl  \
--checkpoint 15800 \
2>&1 | tee logs/Llama_grpo.txt
```

