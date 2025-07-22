```bash
nohup python gemini_main.py > output.log 2>&1 &

nohup python3 check_data.py > check_data.log 2>&1 &

nohup python3 convert_msm_data.py > convert_msm_data.log 2>&1 &

nohup python3 convert_nc.py > convert_nc.log 2>&1 &

```

タスクの削除

```bash
pkill -f "python gemini_main.py"

pkill -f "convert_data_v1.py"

pkill -f "convert_msm_data.py"

pkill -f "convert_nc.py"
```

```bash
for year in MSM_data/*/; do
    echo "Processing: $year"
    rm -f "$year"*.5b7b6.idx
done
```
