```bash
nohup python swinunet_main.py > swinunet_main.log 2>&1 &

nohup python optimization_nc_data.py > optimization_nc_data.log 2>&1 &

nohup python3 check_data.py > check_data.log 2>&1 &

nohup python3 convert_msm_data.py > convert_msm_data.log 2>&1 &

nohup python3 convert_nc.py > convert_nc.log 2>&1 &

nohup python3 check_MSM_data_nc.py > check_MSM_data_nc.log 2>&1 &

```

タスクの削除

```bash
pkill -f "swinunet_main.py"

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

DDPによるGPUエラー時のコマンド
```bash
sudo kill -9 1493785
```