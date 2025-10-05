```bash
notify-run gpu01 -- nohup python swinunet_main_v1.py > swinunet_main_v1.log 2>&1 &

notify-run gpu01 -- nohup python optimization_nc_data.py > optimization_nc_data.log 2>&1 &

notify-run gpu01 -- nohup python3 check_data.py > check_data.log 2>&1 &

notify-run gpu01 -- nohup python3 convert_msm_data.py > convert_msm_data.log 2>&1 &

notify-run gpu01 -- nohup python3 convert_nc.py > convert_nc.log 2>&1 &

notify-run gpu01 -- nohup python3 check_output_nc.py > check_output_nc.log 2>&1 &

notify-run gpu01 -- nohup python swinunet_main_v2.py > swinunet_main_v2.log 2>&1 &

notify-run gpu02 -- nohup python swinunet_main_v5.py > swinunet_main_v5.log 2>&1 &

notify-run gpu02 -- nohup python analyze_1h_bin_distribution.py > analyze_1h_bin_distribution.log 2>&1 &

notify-run gpu01 -- nohup python separate_main_v1.py > separate_main_v1.log 2>&1 &

notify-run gpu01 -- nohup python separate_main_v2.py > separate_main_v2.log 2>&1 &

nohup python data_check_separate.py > data_check_separate.log 2>&1 &

nohup python check_output_nc.py > check_output_nc.log 2>&1 &

```

/home/devel/work_takasuka_git/docker_miniconda/src/CompresionRain/analyze_1h_bin_distribution.py

/home/devel/work_takasuka_git/docker_miniconda/src/CompresionRain/
タスクの削除

```bash
pkill -f "swinunet_main_v1.py"

pkill -f "optimization_nc_data.py"

pkill -f "convert_data_v1.py"

pkill -f "convert_msm_data.py"

pkill -f "convert_nc.py"

pkill -f "swinunet_main_v3.py"

pkill -f "swinunet_main_v5.py"

pkill -f "separate_main_v1.py"

pkill -f "check_output_nc.py"
```

```bash
for year in MSM_data/*/; do
    echo "Processing: $year"
    rm -f "$year"*.5b7b6.idx
done
```

DDP による GPU エラー時のコマンド

```bash
sudo kill -9 1493785
```

## gpu01 → mac

```bash
rsync -avz --progress gpu01:/home/devel/work_takasuka_git/docker_miniconda/src/CompresionRain/swin-unet_ratio_result_v2 /Users/takumi0616/Develop/docker_miniconda/src/CompresionRain/result_gpu01
```
