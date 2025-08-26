```bash
notify-run gpu01 -- nohup python swinunet_main_v1.py > swinunet_main_v1.log 2>&1 &

notify-run gpu01 -- nohup python optimization_nc_data.py > optimization_nc_data.log 2>&1 &

notify-run gpu01 -- nohup python3 check_data.py > check_data.log 2>&1 &

notify-run gpu01 -- nohup python3 convert_msm_data.py > convert_msm_data.log 2>&1 &

notify-run gpu01 -- nohup python3 convert_nc.py > convert_nc.log 2>&1 &

notify-run gpu01 -- nohup python3 check_MSM_data_nc.py > check_MSM_data_nc.log 2>&1 &

notify-run gpu01 -- nohup python swinunet_main_v2.py > swinunet_main_v2.log 2>&1 &

notify-run gpu01 -- nohup python swinunet_main_v3.py > swinunet_main_v3.log 2>&1 &

notify-run gpu01 -- nohup python swinunet_main_v4.py > swinunet_main_v4.log 2>&1 &

```

タスクの削除

```bash
pkill -f "swinunet_main_v1.py"

pkill -f "swinunet_main_v2.py"

pkill -f "convert_data_v1.py"

pkill -f "convert_msm_data.py"

pkill -f "convert_nc.py"

pkill -f "swinunet_main_v3.py"

pkill -f "swinunet_main_v4.py"
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
rsync -avz --progress gpu01:/home/devel/work_takasuka_git/docker_miniconda/src/CompresionRain/swin-unet_main_result_v4 /Users/takumi0616/Develop/docker_miniconda/src/CompresionRain/result_gpu01
```
