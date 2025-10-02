import xarray as xr
import numpy as np
import os
import warnings
import cfgrib
from cfgrib import xarray_store
import pygrib  # より低レベルなGRIB2読み取りのために追加

def analyze_grib_file_with_pygrib(file_path: str):
    """
    pygribを使用してGRIB2ファイルの詳細情報を取得する関数
    
    Args:
        file_path (str): 解析するGRIB2ファイルのパス
    """
    print(f"=================================================")
    print(f" GRIB2ファイル詳細解析: {os.path.basename(file_path)}")
    print(f"=================================================")
    
    if not os.path.exists(file_path):
        print(f"[エラー] ファイルが見つかりません: {file_path}")
        print("=================================================\n")
        return
    
    try:
        # pygribでファイルを開く
        grbs = pygrib.open(file_path)
        
        # ファイル内のメッセージ数を取得
        message_count = grbs.messages
        print(f"ファイル内のメッセージ数: {message_count}")
        print("-------------------------------------------------\n")
        
        # 変数ごとにグループ化するための辞書
        variables_dict = {}
        
        # 各メッセージを読み取り
        for i, grb in enumerate(grbs):
            try:
                # 基本情報の取得
                short_name = grb.shortName if hasattr(grb, 'shortName') else 'unknown'
                param_name = grb.parameterName if hasattr(grb, 'parameterName') else 'unknown'
                level_type = grb.typeOfLevel if hasattr(grb, 'typeOfLevel') else 'unknown'
                level = grb.level if hasattr(grb, 'level') else 0
                
                # 変数キーの作成
                var_key = f"{short_name}_{level_type}_{level}"
                
                if var_key not in variables_dict:
                    variables_dict[var_key] = {
                        'short_name': short_name,
                        'parameter_name': param_name,
                        'level_type': level_type,
                        'level': level,
                        'messages': []
                    }
                
                variables_dict[var_key]['messages'].append(grb)
                
            except Exception as e:
                print(f"メッセージ {i+1} の読み取り中にエラー: {e}")
        
        grbs.close()
        
        # 変数ごとの詳細表示
        print(f"\n検出された変数の種類: {len(variables_dict)}")
        print("=================================================\n")
        
        for idx, (var_key, var_info) in enumerate(variables_dict.items()):
            print(f"---▼ 変数 {idx+1}/{len(variables_dict)}: {var_key} ▼---")
            print(f"  短縮名: {var_info['short_name']}")
            print(f"  パラメータ名: {var_info['parameter_name']}")
            print(f"  レベルタイプ: {var_info['level_type']}")
            print(f"  レベル: {var_info['level']}")
            print(f"  メッセージ数: {len(var_info['messages'])}")
            
            # 最初のメッセージから詳細情報を取得
            if var_info['messages']:
                grb = var_info['messages'][0]
                try:
                    # データの取得と統計情報
                    values = grb.values
                    print(f"\n  [データ情報]")
                    print(f"    - 形状: {values.shape}")
                    print(f"    - 最小値: {np.nanmin(values):.2f}")
                    print(f"    - 最大値: {np.nanmax(values):.2f}")
                    print(f"    - 平均値: {np.nanmean(values):.2f}")
                    
                    # 座標情報
                    lats, lons = grb.latlons()
                    print(f"\n  [座標情報]")
                    print(f"    - 緯度範囲: {lats.min():.2f} ～ {lats.max():.2f}")
                    print(f"    - 経度範囲: {lons.min():.2f} ～ {lons.max():.2f}")
                    
                    # 時刻情報
                    print(f"\n  [時刻情報]")
                    print(f"    - 解析時刻: {grb.analDate}")
                    print(f"    - 予報時刻: {grb.validDate}")
                    
                except Exception as e:
                    print(f"  詳細情報の取得中にエラー: {e}")
            
            print(f"---▲ 変数 {idx+1}/{len(variables_dict)} 終了 ▲---\n")
            
    except Exception as e:
        print(f"[エラー] ファイルの読み取り中にエラーが発生しました: {e}")
        print("pygribがインストールされていない場合は、以下のコマンドでインストールしてください:")
        print("  pip install pygrib")
        print("\nまたは、cfgribのみを使用した代替方法を試します...\n")
        
        # cfgribを使用した代替方法
        analyze_with_cfgrib_alternative(file_path)

def analyze_with_cfgrib_alternative(file_path: str):
    """
    cfgribのみを使用してGRIB2ファイルを解析する代替方法
    
    Args:
        file_path (str): 解析するGRIB2ファイルのパス
    """
    print("[代替方法] cfgribを使用した解析")
    print("-------------------------------------------------")
    
    try:
        # まず全体を読み込んでみる
        ds = xr.open_dataset(file_path, engine='cfgrib')
        
        print("\n[データセット概要]")
        print(ds)
        
        print("\n[座標情報]")
        for coord_name, coord in ds.coords.items():
            print(f"  {coord_name}: {coord.shape}")
            if coord.size < 10:
                print(f"    値: {coord.values}")
            else:
                print(f"    範囲: {coord.values.min()} ～ {coord.values.max()}")
        
        print("\n[データ変数]")
        for var_name in ds.data_vars:
            var = ds[var_name]
            print(f"\n  変数名: {var_name}")
            print(f"    形状: {var.shape}")
            print(f"    次元: {var.dims}")
            
            # 属性情報
            print("    属性:")
            for attr_name, attr_value in var.attrs.items():
                print(f"      - {attr_name}: {attr_value}")
            
            # 統計情報
            try:
                values = var.values
                print(f"    統計:")
                print(f"      - 最小値: {np.nanmin(values):.2f}")
                print(f"      - 最大値: {np.nanmax(values):.2f}")
                print(f"      - 平均値: {np.nanmean(values):.2f}")
            except:
                print("    統計: 計算できません")
        
        ds.close()
        
    except Exception as e:
        print(f"[エラー] cfgribでの読み取りに失敗しました: {e}")
        
        # 複数のデータセットが含まれている可能性があるため、個別に読み込む
        print("\n[追加試行] 個別のデータセットとして読み込みを試みます...")
        
        try:
            # cfgribのopen_datasetsを使用（複数のデータセットを返す）
            datasets = cfgrib.open_datasets(file_path)
            
            print(f"\n検出されたデータセット数: {len(datasets)}")
            
            for i, ds in enumerate(datasets):
                print(f"\n--- データセット {i+1}/{len(datasets)} ---")
                print(f"変数: {list(ds.data_vars)}")
                print(f"座標: {list(ds.coords)}")
                
                # 各変数の簡単な情報
                for var_name in ds.data_vars:
                    var = ds[var_name]
                    print(f"\n  {var_name}:")
                    print(f"    形状: {var.shape}")
                    if 'long_name' in var.attrs:
                        print(f"    説明: {var.attrs['long_name']}")
                    if 'units' in var.attrs:
                        print(f"    単位: {var.attrs['units']}")
                
                ds.close()
                
        except Exception as e2:
            print(f"[エラー] 個別読み込みも失敗しました: {e2}")

def main():
    """
    メイン実行関数
    """
    # 解析対象のファイルリスト（PDFに記載されているファイル名形式）[1]
    files_to_analyze = [
        'MSM_data/201201/Z__C_RJTD_20120101000000_MSM_GPV_Rjp_L-pall_FH00_grib2.bin',
        'MSM_data/201201/Z__C_RJTD_20120101000000_MSM_GPV_Rjp_L-pall_FH03_grib2.bin',
        'MSM_data/201201/Z__C_RJTD_20120101000000_MSM_GPV_Rjp_L-pall_FH06_grib2.bin',
        'MSM_data/201201/Z__C_RJTD_20120101000000_MSM_GPV_Rjp_Lsurf_FH00_grib2.bin',
        'MSM_data/201201/Z__C_RJTD_20120101000000_MSM_GPV_Rjp_Lsurf_FH03_grib2.bin',
        'MSM_data/201201/Z__C_RJTD_20120101000000_MSM_GPV_Rjp_Lsurf_FH06_grib2.bin',
        'MSM_data/201201/Z__C_RJTD_20120101000000_MSM_GPV_Rjp_Prr_FH00-03_grib2.bin',
        'MSM_data/201201/Z__C_RJTD_20120101000000_MSM_GPV_Rjp_Prr_FH03-06_grib2.bin',
    ]
    
    print("=================================================")
    print(" メソアンサンブル数値予報モデル GRIB2ファイル解析")
    print("=================================================")
    print(f"解析対象ファイル数: {len(files_to_analyze)}")
    print("")
    
    # 各ファイルを解析
    for i, file_path in enumerate(files_to_analyze):
        print(f"\n【ファイル {i+1}/{len(files_to_analyze)}】")
        
        # ファイル名からタイプを判定
        basename = os.path.basename(file_path)
        if 'L-pall' in basename:
            print("タイプ: 気圧面データ")
        elif 'Lsurf' in basename:
            print("タイプ: 地上データ")
        elif 'Prr' in basename:
            print("タイプ: 降水量データ")
        
        # 解析実行
        analyze_grib_file_with_pygrib(file_path)
    
    print("\n=================================================")
    print(" 全ファイルの解析が完了しました")
    print("=================================================")

if __name__ == '__main__':
    # 警告を抑制
    warnings.filterwarnings('ignore')
    
    # メイン処理を実行
    main()