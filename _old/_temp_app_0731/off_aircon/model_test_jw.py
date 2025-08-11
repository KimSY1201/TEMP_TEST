import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import logging
from scipy.ndimage import gaussian_filter, zoom
from multiprocessing import Pool, cpu_count
from tqdm import tqdm # 일반 콘솔 환경에서 tqdm 사용 시

# 주의: Jupyter Notebook 환경에서는 matplotlib.pyplot을 Pool 내부에서 사용하면 문제가 생길 수 있습니다.
# plot_results=True 로 시각화 코드를 활성화한다면, plot_results=False 로 설정하고 Pool 내부에서 시각화를 시도하지 마세요.

# 로깅 설정은 if __name__ == '__main__': 블록 밖에 있어도 됩니다.
logging.basicConfig(filename='parallel_processing.log', level=logging.INFO,
                    format='%(asctime)s - %(process)d - %(levelname)s - %(message)s')

# process_data_to_image2 함수 정의 (이전과 동일, plot_results는 이미 주석 처리됨)
def process_data_to_image2(data_64_columns, sigma=1.0, output_shape=(24, 24), plot_results=True):
    if data_64_columns.shape[0] != 64:
        raise ValueError("입력 데이터는 정확히 64개의 원소를 가져야 합니다.")
    
    array_8x8 = data_64_columns.reshape((8, 8))
    zoom_factor_h = output_shape[0] / array_8x8.shape[0]
    zoom_factor_w = output_shape[1] / array_8x8.shape[1]
    array_24x24 = zoom(array_8x8, zoom=(zoom_factor_h, zoom_factor_w), order=3)
    filtered_24x24 = gaussian_filter(array_24x24, sigma=sigma)
    return filtered_24x24

# process_wrapper_with_data 함수 정의:
# 이 함수는 Pool의 작업자가 실행할 함수이며, 필요한 모든 데이터를 인자로 명시적으로 받습니다.
def process_wrapper_with_data(row_idx, pixel_data_values, sigma, output_shape):
    logging.info(f"PID {os.getpid()}: Processing row {row_idx}...")
    
    # 여기서 astype('float32')는 pixel_data_values가 이미 tdf에서 float32로 변환된 값이라고 가정하면 불필요합니다.
    # 하지만 데이터가 어떻게 전달되는지에 따라 안전을 위해 유지할 수도 있습니다.
    # 여기서는 tdf에서 이미 변환된 값을 전달받는다고 가정하고 제거했습니다.
    processed_image_2d = process_data_to_image2(pixel_data_values, sigma=sigma, output_shape=output_shape)
    return processed_image_2d.flatten()


# --- 메인 실행 블록 ---
# 이 블록 안에 모든 데이터 로드, 전처리, 병렬 처리 관련 코드가 들어가야 합니다.
if __name__ == '__main__':
    # 데이터 로드 및 전처리
    dtype_spec = {68: str} # Columns (69) - 인덱스 68번 컬럼
    tdf = pd.DataFrame(pd.read_csv('./total_thermal_noAC.csv', index_col=0, header=0, dtype=dtype_spec))
    tdf = tdf.drop(columns=['timestamp', 'etc_1', 'etc_2'])

    # 'heater' 제거 및 'avg' 컬럼 추가
    # apply(pd.to_numeric, errors='coerce')를 사용하여 NaN 처리
    tdf['avg'] = tdf.iloc[:, 1:65].apply(pd.to_numeric, errors='coerce').mean(axis=1).round(2)
    tdf = tdf.loc[tdf['status']!='heater', :]

    # 'status' 컬럼 클리닝
    tdf['status'] = tdf['status'].astype(str).str.replace('1', '').str.replace('2', '')

    # tdf8 대신 tdf의 필요한 부분만 사용
    # 픽셀 데이터를 float32로 확실히 변환 (전체 데이터프레임에 대해 한 번만)
    tdf.iloc[:,1:65] = tdf.iloc[:,1:65].astype('float32')
    # print(tdf.iloc[:,1:65].info())

    # reset_index는 tdf8이 아니라 tdf에 적용하는 것이 좋습니다.
    tdf.reset_index(drop=True, inplace=True)
    
    # # fdf의 구조 정의 (병렬 처리 결과를 받기 위한 컬럼 이름)
    pixel_col_names = [f'processed_pixel_{j}' for j in range(24 * 24)]

    # print("병렬 처리 시작...")
    logging.info("Main process: Parallel processing started.")

    num_cores = cpu_count()
    print(f"사용할 CPU 코어 수: {num_cores}")
    logging.info(f"Main process: Using {num_cores} CPU cores.")

    # Pool.starmap을 위한 태스크 리스트 생성
    # 각 태스크는 (row_idx, pixel_data_values, sigma, output_shape) 튜플입니다.
    tasks_for_pool = []
    for i in range(len(tdf.index)):
        # tdf.iloc[i, 1:65].values는 이미 위에서 float32로 변환된 상태이므로 다시 astype('float32') 할 필요 없음
        pixel_data = tdf.iloc[i, 1:65].astype('float32').values
        tasks_for_pool.append((i, pixel_data, 0.8, (24, 24)))

    with Pool(processes=num_cores) as pool:
        # pool.starmap 사용: tasks_for_pool의 각 튜플이 process_wrapper_with_data의 인자로 전달됩니다.
        processed_results_list = list(tqdm(pool.starmap(process_wrapper_with_data, tasks_for_pool, chunksize=100),
                                           total=len(tdf.index), desc="Processing Rows"))

    print("병렬 처리 완료.")
    logging.info("Main process: Processing complete.")

    # # 병렬 처리된 결과들로 fdf_parallel_result 데이터프레임 생성
    final_processed_pixel_data = np.array(processed_results_list)

    fdf_parallel_result = pd.DataFrame(
        final_processed_pixel_data,
        index=tdf.index, # tdf와 동일한 인덱스 사용
        columns=pixel_col_names
    )
    # 'status' 컬럼을 tdf에서 복사하여 fdf_parallel_result의 첫 번째 컬럼으로 추가
    mean_pixel_values = fdf_parallel_result.iloc[:, 1:].mean(axis=1).round(2)    

    fdf_parallel_result.insert(0, 'status', tdf['status'])
    fdf_parallel_result.insert(1, 'mean', mean_pixel_values)
    
    print("\n병렬 처리 후 fdf의 shape:", fdf_parallel_result.shape)
    print("병렬 처리 후 fdf의 첫 2행:\n", fdf_parallel_result.head(2))
    fdf_parallel_result.to_csv('./24f_hn_thermal.csv', encoding='utf-8')