from tqdm import tqdm
import numpy as np
from typing import Tuple

def save_descaling_params(k:int, descaling_params:list, pca_data:list) -> Tuple[list, list]: 
    '''
    Min-Max 스케일링을 수행하여 PCA 데이터를 [0, 1] 범위로 정규화하고,
    원본 값으로 복원(역변환)하는 데 필요한 스케일링 파라미터(최솟값, 최댓값)를 저장하는 함수입니다.

    params:
        - k: int. PCA 데이터의 차원(dimension) 크기입니다. (예: PCA 컴포넌트의 개수)
        - descaling_params: list. 각 차원별 [최솟값, 최댓값] 정보를 저장할 빈 리스트입니다.
                          이 리스트는 함수 내에서 채워집니다.
        - pca_data: np.ndarray. 각 이미지에 대해 PCA를 거쳐 차원 축소된 데이터 배열입니다.
                    이 배열은 함수 내에서 직접 [0, 1] 범위로 정규화(in-place modification)됩니다.

    returns:
        - Tuple[list, np.ndarray]:
            - descaling_params: 각 차원별로 저장된 [최솟값, 최댓값] 리스트입니다.
                                (변환 전 원본 데이터의 min, max가 아닌, min-shift 후의 min, max)
            - pca_data: [0, 1] 범위로 Min-Max 정규화된 PCA 데이터 배열입니다.
    '''
    for i in tqdm(range(k)):
        # 최솟값 descaling_params 저장
        descaling_params[i].append(pca_data[:,i].min())
    
        # 최솟값이 0보다 작은 경우, 전체에 min 값을 더해줌
        # 최솟값이 0보다 큰 경우, 전체에 max 값을 빼줌
        if pca_data[:,i].min() < 0:
            pca_data[:,i] += np.abs(pca_data[:,i].min())
        else:
            pca_data[:,i] -= np.abs(pca_data[:,i].min())
    
        # 최대값 descaling_params 저장
        descaling_params[i].append(pca_data[:,i].max())

        # Max로 나눔으로써, 최댓값 1으로 조정
        pca_data[:,i] /= pca_data[:,i].max()

    return descaling_params, pca_data


def descaling_points(d_points:list, scales:list, tfrm):
    '''
    차원 축소된 PCA 데이터(d_points)를 원본 스케일로 역변환(descaling)한 다음,
    PCA 모듈의 inverse_transform을 사용하여 원본 이미지 공간으로 복원하는 함수입니다.

    params:
        - d_points: np.ndarray. 복원할 차원 축소된 PCA 데이터 배열입니다.
                    (shape = (샘플 인덱스, PCA 차원)). 이 데이터는 [0, 1] 범위로 정규화되어 있다고 가정합니다.
        - scales: list. `save_descaling_params` 함수에서 저장된 각 차원별 [최솟값, 최댓값] 리스트입니다.
                  이 정보는 [0, 1] 범위의 데이터를 원래 스케일로 되돌리는 데 사용됩니다.
        - tfrm: PCA 변환기 객체 (예: `sklearn.decomposition.PCA` 인스턴스).
              이 객체는 `inverse_transform` 메서드를 가지고 있어야 합니다.

    returns:
        - np.ndarray: 원본 이미지 크기(1차원 배열)로 복원된 데이터 배열입니다.
                      (예: flatten된 이미지 픽셀 값 배열).
    '''
    for col in range(d_points.shape[1]):
        d_points[:,col] *= scales[col][1] # max 
        d_points[:,col] += scales[col][0] # min
    reconstruction = tfrm.inverse_transform(d_points)
    return reconstruction







    

    