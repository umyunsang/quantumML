import yaml
from . import pca_func as pf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Any, List

def load_config_yaml(filepath="./config.yaml"):
    '''
    지정된 경로의 YAML 설정 파일을 로드하는 함수입니다.
    `yaml.safe_load`를 사용하여 YAML 파일로부터 안전하게 데이터를 읽어옵니다.

    Args:
        filepath (str): 로드할 YAML 설정 파일의 경로입니다. 기본값은 "./config.yaml"입니다.

    Returns:
        Dict[str, Any]: 로드된 설정 내용을 딕셔너리 형태로 반환합니다.    
    '''
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f) # Use safe_load for security
    return config

def save_config_yaml(config, filepath="./config.yaml"):
    """
    주어진 딕셔너리 형태의 설정 데이터를 YAML 파일로 저장하는 함수입니다.
    `yaml.safe_dump`를 사용하여 딕셔너리 데이터를 YAML 형식으로 파일에 씁니다.

    Args:
        config (Dict[str, Any]): 저장할 설정 데이터를 포함하는 딕셔너리입니다.
        filepath (str): 설정 데이터가 저장될 YAML 파일의 경로입니다. 기본값은 "./config.yaml"입니다.
    """    
    with open(filepath, 'w') as f:
        yaml.safe_dump(config, f, indent=2, sort_keys=False) # indent for readability

def get_ordinal_suffix(n: int) -> str:
    """
    정수에 해당하는 서수 접미사(예: 1st, 2nd, 3rd, 4th)를 반환하는 헬퍼 함수입니다.

    Args:
        n (int): 서수 접미사를 얻을 정수입니다.

    Returns:
        str: 해당 정수에 대한 서수 접미사 ('st', 'nd', 'rd', 'th').
    """
    if 11 <= n % 100 <= 13:
        return 'th'
    return {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')

def plot_kde_scatter(
    train_iter: int,
    epoch: int,
    backend,
    training_circuit,
    train_var: Dict,
    pca_data: np.ndarray,
    target_labels: str,
    q: int,
    plot_frequency: int,
    shots: int = 20,
    num_samples: int = 500,  # Renamed 'range' to 'num_samples' to avoid shadowing built-in
    dist_plot_path: str = './images/distribution_plot/'
) -> List:
    '''
    생성된 데이터의 커널 밀도 추정(KDE) 플롯을 그리고, 그 위에 실제 데이터 포인트의 산점도를 오버레이합니다.
    주어진 플롯 빈도 조건에 따라 플롯을 생성하고 저장합니다.

    Args:
        train_iter (int) : QuGAN 훈련의 총 반복 횟수.
        epoch (int) : 현재 QuGAN 훈련 에포크 횟수.
        plot_frequency (int) : 플롯 이미지를 저장할 에포크 빈도.
        backend : 양자 컴퓨터 시뮬레이터 백엔드.
        training_circuit : QuGAN 양자 회로를 생성하는 객체.
        train_var (Dict) : QuGAN의 학습 가능한 파라미터 딕셔너리.
        pca_data (np.ndarray) : 실제 데이터 포인트 (2D 배열).
        target_labels (str) : 플롯 제목에 사용될 클래스 레이블 문자열.
        q (int) : 회로의 총 큐비트 수.
        shots (int) : 회로 실행당 샷 수.
        num_samples (int) : KDE 플롯을 위한 샘플 생성 횟수.
        dist_plot_path (str) : 플롯 저장 경로.

    Returns:
        Tuple[Any, np.ndarray]:
            - plot_circ: 마지막으로 생성된 양자 회로 객체.
            - plot_data: KDE 플롯에 사용된 생성된 확률 데이터 (NumPy 배열).
    '''
    plot_data = []
    
    # 1. Generate the circuit for plotting.
    #    - This uses the current state of `train_var` to build the generator part.
    #    - Note: 'point', 'key', and 'key_value' are not used in `disc_fake_train`
    #      when Sample=True, so they can be omitted from the call here for clarity.
    plot_circ = training_circuit.disc_fake_train(train_var, None, None, None, Sample=True)
    
    # Determine the number of measured qubits (n_results)
    n_results = q // 2

    # 2. Loop to generate the data samples for the KDE plot
    for _ in range(num_samples):
        plot_job = backend.run(plot_circ, shots=shots)
        plot_results = plot_job.result().get_counts(plot_circ)
        
        # Initialize bins for probability calculation (for each qubit)
        bins_per_sample = [[0, 0] for _ in range(n_results)]

        # 3. Process the measurement results to get probabilities for each qubit
        for key_str, value in plot_results.items():
            # Correct the indentation here
            for i in range(n_results):
                # Check the i-th bit from the end of the bitstring
                if key_str[-i - 1] == '1':
                    bins_per_sample[i][0] += value # Count of '1's
                bins_per_sample[i][1] += value     # Total count (always adds up)

        # 4. Convert counts to probabilities for each qubit
        qubit_probs = []
        for b_pair in bins_per_sample:
            # Add a check to prevent ZeroDivisionError
            if b_pair[1] > 0:
                qubit_probs.append(b_pair[0] / b_pair[1])
            else:
                # If no counts were recorded, default to 0.5 (random guess)
                qubit_probs.append(0.5)

        plot_data.append(qubit_probs)
    # Convert the list of samples into a NumPy array
    plot_data = np.array(plot_data)
    
    # Check if this epoch is a plotting epoch
    if (epoch % plot_frequency == 0) or (epoch == train_iter-1):
        # --- Plotting Section ---
        # 5. Create a custom colormap for the KDE plot (smooth gradation from one color)
        #    This resolves the issue of visible color layers.
        cmap_single_color = LinearSegmentedColormap.from_list(
            "blue_gradation", ["aliceblue", "darkblue"]
        )

        # 6. Plot the graph using seaborn.jointplot.
        #    - The KDE will be of the *generated data* (`plot_data`).
        #    - No `xlabel` or `ylabel` in the function call to avoid warnings.
        plot_graph = sns.jointplot(
            x=plot_data[:, 0], y=plot_data[:, 1],
            kind='kde',
            ylim=(0, 1), xlim=(0, 1), # Assuming your probability data is in [0,1]
            fill=True,
            cmap=cmap_single_color,
            color='skyblue', # Used for marginal plots
        )
        
        # 7. Overlay the scatter plot of the *real data* (`pca_data`) for comparison.
        #    - This correctly visualizes the relationship between generated and real data.
        plot_graph.plot_joint(
            plt.scatter,
            marker='o',
            c='salmon', # Use a light red color
            s=3,
            alpha=0.5,
        )
        
        # 8. Set the custom title and remove axis labels.
        #    - This is the correct way to control the axes after plotting.
        epoch_ordinal = f'{epoch+1}{get_ordinal_suffix(epoch)}'
        chart_title = f'QuGAN Generation of class {target_labels}'
        epoch_title = f'Epoch {epoch_ordinal}'
        plot_graph.fig.suptitle(f'{chart_title}\n{epoch_title}', y=0.02)
        
        plot_graph.ax_joint.set_xlabel('')
        plot_graph.ax_joint.set_ylabel('')
        plot_graph.ax_marg_x.set_xlabel('')
        plot_graph.ax_marg_y.set_ylabel('')
        
        # 9. Ensure the save directory exists and save the figure.
        if not os.path.exists(dist_plot_path):
            os.makedirs(dist_plot_path)
            
        plt.figure(plot_graph.fig.number)
        plt.savefig(f'{dist_plot_path}qgan_ICLR-epoch-mnist-{epoch+1}-generated-distribution.png')
        
        # 10. Close the plot to free memory.
        plt.close(plot_graph.fig)
        print(f'Epoch {epoch+1} plot saved....')

    return plot_circ, plot_data

def plot_generated_images(
    train_iter: int,
    epoch: int,    
    backend,
    descaling_params:list,    
    tfrm,
    plot_frequency: int,
    circ,
    n_results:int,
    shots:int=20,
    n_range:int=16,
    generated_plot_path:str='./images/generated_plot/'
):
    '''
    QuGAN의 생성자를 사용하여 이미지를 생성하고, 이를 플롯하여 저장하는 함수입니다.
    주어진 플롯 빈도 조건에 따라 이미지를 생성하고 저장합니다.

    Args:
        train_iter (int) : QuGAN 훈련의 총 반복 횟수.
        epoch (int) : 현재 에포크 횟수.
        backend : 양자 컴퓨터 시뮬레이터 백엔드.
        descaling_params (list) : 역스케일링을 위한 파라미터 리스트.
        tfrm : PCA 역변환을 위한 객체.
        plot_frequency (int) : 이미지 저장 에포크 빈도.
        circ : 이미지 생성에 사용될 양자 회로.
        n_results (int) : 측정 큐비트 수.
        shots (int) : 회로 실행 시 샷 수.
        n_range (int) : 생성할 이미지의 개수.
        generated_plot_path (str) : 이미지 저장 경로.
    '''
    if (epoch % plot_frequency == 0) or (epoch == train_iter-1):
        generated_lst = []
       
        for _ in range(n_range):
            job = backend.run(circ, shots=shots)
            results = job.result().get_counts(circ)
            bins = [[0,0] for _ in range(n_results)]
            
            for key,value in results.items():
                for i in range(n_results):
                    if key[-i-1]== '1':
                        bins[i][0] += value
                    bins[i][1] += value
                    
            for i,pair in enumerate(bins):
                bins[i]= pair[0]/pair[1]
    
            generated_lst.append(bins)
        generated_lst = np.array(generated_lst)
        
        new_info = pf.descaling_points(generated_lst[:n_range], descaling_params, tfrm)
        new_info = new_info.reshape(new_info.shape[0],28,28)
        
        for i in range(new_info.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(new_info[i, :, :], cmap='gray')
            plt.axis('off')
        plt.suptitle(f"Epoch {epoch+1} Generated Images")
        
        if not os.path.exists(generated_plot_path):
            os.makedirs(generated_plot_path)
            
        plt.savefig(f"{generated_plot_path}/qgan_ICLR_-epoch-mnist-{epoch+1}-generated-images")
        plt.close()

def save_variables(
    epoch:int,
    today:str,
    train_var:dict,
    tracked_kl_div:list,
    tracked_g_loss:list,
    tracked_d_loss:list,
    save_path:str ='./results/'
):
    '''
    훈련 과정에서 추적된 손실 값, KL 발산 값, 학습 가능한 변수(파라미터) 등을 텍스트 파일로 저장합니다.
    이는 훈련 진행 상황을 기록하고 나중에 분석하기 위해 사용됩니다.

    Args:
        epoch (int): 현재 훈련 에포크 횟수. 파일 이름에 사용됩니다.
        today (str): 현재 날짜/시간 문자열 (파일 이름에 포함).
        train_var (Dict): 현재 학습 가능한 모델 변수(파라미터) 딕셔너리.
        tracked_kl_div (List[float]): 추적된 KL 발산 값들의 리스트.
        tracked_g_loss (List[float]): 추적된 생성자(Generator) 손실 값들의 리스트.
        tracked_d_loss (List[float]): 추적된 판별자(Discriminator) 손실 값들의 리스트.
        save_path (str): 결과를 저장할 디렉토리 경로. 기본값은 './results/'입니다.
    '''    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    with open(f'{save_path}qgan_results_epoch_{epoch+1}_{today}.txt', 'w') as file:
        file.write(f"Tracked KL Divergence\n{str(tracked_kl_div)}\n")
        file.write(f"Loss Of Generator\n{tracked_g_loss}\n")
        file.write(f"Loss Of Discriminator\n{tracked_d_loss}\n")
        file.write(f"Variable\n{str(train_var)}\n")
    print(f'TRAINED VARAIBLES SAVED.....')