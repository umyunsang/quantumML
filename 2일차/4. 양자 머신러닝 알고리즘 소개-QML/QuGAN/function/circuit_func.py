import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
import seaborn as sns
import os
import sys

def save_n_plot_generated_iamges(images:list, epoch:int):
    '''
    생성된 이미지를 저장하고 시각화하는 함수입니다.

    params
        - images:list = 생성된 이미지 데이터 (NumPy 배열 형태)
        - epoch:int   = 현재 QuGAN 훈련의 에포크(epoch) 횟수
    '''
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(images[i, :, :], cmap='gray')
        plt.axis('off')
    plt.savefig(f"./images/qugan-epoch-mnist-{epoch}-generated-images")
    plt.show()

def rand_ang():
    '''
    양자 게이트의 회전 각도를 위한 0에서 파이(pi) 사이의 무작위 값을 생성하는 함수입니다.
    이는 주로 양자 회로의 초기 파라미터나 변동량을 설정할 때 사용됩니다.
    '''
    return np.random.rand() * np.pi

def single_qubit_unitary(circ_ident, qubit_index, values):
    '''
    단일 큐비트에 RY(Y축 회전) 게이트를 적용하여 중첩 상태를 만듭니다.
    이는 양자 회로에서 데이터 인코딩 또는 학습 가능한 레이어를 구성하는 데 사용됩니다.

    params
    - circ_ident: QuantumCircuit 객체. 게이트를 추가할 양자 회로입니다.
    - qubit_index: int. RY 게이트를 적용할 큐비트의 인덱스입니다.
    - values: list 또는 array. 회전 각도를 포함하는 리스트/배열입니다. (여기서는 values[0]만 사용)
    '''
    circ_ident.ry(values[0], qubit_index)

def dual_qubit_unitary(circ_ident, qubit_1, qubit_2, values):
    '''
    두 개의 큐비트(qubit_1, qubit_2)에 RYY 게이트를 적용합니다.
    RYY 게이트는 두 큐비트 사이에 양자 얽힘을 생성하고 학습 가능한 연결을 모델링하는 데 사용됩니다.

    params
    - circ_ident: QuantumCircuit 객체. 게이트를 추가할 양자 회로입니다.
    - qubit_1: int. 첫 번째 큐비트의 인덱스입니다.
    - qubit_2: int. 두 번째 큐비트의 인덱스입니다.
    - values: list 또는 array. RYY 회전 각도를 포함하는 리스트/배열입니다. (여기서는 values[0]만 사용)
    '''
    circ_ident.ryy(values[0], qubit_1, qubit_2)

def controlled_dual_qubit_unitary(circ_ident, control_qubit, act_qubit, values):
    '''
    제어 큐비트(control_qubit)의 상태에 따라 대상 큐비트(act_qubit)에 RY 게이트를 적용합니다 (CRy 게이트).
    이는 큐비트 간의 조건부 상호작용과 얽힘을 생성하여 양자 회로의 표현력을 높입니다.

    params
    - circ_ident: QuantumCircuit 객체. 게이트를 추가할 양자 회로입니다.
    - control_qubit: int. 제어 큐비트의 인덱스입니다.
    - act_qubit: int. 대상 큐비트의 인덱스입니다.
    - values: list 또는 array. CRy 회전 각도를 포함하는 리스트/배열입니다. (여기서는 values[0]만 사용)
    '''
    circ_ident.cry(values[0], control_qubit, act_qubit)

####################################

def traditional_learning_layer(circ_ident, num_qubits, values, style="Dual",
                               qubit_start=1, qubit_end=5):
    '''
    주어진 스타일(style)에 따라 양자 회로에 학습 가능한 유니터리(unitary) 레이어를 구성합니다.
    이 레이어는 QGAN에서 생성자(Generator) 또는 판별자(Discriminator)의 학습 가능한 부분으로 사용됩니다.

    params
    - circ_ident: QuantumCircuit 객체. 게이트를 추가할 양자 회로입니다.
    - num_qubits: int. 전체 양자 회로의 큐비트 총 개수입니다.
    - values: dict. 각 큐비트 또는 큐비트 쌍에 대한 학습 가능한 파라미터(회전 각도)를 담고 있는 딕셔너리입니다.
      키는 큐비트 인덱스(예: "1"), 또는 큐비트 쌍(예: "1,2", "1--2") 문자열입니다.
    - style: str. 레이어의 구조를 정의합니다 ("Dual", "Single", "Controlled-Dual").
    - qubit_start: int. 레이어를 적용할 큐비트 범위의 시작 인덱스입니다.
    - qubit_end: int. 레이어를 적용할 큐비트 범위의 끝 인덱스입니다 (이 인덱스는 포함되지 않음).
    '''
    if style == "Dual":
        for qub in np.arange(qubit_start, qubit_end):
            single_qubit_unitary(circ_ident, qub, values[str(qub)])

        for qub in np.arange(qubit_start, qubit_end-1):
            dual_qubit_unitary(circ_ident, qub, qub+1,
                               values[str(qub) + "," + str(qub+1)])

    elif style == "Single":
        for qub in np.arange(qubit_start, qubit_end):
            single_qubit_unitary(circ_ident, qub, values[str(qub)])

    elif style == "Controlled-Dual":
        for qub in np.arange(qubit_start, qubit_end):
            single_qubit_unitary(circ_ident, qub, values[str(qub)])

        for qub in np.arange(qubit_start, qubit_end-1):
            dual_qubit_unitary(circ_ident, qub, qub+1,
                               values[str(qub) + "," + str(qub+1)])

        for qub in np.arange(qubit_start, qubit_end-1):
            controlled_dual_qubit_unitary(circ_ident, qub, qub+1,
                                          values[str(qub) + "--" + str(qub+1)])

def data_loading_circuit(circ_ident, num_qubits, values, qubit_start=1, qubit_end=5):
    '''
    고전적인 데이터(values)를 양자 큐비트에 인코딩하는 회로를 구성합니다.
    각 큐비트에 Ry 게이트를 사용하여 데이터 값을 회전 각도로 매핑합니다.
    이는 QGAN의 판별자가 실제 데이터를 입력받을 때 사용될 수 있습니다.

    params
    - circ_ident: QuantumCircuit 객체. 게이트를 추가할 양자 회로입니다.
    - num_qubits: int. 전체 양자 회로의 큐비트 총 개수입니다.
    - values: list 또는 array. 큐비트에 인코딩할 고전적인 데이터 값들의 리스트/배열입니다.
    - qubit_start: int. 데이터를 로딩할 큐비트 범위의 시작 인덱스입니다.
    - qubit_end: int. 데이터를 로딩할 큐비트 범위의 끝 인덱스입니다 (이 인덱스는 포함되지 않음).
    '''
    k = 0
    for qub in np.arange(qubit_start, qubit_end):
        circ_ident.ry(values[k], qub)
        k += 1

def swap_test(circ_ident, num_qubits):
    '''
    스왑 테스트(Swap Test) 회로를 구현합니다. 스왑 테스트는 두 양자 상태의 유사도를 측정하는 데 사용됩니다.
    여기서는 주로 판별자(Discriminator)가 생성된 양자 상태와 실제 양자 상태 또는 기준 상태의 유사도를 측정하는 데 활용됩니다.
    결과적으로 보조 큐비트(ancilla qubit)의 측정 확률이 유사도를 나타냅니다.

    params
    - circ_ident: QuantumCircuit 객체. 게이트를 추가할 양자 회로입니다.
    - num_qubits: int. 전체 양자 회로의 큐비트 총 개수입니다. (보조 큐비트 포함)
    '''
    num_swap = num_qubits//2

    for i in range(num_swap):
        circ_ident.cswap(0, i+1, i+num_swap+1)

    circ_ident.h(0)
    circ_ident.measure(0, 0)

def init_random_variables(q, style):
    '''
    q개의 큐비트에 대한 학습 가능한 변수를 초기화합니다.
    
    - style 'Single': 각 큐비트에 대한 단일 변수만 생성합니다.
    - style 'Dual': 인접한 큐비트 간의 상호작용 변수를 추가로 생성합니다.
    - style 'Controlled-Dual': 'Dual' 스타일에 제어 상호작용 변수를 더 추가합니다.
    '''
    trainable_variables = {}

    for i in np.arange(1, q+1):
        trainable_variables[str(i)] = [rand_ang()]

        if style == "Single":
            pass
        elif style == "Dual":
            if i != q:
                trainable_variables[str(i)+","+str(i+1)]  = [rand_ang()]
        elif style == "Controlled-Dual":
            if i != q:
                trainable_variables[str(i)+","+str(i+1)]  = [rand_ang()]
                trainable_variables[str(i)+"--"+str(i+1)] = [rand_ang()]
    
    return trainable_variables     

def get_probabilities(backend, circ, counts=1000):
    '''
    양자 회로를 실행하여 측정 결과를 얻고, 이를 기반으로 보조 큐비트(ancilla qubit)의 측정 확률을 계산합니다.
    특히 스왑 테스트(Swap Test)의 결과인 0번 보조 큐비트의 측정 확률을 정규화하고 변환하여 반환합니다.

    params
    - backend: 시뮬레이터 또는 실제 양자 장치 백엔드 객체.
    - circ: 실행할 QuantumCircuit 객체.
    - counts: int. 회로를 실행할 샷(shot) 횟수. (측정 결과를 통계적으로 얻기 위함)
    '''
    job     = backend.run(circ, shots=counts)
    results = job.result().get_counts(circ)

    counts_0 = results.get('0', 0)
    counts_1 = results.get('1', 0)

    total_counts = counts_0 + counts_1

    if total_counts == 0:
        print(f"Warning: No counts for '0' or '1' were observed. Results: {results}")
        return 1 # Or some other default/error value
        
    prob = counts_0 / total_counts
    prob = (prob - 0.5)

    if prob <= 0.005:
        prob = 0.005
    else:
        prob = prob * 2

    return prob

def cost_function(p, yreal):
    '''
    판별자(Discriminator)의 손실 함수(Loss Function)입니다.
    이 함수는 판별자의 출력 확률(`p`)과 실제 레이블(`yreal`)을 비교하여 손실을 계산합니다.
    일반적인 이진 교차 엔트로피(Binary Cross-Entropy)와 유사한 형태를 가집니다.

    - `yreal`이 0일 때 (실제 데이터): 판별자가 '0'에 가까운 확률을 출력할수록 손실이 낮아집니다.
      즉, 실제 데이터를 실제(0)로 잘 분류하는지 평가합니다.
    - `yreal`이 1일 때 (가짜 데이터): 판별자가 '1'에 가까운 확률을 출력할수록 손실이 낮아집니다.
      즉, 가짜 데이터를 가짜(1)로 잘 분류하는지 평가합니다.

    params
    - p: float. 판별자가 출력한 확률 값 (0과 1 사이). 스왑 테스트의 결과 `prob` 값.
    - yreal: int. 실제 레이블 (0 또는 1). 0은 '실제(Real)', 1은 '가짜(Fake)'를 의미할 수 있습니다.
    '''
    assert yreal in [0, 1], "yreal must be 0 or 1"
    assert 0 <= p <= 1, "p must be between 0 and 1"
    e = sys.float_info.epsilon

    if yreal == 0:
        return -np.log(p)
    else:  # yreal == 1:
        return -np.log(1 -p +e)

def generator_cost_function(p):
    '''
    생성자(Generator)의 손실 함수입니다.
    생성자의 목표는 판별자를 속여서, 자신이 생성한 가짜 데이터가 판별자에 의해 '실제(0)'로 분류되도록 하는 것입니다.
    따라서 생성자는 판별자가 가짜 데이터에 대해 출력하는 확률 `p`를 최소화(혹은 `-log(p)`를 최대화)하려고 합니다.
    이것은 `cost_function`에서 `yreal=0`인 경우와 동일한 형태를 가집니다.

    params
    - p: float. 판별자가 가짜 데이터에 대해 출력한 확률 값 (0과 1 사이).
    '''
    assert 0 <= p <= 1, "p must be between 0 and 1"
    e = sys.float_info.epsilon

    return -np.log(p +e)


def update_weights(init_value, lr: float, grad: float):
    '''
    양자 회로의 학습 가능한 파라미터(가중치)를 업데이트하는 함수입니다.
    주어진 학습률(lr)과 기울기(grad)를 사용하여 파라미터를 조정합니다.
    특히, 회전 각도 파라미터의 주기성(예: 2*pi 주기)을 고려하여 값을 [0, 2*pi) 범위 내에 유지합니다.

    params
        - init_value: float. 현재 파라미터의 초기 값.
        - lr:float   = 모델 학습 비율 (learning_rate).
        - grad:float = 계산된 모델 학습 기울기 (gradient).
    '''
    tau = 2*np.pi
    while lr*grad > tau:
        lr /= 10
        print(f"Warning - Gradient taking steps that are very large. Drop learning rate to {lr}.")

    weight_update = lr*grad
    new_value = init_value
    print(f'Updating with a new value of {weight_update}')

    if new_value - weight_update > tau:
        new_value = (new_value - weight_update) - tau

    elif new_value - weight_update < 0:
        new_value = (new_value - weight_update) + tau

    else:
        new_value = (new_value - weight_update)

    return new_value


def save_variables(var_dict, epoch):
    '''
    모델의 학습 가능한 변수(파라미터) 딕셔너리를 텍스트 파일로 저장하는 함수입니다.
    훈련 진행 상황을 기록하고 나중에 모델 상태를 로드할 수 있도록 합니다.

    params
        - var_dict: dict. 저장할 학습 가능한 변수들을 포함하는 딕셔너리.
        - epoch: int. 현재 훈련 에포크 횟수. 파일 이름에 사용됩니다.
    '''
    variable_path = './variables'
    if not os.path.exists(variable_path):
        os.makedirs(variable_path)
    with open(f'{variable_path}/Epoch-{epoch}-Variables-numbers-9.txt', 'w') as file:
        file.write(str(var_dict))

class training_circuit:
    '''
    QGAN 훈련에 사용되는 양자 회로를 생성하고 관리하는 클래스입니다.
    판별자(Discriminator)의 학습 과정을 모듈화하여 가짜 데이터 훈련 회로와 실제 데이터 훈련 회로를 생성합니다.
    '''
    def __init__(self, q, c, par_shift, layer_style):
        '''
        training_circuit 클래스를 초기화합니다.

        params
        - q: int. 전체 큐비트의 개수.
        - c: int. 고전 비트의 개수 (측정 결과를 저장하는 데 사용).
        - par_shift: float. 파라미터 이동 규칙(Parameter Shift Rule)에 사용되는 변동량.
        - layer_style: str. 양자 학습 레이어의 스타일 ("Dual", "Single", "Controlled-Dual").
        '''        
        self.q = q
        self.c = c
        self.par_shift   = par_shift
        self.layer_style = layer_style

    def _apply_parameter_shift(self, trainable_variables, key, key_value, diff, fwd_diff, apply_shift):
        """
        파라미터 이동 규칙(Parameter Shift Rule)에 따라 학습 가능한 변수 값을 일시적으로 조정하거나 복원하는 헬퍼 함수입니다.
        기울기(gradient)를 계산하기 위해 사용됩니다.

        params
        - trainable_variables: dict. 학습 가능한 파라미터들을 담고 있는 딕셔너리.
        - key: str. 조정할 파라미터의 딕셔너리 키 (예: "1", "1,2").
        - key_value: int. 조정할 파라미터 리스트 내의 인덱스 (대부분 0).
        - diff: bool. 파라미터 이동을 적용할지 여부 (True일 때만 적용).
        - fwd_diff: bool. 정방향(True) 또는 역방향(False) 이동을 적용할지 여부.
        - apply_shift: bool. 현재 파라미터 이동을 적용하는 단계(True)인지, 아니면 복원하는 단계(False)인지.
        """
        if diff:
            if fwd_diff == apply_shift:
                trainable_variables[key][key_value] += self.par_shift
            else:
                trainable_variables[key][key_value] -= self.par_shift

    def disc_fake_train(self, trainable_variables, key, key_value,
                        diff=False, fwd_diff=False, Sample=False):
        '''
        판별자(Discriminator)가 "가짜(fake)" 데이터를 훈련하는 데 사용되는 양자 회로를 생성합니다.
        이 회로는 생성자의 현재 파라미터를 사용하여 가짜 데이터를 표현하고, 판별자가 이를 평가합니다.

        params
        - trainable_variables: dict. 학습 가능한 파라미터들을 담고 있는 딕셔너리 (생성자 및 판별자 파라미터 포함).
        - key, key_value: 파라미터 이동 규칙 적용 시 특정 파라미터를 식별하는 데 사용.
        - diff: bool. 파라미터 이동을 적용할지 여부.
        - fwd_diff: bool. 정방향 이동(True) 또는 역방향 이동(False)을 적용할지 여부.
        - Sample: bool. True이면 큐비트를 직접 측정하여 샘플링하고, False이면 스왑 테스트를 수행하여 유사도를 측정.
        '''
        q = self.q
        c = self.c
        layer_style = self.layer_style

        if Sample:
            z    = q//2
            circ = QuantumCircuit(q, z)
        else:
            circ = QuantumCircuit(q, c)

        circ.h(0)

        self._apply_parameter_shift(trainable_variables, key, key_value, diff, fwd_diff, True)

        traditional_learning_layer(circ, q, trainable_variables, style=layer_style,
                                   qubit_start=1, qubit_end=q//2 + 1)
        traditional_learning_layer(circ, q, trainable_variables, style=layer_style,
                                   qubit_start=q//2+1, qubit_end=q)

        if Sample:
            for qub in range(q//2):
                circ.measure(q//2 + 1 + qub, qub)
        else:
            swap_test(circ, q)

        self._apply_parameter_shift(trainable_variables, key, key_value, diff, fwd_diff, False)

        return circ

    def disc_real_train(self, training_variables, data, key, key_value, diff, fwd_diff):
        '''
        판별자(Discriminator)가 "실제(real)" 데이터를 훈련하는 데 사용되는 양자 회로를 생성합니다.
        이 회로는 실제 데이터를 양자 상태로 인코딩하고, 판별자가 이를 평가합니다.

        params
        - training_variables: dict. 학습 가능한 파라미터들을 담고 있는 딕셔너리 (판별자 파라미터 포함).
        - data: list 또는 array. 양자 회로에 로딩할 실제 고전 데이터.
        - key, key_value: 파라미터 이동 규칙 적용 시 특정 파라미터를 식별하는 데 사용.
        - diff: bool. 파라미터 이동을 적용할지 여부.
        - fwd_diff: bool. 정방향 이동(True) 또는 역방향 이동(False)을 적용할지 여부.
        '''        
        q = self.q
        c = self.c
        layer_style = self.layer_style

        circ = QuantumCircuit(q, c)
        circ.h(0)

        self._apply_parameter_shift(training_variables, key, key_value, diff, fwd_diff, True)

        traditional_learning_layer(circ, q, training_variables, style=layer_style,
                                   qubit_start=1, qubit_end=q//2 + 1)
        data_loading_circuit(circ, q, data, qubit_start=q//2 + 1, qubit_end=q)

        self._apply_parameter_shift(training_variables, key, key_value, diff, fwd_diff, False)

        swap_test(circ, q)
        return circ

class DistributionAnalyzer:
    '''
    데이터 분포를 분석하고, 두 분포 간의 유사도/차이(KL Divergence와 유사한 지표)를 측정하는 클래스입니다.
    양자 GAN에서 생성된 데이터와 실제 데이터 분포를 비교하는 데 사용됩니다.
    '''
    def __init__(self, num_bins=10):
        '''
        DistributionAnalyzer 클래스를 초기화합니다.

        params
        - num_bins (int): 데이터를 이산화(binning)할 때 사용할 빈(bin)의 개수. 기본값은 10입니다.
        '''
        if num_bins <= 0:
            raise ValueError('num_bins must be a postive integer.')

        self.num_bins = num_bins

    def bin_data(self, dataset):
        '''
        1차원 숫자형 데이터를 미리 정의된 빈(bin) 개수(num_bins)로 이산화하여
        정규화된 확률 분포를 생성합니다.
        함수는 소수점 이하 첫째 자리 숫자를 기준으로 데이터를 분류하는 독특한 방식을 사용합니다.
        데이터가 이 형식(예: 0.X, 1.Y 등)에 해당하지 않으면 예상과 다르게 동작할 수 있습니다.

        params
        - dataset: np.ndarray. 이산화할 1차원 numpy 배열 형태의 숫자형 데이터.

        returns
        - np.ndarray: 각 빈에 해당하는 데이터의 비율을 나타내는 정규화된 확률 분포 배열.
        '''
        if not isinstance(dataset, np.ndarray) or dataset.ndim != 1:
            raise ValueError("dataset must be a 1D numpy array")

        if len(dataset) == 0:
            return np.zeros(self.num_bins)

        bins = np.zeros(self.num_bins)

        for point in dataset:
            try:
                str_point = str(float(point))
                if '.' in str_point:
                    decimal_part = str_point.split('.')[-1]
                    if decimal_part and decimal_part.isdigit():
                        indx = int(decimal_part[0])
                        if 0 <= indx < self.num_bins:
                            bins[indx] += 1

            except ValueError:
                print(f'Warning: Skipping non-numeric data point : {point}')
                pass

        total_sum = np.sum(bins)
        if total_sum == 0:
            return np.zeros(self.num_bins)

        bins /= total_sum
        return bins

    def kl_divergence(self, p_dist: np.ndarray, q_dist: np.ndarray):
        '''
        두 분포 `p_dist`와 `q_dist` 사이의 유사도를 측정하는 지표를 계산합니다.
        이 함수가 계산하는 수식은 쿨백-라이블러 발산(Kullback-Leibler Divergence)의 표준 정의가 아니며,
        헬링거 거리(Hellinger Distance)와 관련된 변형된 형태입니다:
        `sqrt(sum((sqrt(p_i) - sqrt(q_i))^2)) / sqrt(2)`

        params
        - p_dist: np.ndarray. 기준이 되는 ("실제") 분포의 1차원 데이터 배열.
        - q_dist: np.ndarray. 비교 대상이 되는 ("생성된") 분포의 1차원 데이터 배열.

        returns
        - float: 계산된 발산(divergence) 값.
        '''
        p = self.bin_data(p_dist)
        q = self.bin_data(q_dist)

        kldiv_sum_of_squares = 0
        for p_point, q_point in zip(p, q):
            kldiv_sum_of_squares += (np.sqrt(p_point) - np.sqrt(q_point))**2

        kldiv = (1 / np.sqrt(2)) * np.sqrt(kldiv_sum_of_squares)
        return kldiv

    def generate_kl_divergence_hist(self, actual_data, epoch_results_data):
        '''
        주어진 실제 데이터와 에포크 결과 데이터의 각 차원(feature)에 대해
        헬링거 거리와 유사한 발산 값을 계산하여 리스트로 반환합니다.
        이 값은 생성된 데이터 분포가 실제 데이터 분포에 얼마나 가까운지 측정하는 지표로 사용됩니다.

        params
        - actual_data: np.ndarray. 실제(참) 데이터의 2차원 배열 (샘플 수, 차원 수).
        - epoch_results_data: np.ndarray. 특정 에포크에서 생성된(예측된) 데이터의 2차원 배열 (샘플 수, 차원 수).

        returns
        - List[float]: 각 차원별로 계산된 발산 값들의 리스트.
        '''
        plt.clf()  # Figure 정리
        sns.set_theme()

        kl_div_vec = []

        # Ensure both inputs are 2D arrays and have matching dimensions
        if actual_data.ndim != 2 or epoch_results_data.ndim != 2:
            raise ValueError("actual_data and epoch_results_data must be 2D arrays.")
        if actual_data.shape[1] != epoch_results_data.shape[1]:
            raise ValueError("actual_data and epoch_results_data must have the same number of columns (dimensions).")

        for kl_dim in range(actual_data.shape[1]):
            kl_div = self.kl_divergence(actual_data[:, kl_dim], 
                                        epoch_results_data[:, kl_dim])
            kl_div_vec.append(kl_div)

        return kl_div_vec







