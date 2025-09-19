import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
import pickle
import os
from dataclasses import dataclass
from typing import Optional, Tuple
import matplotlib.pyplot as plt

@dataclass
class ModelConfig:
    """모델 설정 클래스"""
    # 데이터 설정
    vocab_size: int = 1000
    input_dim: int = 18         
    output_dim: int = 300       
    
    # GPT-2 모델 설정 (훨씬 더 크게)
    n_embd: int = 1024          # 768 → 1024 (GPT-2 Large 수준)
    n_layer: int = 24           # 12 → 24 (2배 더 깊게)
    n_head: int = 16            # 12 → 16 (더 많은 어텐션 헤드)
    n_positions: int = 1024     
    
    # 훈련 설정 (큰 모델에 맞게 조정)
    batch_size: int = 4         # 8 → 4 (GPU 메모리 고려)
    learning_rate: float = 1e-5 # 2e-5 → 1e-5 (더 안정적)
    num_epochs: int = 100       # 50 → 100 (더 오래)
    warmup_steps: int = 1000    # 500 → 1000 (점진적 워밍업)
    weight_decay: float = 0.01  
    
    # 기타 설정
    dropout: float = 0.15       # 0.1 → 0.15 (과적합 방지)
    use_cpu: bool = False       
    random_seed: int = 42

class MARSTimeSeriesDataset(Dataset):
    """MARS 데이터를 GPT-2용 토큰 시퀀스로 변환하는 데이터셋"""
    
    def __init__(self, X, y, config: ModelConfig):
        self.X = X
        self.y = y
        self.config = config
        self.vocab_size = config.vocab_size
        self.sep_token_id = config.vocab_size
        
        # 데이터 검증
        self._validate_data()
        
        # 토큰 변환
        self.X_tokens = self._float_to_tokens(X)
        self.y_tokens = self._float_to_tokens(y)
        
        # 시퀀스 생성
        self.sequences = self._create_sequences()
    
    def _validate_data(self):
        """데이터 유효성 검사"""
        print(f"X shape: {self.X.shape}, y shape: {self.y.shape}")
        print(f"X range: [{self.X.min():.4f}, {self.X.max():.4f}]")
        print(f"y range: [{self.y.min():.4f}, {self.y.max():.4f}]")
        
        # 0-1 범위 확인
        if self.X.min() < 0 or self.X.max() > 1:
            print("경고: X 데이터가 0-1 범위를 벗어남")
        if self.y.min() < 0 or self.y.max() > 1:
            print("경고: y 데이터가 0-1 범위를 벗어남")
    
    def _float_to_tokens(self, data):
        """0-1 float을 토큰 ID로 변환"""
        # 안전장치: 0-1 범위로 클리핑
        data_clipped = np.clip(data, 0.0, 1.0)
        
        # 토큰 변환
        tokens = (data_clipped * (self.vocab_size - 1)).astype(int)
        tokens = np.clip(tokens, 0, self.vocab_size - 1)
        
        return tokens
    
    def _create_sequences(self):
        """GPT-2 입력용 시퀀스 생성"""
        sequences = []
        max_seq_length = self.config.input_dim + 1 + self.config.output_dim
        
        if max_seq_length > self.config.n_positions:
            print(f"경고: 시퀀스 길이 {max_seq_length}가 최대 위치 {self.config.n_positions}를 초과")
        
        for i in range(len(self.X)):
            x_seq = self.X_tokens[i].tolist()
            sep_seq = [self.sep_token_id]
            y_seq = self.y_tokens[i].tolist()
            
            full_seq = x_seq + sep_seq + y_seq
            sequences.append(full_seq)
        
        print(f"시퀀스 생성 완료: {len(sequences)}개, 길이: {len(sequences[0])}")
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        input_ids = sequence[:-1]
        labels = sequence[1:]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

class MARSGPT2Model:
    """설정 가능한 MARS GPT-2 예측 모델"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self._set_device()
        self._set_seed()
        self._build_model()
        
        # 데이터셋 저장용
        self.train_dataset = None
        self.val_dataset = None
    
    def _set_device(self):
        """디바이스 설정"""
        if self.config.use_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"사용 디바이스: {self.device}")
    
    def _set_seed(self):
        """랜덤 시드 설정"""
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.random_seed)
    
    def _build_model(self):
        """GPT-2 모델 구성"""
        gpt2_config = GPT2Config(
            vocab_size=self.config.vocab_size + 1,  # SEP 토큰 포함
            n_positions=self.config.n_positions,
            n_embd=self.config.n_embd,
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
            resid_pdrop=self.config.dropout,
            embd_pdrop=self.config.dropout,
            attn_pdrop=self.config.dropout,
            bos_token_id=0,
            eos_token_id=0,
            pad_token_id=0
        )
        
        self.model = GPT2LMHeadModel(gpt2_config).to(self.device)
        
        # 모델 파라미터 수 계산
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"모델 파라미터: 총 {total_params:,}개, 훈련 가능 {trainable_params:,}개")
    
    def load_data(self, test_size: float = 0.2):
        """데이터 로드 및 전처리"""
        from datasetup import MARSDatasetPreprocessor
        preprocessor = MARSDatasetPreprocessor("./combined_dataset")
        X_raw, y_raw = preprocessor.load_raw_data()
        X_processed, y_processed = preprocessor.preprocess_data(
            X_method='minmax',      # X는 MinMax 정규화
            y_method='minmax',      # y도 MinMax 정규화
            handle_outliers=True,   # 이상치 처리
            outlier_method='clip'   # 클리핑 방식
        )

        # 데이터 다운 샘플링
        X = X_processed # (2999, 18)
        y = y_processed[:, ::10][:, :300] # (2999, 300)

        # 훈련/검증 분할 및 저장
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.config.random_seed
        )
        np.savez('./train_test_dataset/train_test_dataset.npz', X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)
        
        # 데이터셋 생성
        self.train_dataset = MARSTimeSeriesDataset(X_train, y_train, self.config)
        self.val_dataset = MARSTimeSeriesDataset(X_val, y_val, self.config)
        
        print(f"훈련 데이터: {len(self.train_dataset)}개")
        print(f"검증 데이터: {len(self.val_dataset)}개")
        
        return X_train, X_val, y_train, y_val
    
    def train(self, output_dir: str = './trained_network'):
        """모델 훈련"""
        if self.train_dataset is None:
            raise ValueError("데이터를 먼저 로드하세요 (load_data)")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=100,
            save_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
            dataloader_pin_memory=False,  # GPU 메모리 문제 방지
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
        )
        
        print("훈련 시작...")
        trainer.train()
        
        # 모델 저장
        trainer.save_model()
        
        # 설정도 함께 저장
        config_path = os.path.join(output_dir, 'model_config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(self.config, f)
        
        print(f"모델과 설정이 {output_dir}에 저장되었습니다.")
        
        return trainer

# 사용 예시
def main():
    """사용 예시"""
    # 모델 설정
    config = ModelConfig(
        vocab_size=1000,
        n_embd=256,      # 더 작은 모델로 테스트
        n_layer=4,
        n_head=4,
        batch_size=8,
        num_epochs=5,
        learning_rate=1e-4
    )
    
    # 모델 생성
    model = MARSGPT2Model(config)
    
    # 데이터 로드
    model.load_data()
    
    # 훈련
    model.train()

if __name__ == "__main__":
    main()