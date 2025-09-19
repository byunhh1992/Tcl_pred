import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from dataclasses import dataclass
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

@dataclass
class LSTMConfig:
    """LSTM 모델 설정 클래스"""
    # 데이터 설정
    input_dim: int = 18         
    output_dim: int = 300       
    
    # LSTM 모델 설정
    hidden_dim: int = 512       # 은닉 차원
    num_layers: int = 4         # LSTM 레이어 수
    dropout: float = 0.2        # 드롭아웃 비율
    bidirectional: bool = True  # 양방향 LSTM
    
    # 훈련 설정
    batch_size: int = 32        
    learning_rate: float = 1e-3 
    num_epochs: int = 100       
    weight_decay: float = 1e-4  
    patience: int = 15          # Early stopping patience
    
    # 기타 설정
    use_cpu: bool = False       
    random_seed: int = 42
    
    def __post_init__(self):
        """설정 후처리"""
        if self.bidirectional:
            self.effective_hidden_dim = self.hidden_dim * 2
        else:
            self.effective_hidden_dim = self.hidden_dim

class MARSLSTMDataset(Dataset):
    """MARS 데이터를 LSTM용으로 변환하는 데이터셋"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)  # (N, 18)
        self.y = torch.FloatTensor(y)  # (N, 300)
        
        print(f"LSTM Dataset - X shape: {self.X.shape}, y shape: {self.y.shape}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'input': self.X[idx],      # (18,)
            'target': self.y[idx]      # (300,)
        }

class LSTMPredictor(nn.Module):
    """LSTM 기반 시계열 예측 모델"""
    
    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config
        
        # 입력 임베딩 레이어 (입력 차원을 확장)
        self.input_embedding = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # LSTM 레이어들
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=True
        )
        
        # 출력 레이어들
        self.output_layers = nn.Sequential(
            nn.Linear(config.effective_hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim)
        )
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
            elif 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param.data)
    
    def forward(self, x):
        """순전파"""
        batch_size = x.size(0)
        
        # 입력 임베딩
        x = self.input_embedding(x)  # (batch_size, hidden_dim)
        
        # LSTM을 위해 시퀀스 차원 추가 (단일 시점이므로 길이 1)
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # LSTM 통과
        lstm_out, _ = self.lstm(x)  # (batch_size, 1, effective_hidden_dim)
        
        # 마지막 시점의 출력만 사용
        lstm_out = lstm_out[:, -1, :]  # (batch_size, effective_hidden_dim)
        
        # 출력 레이어 통과
        output = self.output_layers(lstm_out)  # (batch_size, output_dim)
        
        return output

class MARSLSTMModel:
    """LSTM 예측 모델 래퍼 클래스"""
    
    def __init__(self, config: LSTMConfig):
        self.config = config
        self._set_device()
        self._set_seed()
        self._build_model()
        
        # 데이터셋 저장용
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        
        # 훈련 기록
        self.train_losses = []
        self.val_losses = []
    
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
        """LSTM 모델 구성"""
        self.model = LSTMPredictor(self.config).to(self.device)
        
        # 옵티마이저와 손실함수
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.criterion = nn.MSELoss()
        
        # 학습률 스케줄러
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5,
        )
        
        # 모델 파라미터 수 계산
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"LSTM 모델 파라미터: 총 {total_params:,}개, 훈련 가능 {trainable_params:,}개")
    
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

        # 데이터 다운 샘플링 (GPT-2와 동일)
        X = X_processed # (2999, 18)
        y = y_processed[:, ::10][:, :300] # (2999, 300)

        # 훈련/검증 분할
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.config.random_seed
        )
        
        # 데이터셋 생성
        self.train_dataset = MARSLSTMDataset(X_train, y_train)
        self.val_dataset = MARSLSTMDataset(X_val, y_val)
        
        # 데이터로더 생성
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        print(f"훈련 데이터: {len(self.train_dataset)}개")
        print(f"검증 데이터: {len(self.val_dataset)}개")
        
        return X_train, X_val, y_train, y_val
    
    def train_epoch(self):
        """한 에포크 훈련"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in self.train_loader:
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # 순전파
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 역전파
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate_epoch(self):
        """한 에포크 검증"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, output_dir: str = './trained_lstm_network'):
        """모델 훈련"""
        if self.train_loader is None:
            raise ValueError("데이터를 먼저 로드하세요 (load_data)")
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("LSTM 훈련 시작...")
        best_val_loss = float('inf')
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            # 훈련
            train_loss = self.train_epoch()
            
            # 검증
            val_loss = self.validate_epoch()
            
            # 기록
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 학습률 조정
            self.scheduler.step(val_loss)
            
            # 로그 출력
            if (epoch + 1) % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f"Epoch {epoch+1}/{self.config.num_epochs}")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val Loss: {val_loss:.6f}")
                print(f"  Elapsed: {elapsed_time:.2f}s")
                print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # 최고 모델 저장
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'config': self.config,
                    'epoch': epoch,
                    'val_loss': val_loss
                }, os.path.join(output_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                
                if patience_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 최고 모델 로드
        checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'), weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 설정 저장
        with open(os.path.join(output_dir, 'lstm_config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
        
        # 훈련 기록 저장
        self._save_training_plots(output_dir)
        
        print(f"LSTM 모델과 설정이 {output_dir}에 저장되었습니다.")
        print(f"최고 검증 손실: {best_val_loss:.6f}")
        
        return best_val_loss
    
    def _save_training_plots(self, output_dir):
        """훈련 과정 그래프 저장"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('LSTM Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('LSTM Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lstm_training_progress.png'), dpi=300)
        plt.close()
        
        # 손실 데이터도 저장
        np.savez(os.path.join(output_dir, 'training_history.npz'),
                 train_losses=self.train_losses,
                 val_losses=self.val_losses)
    
    def predict(self, X):
        """예측 수행"""
        self.model.eval()
        
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        
        if len(X.shape) == 1:
            X = X.unsqueeze(0)
        
        X = X.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X)
        
        return outputs.cpu().numpy()

def load_trained_lstm_model(model_path: str = './trained_lstm_network'):
    """훈련된 LSTM 모델 로드"""
    # 경로 확인
    config_path = os.path.join(model_path, 'lstm_config.pkl')
    model_file_path = os.path.join(model_path, 'best_model.pth')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_file_path}")
    
    # 설정 로드
    try:
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"설정 파일 로드 중 오류 발생: {e}")
    
    # 모델 생성
    model_wrapper = MARSLSTMModel(config)
    
    # 체크포인트 로드
    try:
        checkpoint = torch.load(model_file_path, 
                               map_location=model_wrapper.device, weights_only=False)
        model_wrapper.model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        raise RuntimeError(f"모델 파일 로드 중 오류 발생: {e}")
    
    print(f"LSTM 모델 로드 완료 (Epoch {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.6f})")
    
    return model_wrapper, config

# 사용 예시
def main():
    """사용 예시"""
    # 모델 설정
    config = LSTMConfig(
        hidden_dim=512,
        num_layers=4,
        batch_size=32,
        num_epochs=100,
        learning_rate=1e-3
    )
    
    # 모델 생성
    model = MARSLSTMModel(config)
    
    # 데이터 로드
    model.load_data()
    
    # 훈련
    model.train()

if __name__ == "__main__":
    main()