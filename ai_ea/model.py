"""
FX予測モデル v3 - 10種アーキテクチャ対応

arch 一覧:
  gru_attn   : GRU + Temporal Attention (baseline)
  bigru      : Bidirectional GRU + Attention
  lstm_attn  : LSTM + Temporal Attention
  cnn        : 1D Dilated CNN
  tcn        : Temporal Convolutional Network (causal dilated)
  cnn_gru    : 1D CNN → GRU (ハイブリッド)
  transformer: 軽量 Transformer Encoder
  mlp        : Pure MLP (最終バーのみ使用・高速)
  resnet     : Residual 1D CNN
  inception  : Inception (マルチスケール 1D Conv)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 共通モジュール
# ─────────────────────────────────────────────────────────────────────────────
class TemporalAttention(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.attn = nn.Linear(hidden, 1, bias=False)

    def forward(self, x: torch.Tensor):
        score  = self.attn(x).squeeze(-1)
        weight = torch.softmax(score, dim=-1)
        ctx    = (x * weight.unsqueeze(-1)).sum(dim=1)
        return ctx, weight


def _head(in_dim: int, hidden: int, n_classes: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Dropout(dropout),
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout * 0.5),
        nn.Linear(hidden, n_classes),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. GRU + Temporal Attention (baseline)
# ─────────────────────────────────────────────────────────────────────────────
class GRUAttn(nn.Module):
    """GRU + Temporal Attention。最終ステップとContext Vectorをconcatして分類"""
    def __init__(self, n_features, seq_len, hidden, layers, dropout, n_classes=3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_features, hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Dropout(dropout * 0.5),
        )
        self.gru  = nn.GRU(hidden, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.attn = TemporalAttention(hidden)
        self.head = _head(hidden * 2, hidden, n_classes, dropout)

    def forward(self, x):
        x       = self.proj(x)
        out, _  = self.gru(x)
        last    = out[:, -1, :]
        ctx, _  = self.attn(out)
        return self.head(torch.cat([last, ctx], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bidirectional GRU + Attention
# ─────────────────────────────────────────────────────────────────────────────
class BiGRU(nn.Module):
    """双方向GRU: 過去→未来・未来→過去の両方向から文脈を把握"""
    def __init__(self, n_features, seq_len, hidden, layers, dropout, n_classes=3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_features, hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Dropout(dropout * 0.5),
        )
        self.gru  = nn.GRU(hidden, hidden, layers, batch_first=True,
                           bidirectional=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.attn = TemporalAttention(hidden * 2)
        self.head = _head(hidden * 2, hidden, n_classes, dropout)

    def forward(self, x):
        x      = self.proj(x)
        out, _ = self.gru(x)           # [B, T, 2H]
        ctx, _ = self.attn(out)        # [B, 2H]
        return self.head(ctx)


# ─────────────────────────────────────────────────────────────────────────────
# 3. LSTM + Temporal Attention
# ─────────────────────────────────────────────────────────────────────────────
class LSTMAttn(nn.Module):
    """LSTMセル: 長期依存関係をCell Stateで保持"""
    def __init__(self, n_features, seq_len, hidden, layers, dropout, n_classes=3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_features, hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Dropout(dropout * 0.5),
        )
        self.lstm = nn.LSTM(hidden, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.attn = TemporalAttention(hidden)
        self.head = _head(hidden * 2, hidden, n_classes, dropout)

    def forward(self, x):
        x       = self.proj(x)
        out, _  = self.lstm(x)
        last    = out[:, -1, :]
        ctx, _  = self.attn(out)
        return self.head(torch.cat([last, ctx], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# 4. 1D Dilated CNN
# ─────────────────────────────────────────────────────────────────────────────
class CNNBlock(nn.Module):
    def __init__(self, ch, dilation, dropout):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, kernel_size=3, padding=dilation,
                              dilation=dilation)
        self.norm = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return F.gelu(self.drop(self.norm(self.conv(x)))) + x


class CNNNet(nn.Module):
    """Dilated 1D CNN: 受容野を指数的に拡張 (1→2→4→8)"""
    def __init__(self, n_features, seq_len, hidden, layers, dropout, n_classes=3):
        super().__init__()
        self.proj   = nn.Conv1d(n_features, hidden, 1)
        dilations   = [1, 2, 4, 8][:max(2, layers)]
        self.blocks = nn.Sequential(*[CNNBlock(hidden, d, dropout) for d in dilations])
        self.head   = _head(hidden, hidden, n_classes, dropout)

    def forward(self, x):
        x = self.proj(x.permute(0, 2, 1))   # [B, H, T]
        x = self.blocks(x)
        x = x.mean(dim=-1)                   # Global Average Pooling
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# 5. TCN (Temporal Convolutional Network - Causal Dilated)
# ─────────────────────────────────────────────────────────────────────────────
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_ch); self.norm2 = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.res   = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.pad   = pad

    def forward(self, x):
        out = F.gelu(self.norm1(self.conv1(x)[..., :-self.pad]))
        out = self.drop(F.gelu(self.norm2(self.conv2(out)[..., :-self.pad])))
        return out + self.res(x)


class TCNNet(nn.Module):
    """因果的拡張畳み込み: 未来情報を使わず長距離依存を学習"""
    def __init__(self, n_features, seq_len, hidden, layers, dropout, n_classes=3):
        super().__init__()
        n_blocks = max(2, layers)
        blocks   = []
        in_ch    = n_features
        for i in range(n_blocks):
            blocks.append(TCNBlock(in_ch, hidden, kernel=3,
                                   dilation=2**i, dropout=dropout))
            in_ch = hidden
        self.net  = nn.Sequential(*blocks)
        self.head = _head(hidden, hidden, n_classes, dropout)

    def forward(self, x):
        x = self.net(x.permute(0, 2, 1))  # [B, H, T]
        x = x[:, :, -1]                    # 最後のタイムステップ
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# 6. CNN → GRU ハイブリッド
# ─────────────────────────────────────────────────────────────────────────────
class CNNGRUNet(nn.Module):
    """1D CNN で局所特徴を抽出 → GRU で系列を統合"""
    def __init__(self, n_features, seq_len, hidden, layers, dropout, n_classes=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, hidden, 3, padding=1),
            nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout * 0.3),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout * 0.3),
        )
        self.gru  = nn.GRU(hidden, hidden, 1, batch_first=True)
        self.attn = TemporalAttention(hidden)
        self.head = _head(hidden * 2, hidden, n_classes, dropout)

    def forward(self, x):
        x      = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, T, H]
        out, _ = self.gru(x)
        last   = out[:, -1, :]
        ctx, _ = self.attn(out)
        return self.head(torch.cat([last, ctx], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# 7. Lightweight Transformer Encoder
# ─────────────────────────────────────────────────────────────────────────────
class TransformerNet(nn.Module):
    """Self-Attention: 全タイムステップ間の関係を直接学習"""
    def __init__(self, n_features, seq_len, hidden, layers, dropout, n_classes=3):
        super().__init__()
        # nhead は hidden の約数で 4 以下
        nhead   = 4 if hidden >= 32 else (2 if hidden >= 16 else 1)
        d_model = (hidden // nhead) * nhead  # 整合性確保
        self.proj = nn.Linear(n_features, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, layers))
        self.head    = _head(d_model, d_model, n_classes, dropout)

    def forward(self, x):
        x = self.proj(x)
        x = self.encoder(x)
        x = x[:, -1, :]    # CLSトークンとして最終ステップを使用
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Pure MLP (最終バーのみ・最速)
# ─────────────────────────────────────────────────────────────────────────────
class MLPNet(nn.Module):
    """最終バーの特徴量のみ使用。テクニカル指標の直接解釈に最適"""
    def __init__(self, n_features, seq_len, hidden, layers, dropout, n_classes=3):
        super().__init__()
        n_layers = max(2, layers + 1)
        mods     = [nn.Linear(n_features, hidden), nn.LayerNorm(hidden),
                    nn.GELU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            mods += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
                     nn.GELU(), nn.Dropout(dropout * 0.5)]
        self.net  = nn.Sequential(*mods)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = x[:, -1, :]    # 最終バーのみ使用
        return self.head(self.net(x))


# ─────────────────────────────────────────────────────────────────────────────
# 9. Residual 1D CNN
# ─────────────────────────────────────────────────────────────────────────────
class ResBlock1D(nn.Module):
    def __init__(self, ch, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch),
        )

    def forward(self, x):
        return F.gelu(self.block(x) + x)


class ResNet1D(nn.Module):
    """残差接続 1D CNN: 勾配の安定化 + 深いネットワーク"""
    def __init__(self, n_features, seq_len, hidden, layers, dropout, n_classes=3):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv1d(n_features, hidden, 3, padding=1),
            nn.BatchNorm1d(hidden), nn.GELU(),
        )
        n_blocks    = max(2, layers + 1)
        self.blocks = nn.Sequential(*[ResBlock1D(hidden, dropout) for _ in range(n_blocks)])
        self.head   = _head(hidden, hidden, n_classes, dropout)

    def forward(self, x):
        x = self.stem(x.permute(0, 2, 1))
        x = self.blocks(x)
        x = x.mean(dim=-1)    # GAP
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Inception 1D (マルチスケール畳み込み)
# ─────────────────────────────────────────────────────────────────────────────
class InceptionBlock1D(nn.Module):
    """3種のカーネルサイズ (1, 3, 5) で並列特徴抽出"""
    def __init__(self, in_ch, out_ch, dropout):
        super().__init__()
        h = out_ch // 3
        r = out_ch - h * 2
        self.b1 = nn.Sequential(nn.Conv1d(in_ch, h,  1),              nn.BatchNorm1d(h),  nn.GELU())
        self.b3 = nn.Sequential(nn.Conv1d(in_ch, h,  3, padding=1),   nn.BatchNorm1d(h),  nn.GELU())
        self.b5 = nn.Sequential(nn.Conv1d(in_ch, r,  5, padding=2),   nn.BatchNorm1d(r),  nn.GELU())
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.drop(torch.cat([self.b1(x), self.b3(x), self.b5(x)], dim=1))
        return F.gelu(out + self.res(x))


class InceptionNet(nn.Module):
    """マルチスケール: 短期(1本)・中期(3本)・長期(5本)のパターンを同時学習"""
    def __init__(self, n_features, seq_len, hidden, layers, dropout, n_classes=3):
        super().__init__()
        self.stem   = nn.Conv1d(n_features, hidden, 1)
        n_blocks    = max(2, layers)
        self.blocks = nn.Sequential(
            *[InceptionBlock1D(hidden, hidden, dropout) for _ in range(n_blocks)]
        )
        self.head   = _head(hidden, hidden, n_classes, dropout)

    def forward(self, x):
        x = self.blocks(self.stem(x.permute(0, 2, 1)))
        x = x.mean(dim=-1)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# ファクトリ関数
# ─────────────────────────────────────────────────────────────────────────────
ARCH_MAP = {
    'gru_attn':   GRUAttn,
    'bigru':      BiGRU,
    'lstm_attn':  LSTMAttn,
    'cnn':        CNNNet,
    'tcn':        TCNNet,
    'cnn_gru':    CNNGRUNet,
    'transformer':TransformerNet,
    'mlp':        MLPNet,
    'resnet':     ResNet1D,
    'inception':  InceptionNet,
}


def build_model(arch: str, n_features: int, seq_len: int,
                hidden: int, layers: int, dropout: float,
                n_classes: int = 3) -> nn.Module:
    cls = ARCH_MAP.get(arch)
    if cls is None:
        raise ValueError(f"未知のアーキテクチャ: {arch}  選択肢: {list(ARCH_MAP)}")
    return cls(n_features, seq_len, hidden, layers, dropout, n_classes)


# ─────────────────────────────────────────────────────────────────────────────
# 正規化ラッパー (ONNX エクスポート用)
# ─────────────────────────────────────────────────────────────────────────────
class FXPredictorWithNorm(nn.Module):
    """
    生特徴量 → z-score正規化 → モデル推論 → Softmax確率
    Input : [1, seq_len, n_features]  float32
    Output: [1, 3]                    P(HOLD), P(BUY), P(SELL)
    """
    def __init__(self, predictor: nn.Module,
                 mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.predictor = predictor
        self.register_buffer('feat_mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('feat_std',  torch.tensor(std,  dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.feat_mean) / (self.feat_std + 1e-8)
        x = torch.clamp(x, -5.0, 5.0)
        return torch.softmax(self.predictor(x), dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# ONNX エクスポート / 検証
# ─────────────────────────────────────────────────────────────────────────────
def export_onnx(model: FXPredictorWithNorm,
                seq_len: int, n_features: int,
                out_path: str, opset: int = 14) -> None:
    model.eval().cpu()
    # batch=1 固定: MQL5 は常に batch=1 で推論するため dynamic batch 不要
    # dynamic_axes に batch を含めると GRU/LSTM で UserWarning が出るため除去
    dummy = torch.zeros(1, seq_len, n_features, dtype=torch.float32)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning,
                                module='torch.onnx')
        torch.onnx.export(
            model, dummy, out_path,
            export_params=True, opset_version=opset,
            do_constant_folding=True,
            input_names=['input'], output_names=['output'],
            dynamic_axes={},   # batch=1 固定 (GRU警告を回避)
        )
    print(f"ONNX exported → {out_path}")


def verify_onnx(onnx_path: str, seq_len: int, n_features: int) -> None:
    import onnx, onnxruntime as ort
    onnx.checker.check_model(onnx.load(onnx_path))
    sess  = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    dummy = np.zeros((1, seq_len, n_features), dtype=np.float32)
    out   = sess.run(None, {'input': dummy})
    print(f"ONNX verify OK  output={np.round(out[0], 4)}")


# ─────────────────────────────────────────────────────────────────────────────
# 後方互換エイリアス
# ─────────────────────────────────────────────────────────────────────────────
FXPredictor = GRUAttn
