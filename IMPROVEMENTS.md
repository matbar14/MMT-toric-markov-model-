# Рекомендации по улучшению Toric Markov Trading Model

## ✅ Уже исправлено

### 1. Архитектура модели - сохранение фазовой информации
**Статус**: ИСПРАВЛЕНО
- Добавлены magnitude и phase к real/imag частям комплексных чисел
- Добавлен `complex_feature_fusion` слой для проекции 5*dim_angles → 2*dim_angles

---

## 🔴 Критические проблемы

### 2. Переобучение модели (Train acc=62%, Val acc=85%)
**Проблема**: Модель переобучается - валидационная точность выше обучающей
**Причина**: Дисбаланс классов + недостаточная регуляризация

**Решение**:
```python
# В train_trading_v3_basis.py добавить:
parser.add_argument("--dropout", type=float, default=0.2)  # Увеличить с 0.1
parser.add_argument("--weight-decay", type=float, default=5e-5)  # Увеличить с 1e-5
parser.add_argument("--label-smoothing", type=float, default=0.1)  # Для pattern_loss

# В модели trading_model_v3.py:
# Увеличить dropout во всех головах с 0.1 до 0.2-0.3
```

### 3. Очень низкая Precision (4.73% на val)
**Проблема**: Модель генерирует слишком много ложных сигналов
**Причина**: 
- Слишком низкий порог уверенности (0.45)
- Паттерны слишком редкие (label starvation)

**Решение**:
```python
# 1. Адаптивные пороги для разных паттернов
def get_pattern_specific_thresholds(pattern_counts):
    """Разные пороги для частых и редких паттернов"""
    thresholds = torch.ones(17) * 0.5
    # Для редких паттернов (< 100 примеров) - выше порог
    rare_mask = pattern_counts < 100
    thresholds[rare_mask] = 0.7
    return thresholds

# 2. Улучшить детекцию паттернов в trading_dataset_v3.py
# Сделать условия менее строгими:
long_move_threshold = df['spot_close'] * rolling_vol.mul(1.0).clip(lower=0.001, upper=0.02)
# Было: mul(1.5).clip(lower=0.002, upper=0.03)
```

### 4. Проблема с OI данными
**Проблема**: OI данные доступны только на части истории
**Текущее**: Forward fill создает ложные сигналы

**Решение**:
```python
# В trading_dataset_v3.py улучшить обработку OI:
def _add_open_interest_features(self, df: pd.DataFrame) -> pd.DataFrame:
    # Добавить маску валидности для каждого производного признака
    oi_available = df['oi_available'] > 0.5
    
    # Вместо ffill использовать 0 для отсутствующих данных
    df['open_interest'] = df['open_interest'].where(oi_available, 0.0)
    
    # Добавить признак "сколько времени прошло с последнего OI"
    df['oi_staleness'] = (~oi_available).cumsum()
    df.loc[oi_available, 'oi_staleness'] = 0
    
    return df
```

---

## 🟡 Важные улучшения

### 5. Улучшить loss функцию
**Проблема**: Focal loss с gamma=0 не работает, BCE не справляется с дисбалансом

**Решение**:
```python
# Использовать Asymmetric Loss для multi-label
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Asymmetric focusing
        probs_pos = probs.clamp(min=self.clip)
        probs_neg = (1 - probs).clamp(min=self.clip)
        
        loss_pos = targets * torch.log(probs_pos) * (1 - probs_pos).pow(self.gamma_pos)
        loss_neg = (1 - targets) * torch.log(probs_neg) * probs.pow(self.gamma_neg)
        
        return -(loss_pos + loss_neg).mean()
```

### 6. Добавить Temporal Consistency Loss
**Проблема**: Модель предсказывает независимо на каждом шаге, игнорируя временную согласованность

**Решение**:
```python
def temporal_consistency_loss(predictions_t, predictions_t_minus_1, alpha=0.1):
    """Паттерны не должны резко меняться между соседними барами"""
    return alpha * F.mse_loss(predictions_t, predictions_t_minus_1.detach())
```

### 7. Улучшить детекцию паттернов
**Проблема**: Слишком строгие условия → мало паттернов → переобучение

**Решение в trading_dataset_v3.py**:
```python
# 1. Использовать скользящее окно для валидации паттернов
def validate_pattern_with_window(self, pattern_mask, future_returns, window=3):
    """Паттерн валиден если прибыль в любом из следующих N баров"""
    valid = torch.zeros_like(pattern_mask)
    for i in range(1, window + 1):
        future_ret = future_returns.shift(-i)
        valid |= (pattern_mask & (future_ret > self.min_pattern_profit))
    return valid

# 2. Добавить "слабые" паттерны с меньшим порогом прибыли
df['pattern_bullish_div_weak'] = (
    (price_change_long < -long_move_threshold * 0.5) &
    (spot_cvd_change_long > 0) &
    (price_change_short > 0) &
    (future_return > self.min_pattern_profit * 0.5)
)
```

### 8. Добавить Ensemble предсказаний
**Решение**:
```python
# В backtest добавить усреднение по нескольким последним барам
def ensemble_predict(model, features_history, weights=[0.5, 0.3, 0.2]):
    """Усреднить предсказания по последним N барам"""
    predictions = []
    for i, w in enumerate(weights):
        feat = features_history[-len(weights) + i]
        pred = model.detect_patterns(feat)
        predictions.append({k: v * w for k, v in pred.items()})
    
    # Weighted average
    ensemble = {}
    for key in predictions[0].keys():
        ensemble[key] = sum(p[key] for p in predictions)
    return ensemble
```

---

## 🟢 Оптимизации производительности

### 9. Кэширование эмбеддингов
```python
# В ContinuousFeatureEmbedding добавить кэш
class ContinuousFeatureEmbedding(nn.Module):
    def __init__(self, num_features, embedding_dim, n_bits=8, use_cache=True):
        super().__init__()
        self.use_cache = use_cache
        self.cache = {}
        
    def forward(self, features):
        if self.use_cache and not self.training:
            key = hash(features.data_ptr())
            if key in self.cache:
                return self.cache[key]
        
        result = self._compute_embedding(features)
        
        if self.use_cache and not self.training:
            self.cache[key] = result
        return result
```

### 10. Mixed Precision Training
```python
# В train_trading_v3_basis.py
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# В train_epoch:
with autocast():
    outputs = model(features)
    loss = compute_loss(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 📊 Улучшения метрик и мониторинга

### 11. Добавить per-pattern метрики
```python
def compute_per_pattern_metrics(predictions, targets, pattern_names):
    """Метрики для каждого паттерна отдельно"""
    metrics = {}
    for i, name in enumerate(pattern_names):
        pred_i = predictions[:, i]
        target_i = targets[:, i]
        
        metrics[f'{name}_precision'] = precision_score(target_i, pred_i)
        metrics[f'{name}_recall'] = recall_score(target_i, pred_i)
        metrics[f'{name}_f1'] = f1_score(target_i, pred_i)
    return metrics
```

### 12. Добавить визуализацию паттернов
```python
def plot_pattern_distribution(dataset, save_path='pattern_dist.png'):
    """Визуализировать распределение паттернов по времени"""
    import matplotlib.pyplot as plt
    
    pattern_counts = dataset.patterns.sum(axis=0)
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(pattern_counts)), pattern_counts)
    plt.xticks(range(len(pattern_counts)), 
               TradingBacktest.PATTERN_NAMES, rotation=45)
    plt.ylabel('Count')
    plt.title('Pattern Distribution')
    plt.tight_layout()
    plt.savefig(save_path)
```

---

## 🎯 Приоритеты внедрения

### Немедленно (критично для работы):
1. ✅ Фазовая информация (уже сделано)
2. 🔴 Увеличить dropout и weight_decay (переобучение)
3. 🔴 Адаптивные пороги для паттернов (низкая precision)
4. 🔴 Улучшить обработку OI данных

### В ближайшее время:
5. 🟡 Asymmetric Loss вместо BCE
6. 🟡 Temporal Consistency Loss
7. 🟡 Ослабить условия детекции паттернов

### Когда будет время:
8. 🟢 Ensemble предсказаний
9. 🟢 Mixed Precision Training
10. 🟢 Per-pattern метрики
11. 🟢 Визуализация

---

## 📝 Дополнительные рекомендации

### Гиперпараметры для экспериментов:
```bash
# Попробовать более агрессивную регуляризацию
python train_trading_v3_basis.py \
  --dropout 0.3 \
  --weight-decay 1e-4 \
  --focal-gamma 2.0 \
  --confidence-label-smoothing 0.1 \
  --min-pattern-profit 0.002 \
  --pattern-threshold 0.35

# Попробовать другую архитектуру
python train_trading_v3_basis.py \
  --num-layers 3 \
  --dim-angles 96 \
  --num-states 192
```

### Улучшение датасета:
1. Добавить больше исторических данных (сейчас только 1-3 года)
2. Использовать данные с нескольких бирж для робастности
3. Добавить макроэкономические индикаторы (funding rate, liquidations)
4. Аугментация данных: добавить шум, масштабирование

### Backtesting:
1. Добавить slippage моделирование
2. Учитывать market impact для больших позиций
3. Добавить реалистичное исполнение ордеров (не по open price)
4. Walk-forward оптимизация вместо одного train/val split
