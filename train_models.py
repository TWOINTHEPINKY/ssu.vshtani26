import re
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score, average_precision_score
from sklearn.model_selection import train_test_split

DATA_DIR = "vk_data"   
all_files = glob.glob(f"{DATA_DIR}/**/*.parquet", recursive=True)
print(f"Найдено файлов: {len(all_files)}")
if not all_files:
    raise FileNotFoundError(f"Не найдено .parquet файлов в папке {DATA_DIR}")
dfs = []
for file_path in all_files:
    print(f"Чтение: {file_path}")
    df_temp = pd.read_parquet(file_path)
    match = re.search(r'week[_]?(\d+)', file_path)
    if match:
        week_num = int(match.group(1))
        df_temp['week'] = week_num
        print(f"  -> Добавлена колонка 'week' = {week_num}")
    else:
        df_temp['week'] = 0
        print("  -> Не удалось извлечь неделю, установлено 0")
    dfs.append(df_temp)
df = pd.concat(dfs, ignore_index=True)
print(f"Уникальные недели в данных: {sorted(df['week'].unique())}")

WEIGHTS = {
    'like': 1.0,
    'share': 2.0,
    'bookmark': 1.5,
    'click_on_author': 0.5,
    'open_comments': 0.3
}
df['rating'] = 1.0
for action, weight in WEIGHTS.items():
    df['rating'] += df[action] * weight
max_watch_time = df['timespent'].max()
if max_watch_time > 0:
    df['watch_bonus'] = np.log1p(df['timespent'] / max_watch_time)
else:
    df['watch_bonus'] = 0
df['rating'] += df['watch_bonus']
print("\nСтатистика по рейтингу:")
print(df['rating'].describe())

if len(df['week'].unique()) > 1:
    weeks = sorted(df['week'].unique())
    train_weeks = weeks[:-1]
    test_week = weeks[-1]
    train_data = df[df['week'].isin(train_weeks)]
    test_data = df[df['week'] == test_week]
    print(f"\nВременной сплит: обучение на неделях {train_weeks}, тест на неделе {test_week}")
else:
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    print("\nВсего одна неделя в данных, поэтому я использую случайное разбиение 80/20")
print(f"Train размер: {len(train_data)}, Test размер: {len(test_data)}")

def precision_at_k(r, k):
    return sum(r[:k]) / k if k <= len(r) else 0
def recall_at_k(r, k):
    total = sum(r)
    return sum(r[:k]) / total if total > 0 else 0
def evaluate_model(model, test_df, k=10, max_users=None):
    users = test_df['user_id'].unique()
    if max_users is not None:
        users = users[:max_users]
    precisions, recalls, ndcgs, maps = [], [], [], []
    for user in users:
        true_items = set(test_df[test_df['user_id'] == user]['item_id'])
        if not true_items:
            continue
        # Теперь метод recommend может вернуть кортеж (список, оценки)
        result = model.recommend(user, n=k, with_scores=True)
        # Если модель не поддерживает with_scores (старая версия), result будет списком
        if isinstance(result, tuple):
            rec_items, rec_scores = result
        else:
            rec_items = result
            rec_scores = None
        r = [1 if item in true_items else 0 for item in rec_items]
        precisions.append(precision_at_k(r, k))
        total_rel = sum(r)
        recalls.append(sum(r[:k]) / total_rel if total_rel > 0 else 0)
        if any(r) and rec_scores is not None:
            ndcgs.append(ndcg_score([r], [rec_scores]))
            maps.append(average_precision_score(r, rec_scores))
        else:
            ndcgs.append(0)
            maps.append(0)
    return {f'Precision@{k}': np.mean(precisions),
            f'Recall@{k}': np.mean(recalls),
            f'NDCG@{k}': np.mean(ndcgs),
            f'MAP@{k}': np.mean(maps)}

class MostPopular:
    def __init__(self, item_popularity):
        self.sorted_items = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)
    def recommend(self, user_id, n=10, with_scores=False):
        top_items = [item for item, score in self.sorted_items[:n]]
        if with_scores:
            # фиктивные оценки – просто убывающая последовательность
            top_scores = [score for _, score in self.sorted_items[:n]]
            return top_items, top_scores
        return top_items
item_popularity = train_data.groupby('item_id')['rating'].sum().to_dict()
mp_model = MostPopular(item_popularity)
print("\nОценка MostPopular@10:")
mp_metrics = evaluate_model(mp_model, test_data, k=10)
for metric, value in mp_metrics.items():
    print(f"{metric}: {value:.4f}")


from surprise import Dataset, Reader, SVD, BaselineOnly
from surprise.model_selection import train_test_split as surprise_split

print("\n" + "="*50)
print("Добавляем SVD и ALS")

train_surprise = train_data[['user_id', 'item_id', 'rating']].copy()
test_surprise = test_data[['user_id', 'item_id', 'rating']].copy()
train_surprise['user_id'] = train_surprise['user_id'].astype(str)
train_surprise['item_id'] = train_surprise['item_id'].astype(str)
test_surprise['user_id'] = test_surprise['user_id'].astype(str)
test_surprise['item_id'] = test_surprise['item_id'].astype(str)
reader = Reader(rating_scale=(train_data['rating'].min(), train_data['rating'].max()))
data = Dataset.load_from_df(train_surprise[['user_id', 'item_id', 'rating']], reader)
trainset, valset = surprise_split(data, test_size=0.2, random_state=42)

all_items = pd.concat([train_data['item_id'], test_data['item_id']]).unique()
print(f"Всего уникальных видео: {len(all_items)}")
class SurpriseWrapperFast:
    def __init__(self, model, all_items, top_items=None):
        self.model = model
        self.all_items = all_items
        self.top_items = top_items if top_items is not None else all_items
    def recommend(self, user_id, n=10, with_scores=False):
        user_id = str(user_id)
        preds = []
        for item in self.top_items:
            pred = self.model.predict(user_id, str(item)).est
            preds.append((item, pred))
        preds.sort(key=lambda x: x[1], reverse=True)
        top_items = [item for item, _ in preds[:n]]
        if with_scores:
            top_scores = [score for _, score in preds[:n]]
            return top_items, top_scores
        return top_items
item_pop = train_data.groupby('item_id')['rating'].sum().sort_values(ascending=False)
top2000_items = item_pop.head(2000).index.tolist()
print(f"Для ускорения будем предсказывать только для топ-{len(top2000_items)} видео")

print("Обучение SVD.")
svd = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
svd.fit(trainset)
svd_wrapper = SurpriseWrapperFast(svd, all_items, top_items=top2000_items)
print("Оценка SVD@10 на подвыборке пользователей.")
svd_metrics = evaluate_model(svd_wrapper, test_data, k=10, max_users=500)
for metric, value in svd_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nОбучение ALS...")
als = BaselineOnly(bsl_options={'method': 'als', 'n_epochs': 10, 'reg_u': 12, 'reg_i': 5})
als.fit(trainset)
als_wrapper = SurpriseWrapperFast(als, all_items, top_items=top2000_items)
print("Оценка ALS@10 на подвыборке пользователей.")
als_metrics = evaluate_model(als_wrapper, test_data, k=10, max_users=500)
for metric, value in als_metrics.items():
    print(f"{metric}: {value:.4f}")


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

print("\n" + "="*50)
print("Добавляем NCF")

train_small = train_data.sample(frac=0.3, random_state=42)
print(f"Обучающих записей: {len(train_small)}")
user_enc = LabelEncoder()
item_enc = LabelEncoder()
train_small['user_idx'] = user_enc.fit_transform(train_small['user_id'])
train_small['item_idx'] = item_enc.fit_transform(train_small['item_id'])
num_users = len(user_enc.classes_)
num_items = len(item_enc.classes_)
print(f"Пользователей: {num_users}, видео: {num_items}")
class NCFDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user_idx'].values, dtype=torch.long)
        self.items = torch.tensor(df['item_idx'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)
    def __len__(self):
        return len(self.ratings)
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]
batch_size = 512
loader = DataLoader(NCFDataset(train_small), batch_size=batch_size, shuffle=True)

class FastNCF(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=16):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim*2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, user, item):
        u = self.user_emb(user)
        i = self.item_emb(item)
        out = self.fc(torch.cat([u, i], dim=1))
        return out.squeeze()
device = torch.device("cpu")
model = FastNCF(num_users, num_items)
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
print("Обучение NCF.")
for epoch in range(epochs):
    model.train()
    loss_sum = 0
    for users, items, ratings in loader:
        users, items, ratings = users.to(device), items.to(device), ratings.to(device)
        optimizer.zero_grad()
        preds = model(users, items)
        loss = criterion(preds, ratings)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    print(f"Эпоха {epoch+1}/{epochs}, Loss: {loss_sum/len(loader):.4f}")

model.eval()
with torch.no_grad():
    user_embs = model.user_emb.weight   
    item_embs = model.item_emb.weight   
top_items_idx = torch.tensor(train_small.groupby('item_idx')['rating'].sum().nlargest(1000).index.tolist())
print(f"Инференс для топ-{len(top_items_idx)} видео")

class MatrixNCFWrapper:
    def __init__(self, model, user_enc, item_enc, top_items_idx):
        self.model = model
        self.user_enc = user_enc
        self.item_enc = item_enc
        self.top_items_idx = top_items_idx
        self.device = next(model.parameters()).device
    def recommend(self, user_id, n=10, with_scores=False):
        try:
            u_idx = self.user_enc.transform([user_id])[0]
        except:
            # холодный пользователь – популярные видео
            cold_items = [self.item_enc.inverse_transform([idx.item()])[0] for idx in self.top_items_idx[:n]]
            if with_scores:
                return cold_items, [0.0]*n
            return cold_items
        u_tensor = torch.tensor([u_idx], device=self.device)
        i_tensor = self.top_items_idx.to(self.device)
        with torch.no_grad():
            users_batch = u_tensor.repeat(len(i_tensor))
            preds = self.model(users_batch, i_tensor).cpu().numpy()
        sorted_idx = preds.argsort()[::-1][:n]
        recs = [self.item_enc.inverse_transform([self.top_items_idx[i].item()])[0] for i in sorted_idx]
        if with_scores:
            scores = [preds[i] for i in sorted_idx]
            return recs, scores
        return recs
ncf_wrapper = MatrixNCFWrapper(model, user_enc, item_enc, top_items_idx)
print("Оценка NCF@10.")
ncf_metrics = evaluate_model(ncf_wrapper, test_data, k=10, max_users=200)
print(f"Precision@10: {ncf_metrics['Precision@10']:.4f}")
print(f"Recall@10: {ncf_metrics['Recall@10']:.4f}")
print(f"NDCG@10: {ncf_metrics['NDCG@10']:.4f}")
print(f"MAP@10: {ncf_metrics['MAP@10']:.4f}")