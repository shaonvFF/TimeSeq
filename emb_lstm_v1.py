import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from rhlstm.tree_model import bi_train
from rhlstm.eval import *
import numpy as np
import jieba


torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def emergency_memory_cleanup():
    """紧急显存清理"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清空缓存
        torch.cuda.synchronize()  # 同步
    gc.collect()  # 垃圾回收
    print("紧急显存清理完成")


def check_gpu_memory():
    """检查GPU显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"已分配显存: {allocated:.2f} GB")
        print(f"缓存显存: {cached:.2f} GB")
        # print(cached)
    else:
        print("CUDA不可用")


def check_gpu_memory_basic():
    """基础GPU显存检查"""
    if torch.cuda.is_available():
        # 获取GPU数量
        gpu_count = torch.cuda.device_count()
        print(f"可用GPU数量: {gpu_count}")
        
        for i in range(gpu_count):
            # 获取显存总量和已使用量
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            allocated_memory = torch.cuda.memory_allocated(i) / 1024**3  # GB
            cached_memory = torch.cuda.memory_reserved(i) / 1024**3  # GB
            free_memory = total_memory - allocated_memory - cached_memory
            
            print(f"GPU {i} ({torch.cuda.get_device_name(i)}):")
            print(f"  总显存: {total_memory:.2f} GB")
            print(f"  已分配: {allocated_memory:.2f} GB")
            print(f"  缓存: {cached_memory:.2f} GB")
            print(f"  可用: {free_memory:.2f} GB")
    else:
        print("CUDA不可用")


def split(s):
    for i in ['€',]:
        s = str(s).replace(i, '')
    for i in ['省', '市', '自治区', '小区', '区', '自治州', '州', '镇', '县', '乡', '村', '庄', '组', '弄', '路', '街道', '街']:
        s = str(s).replace(i, '|')
    res = s.split('|')
    return [i for i in res if i != '']


def get_index_fill(x, dic, split_type, max_len):
    if split_type == ',':
        l = x.split(',')
    elif split_type == 'jieba':
        l = jieba.lcut(x, cut_all=False)
    elif split_type == '':
        l = list(x)
    else:
        l = list(x)

    l_extend = pad_sequence(l, max_len)
    return torch.tensor([dic[i] if i in dic.keys() else dic['else'] for i in l_extend])


def get_index_list(x, dic, split_type):
    if split_type == ',':
        l = x.split(',')
    elif split_type == 'jieba':
        l = jieba.lcut(x, cut_all=False)
    elif split_type == '':
        l = list(x)
    else:
        l = list(x)
    if x == '':
        l.append('<PAD>')

    return torch.tensor([dic[i] if i in dic.keys() else dic['else'] for i in l])


def get_index_value(x, dic,):
    return torch.tensor(dic[x] if x in dic.keys() else dic['else'])


def pad_sequence(words, max_len, pad='<PAD>'):
    if not isinstance(words, list):
        words = list(words)
    if len(words) < max_len:
        words.extend([pad] * (max_len - len(words)))
        return words
    else:
        return words[:max_len]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_list_str(s):
    s.replace('深圳市乐信融资担保有限公司', 'lxrd')
    s.replace('X4403010001196', 'lxrd')
    lst = s.split(',')
    return [i.replace("'", '') if i != "''" else '' for i in lst]

def get_list_float(s):
    lst = s.split(',')
    return [float(i) if i != "''" else np.nan for i in lst]

def preprocess(x):
    (lst_query_org, lst_query_org_type, lst_query_reason, lst_day_diff_cx, lst_sin_month, lst_cos_month_cx,
     lst_sin_week_cx, lst_cos_week_cx, lst_sin_day_cx, lst_cos_day_cx,
     lst_forg_code, lst_forg_type, lst_frepay_type, lst_frepay_freq, lst_floan_type, lst_floan_amount,
     lst_fterms, lst_facc_type, lst_fcreditline, lst_fcurrency_type, lst_day_diff_loan, lst_sin_month_loan,
     lst_cos_month_loan, lst_sin_week_loan, lst_cos_week_loan, lst_sin_day_loan, lst_cos_day_loan,
     business_status, gender, education, degree, age, nationality, living_status,
     career_status, enterprise_attribute, business, occupation, job, title, marriage_status,
     postal_address, permanent_residence_address, living_address, enterprise, enterprise_address) = x

    lst_query_org = get_list_str(lst_query_org)
    lst_query_org_type = get_list_str(lst_query_org_type)
    lst_query_reason = get_list_str(lst_query_reason)
    lst_forg_code = get_list_str(lst_forg_code)
    lst_forg_type = get_list_str(lst_forg_type)
    lst_frepay_type = get_list_str(lst_frepay_type)
    lst_frepay_freq = get_list_str(lst_frepay_freq)
    lst_floan_type = get_list_str(lst_floan_type)
    lst_facc_type = get_list_str(lst_facc_type)
    lst_fcurrency_type = get_list_str(lst_fcurrency_type)

    lst_day_diff_cx = get_list_float(lst_day_diff_cx)
    lst_sin_month = get_list_float(lst_sin_month)
    lst_cos_month_cx = get_list_float(lst_cos_month_cx)
    lst_sin_week_cx = get_list_float(lst_sin_week_cx)
    lst_cos_week_cx = get_list_float(lst_cos_week_cx)
    lst_sin_day_cx = get_list_float(lst_sin_day_cx)
    lst_cos_day_cx = get_list_float(lst_cos_day_cx)
    lst_floan_amount = get_list_float(lst_floan_amount)
    lst_fterms = get_list_float(lst_fterms)
    lst_fcreditline = get_list_float(lst_fcreditline)
    
    lst_day_diff_loan = get_list_float(lst_day_diff_loan)
    lst_sin_month_loan = get_list_float(lst_sin_month_loan)
    lst_cos_month_loan = get_list_float(lst_cos_month_loan)
    lst_sin_week_loan = get_list_float(lst_sin_week_loan)
    lst_cos_week_loan = get_list_float(lst_cos_week_loan)
    lst_sin_day_loan = get_list_float(lst_sin_day_loan)
    lst_cos_day_loan = get_list_float(lst_cos_day_loan)

    return (lst_query_org, lst_query_org_type, lst_query_reason, lst_day_diff_cx, lst_sin_month, lst_cos_month_cx,
    lst_sin_week_cx, lst_cos_week_cx, lst_sin_day_cx, lst_cos_day_cx,
    lst_forg_code, lst_forg_type, lst_frepay_type, lst_frepay_freq, lst_floan_type, lst_floan_amount,
    lst_fterms, lst_facc_type, lst_fcreditline, lst_fcurrency_type, lst_day_diff_loan, lst_sin_month_loan,
    lst_cos_month_loan, lst_sin_week_loan, lst_cos_week_loan, lst_sin_day_loan, lst_cos_day_loan,
    business_status, gender, education, degree, age, nationality, living_status,
    career_status, enterprise_attribute, business, occupation, job, title, marriage_status,
    postal_address, permanent_residence_address, living_address, enterprise, enterprise_address)

class EmbeddingLayerNorm(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0,
                 scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None):
        super(EmbeddingLayerNorm, self).__init__()
        self.Embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm, 
                                      norm_type=norm_type,
                                      scale_grad_by_freq=scale_grad_by_freq, sparse=sparse, _weight=_weight, 
                                      _freeze=_freeze, device=device,
                                      dtype=dtype)
        
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        return self.layer_norm(self.Embedding(x))


class CreditRiskModel(nn.Module):
    def __init__(self, dic, query_event_dim, loan_event_dim, hidden_size_query=64, hidden_size_loan=128,
                 num_layers_query=2, num_layers_loan=2,
                 dropout=0.3):
        super(CreditRiskModel, self).__init__()

        self.query_event_dim = query_event_dim
        self.loan_event_dim = loan_event_dim

        self.hidden_size_query = hidden_size_query
        self.hidden_size_loan = hidden_size_loan

        self.num_layers_query = num_layers_query
        self.num_layers_loan = num_layers_loan

        self.embed_query_org = EmbeddingLayerNorm(max(dic['lst_query_org'].values())+1, 64, padding_idx=0)
        self.embed_query_reason = EmbeddingLayerNorm(max(dic['lst_query_reason'].values())+1, 16, padding_idx=0)
        self.embed_query_org_type = EmbeddingLayerNorm(max(dic['lst_query_org_type'].values())+1, 16, padding_idx=0)

        self.embed_loan_org = EmbeddingLayerNorm(max(dic['lst_forg_code'].values())+1, 64, padding_idx=0)
        self.embed_loan_org_type = EmbeddingLayerNorm(max(dic['lst_forg_type'].values())+1, 32, padding_idx=0)
        self.embed_loan_repay_type = EmbeddingLayerNorm(max(dic['lst_frepay_type'].values())+1, 32, padding_idx=0)
        self.embed_loan_repay_freq = EmbeddingLayerNorm(max(dic['lst_frepay_freq'].values())+1, 16, padding_idx=0)
        self.embed_loan_loan_type = EmbeddingLayerNorm(max(dic['lst_floan_type'].values())+1, 32, padding_idx=0)
        self.embed_loan_acc_type = EmbeddingLayerNorm(max(dic['lst_facc_type'].values())+1, 16, padding_idx=0)
        self.embed_loan_currency_type = EmbeddingLayerNorm(max(dic['lst_fcurrency_type'].values())+1, 16, padding_idx=0)

        self.embed_business_status = EmbeddingLayerNorm(max(dic['business_status'].values())+1, 32, padding_idx=0)
        self.embed_nationality = EmbeddingLayerNorm(max(dic['nationality'].values())+1, 8, padding_idx=0)
        self.embed_education = EmbeddingLayerNorm(max(dic['education'].values())+1, 16, padding_idx=0)
        self.embed_degree = EmbeddingLayerNorm(max(dic['degree'].values())+1, 16, padding_idx=0)
        self.embed_gender = EmbeddingLayerNorm(max(dic['gender'].values())+1, 8, padding_idx=0)
        self.embed_living_status = EmbeddingLayerNorm(max(dic['living_status'].values())+1, 16, padding_idx=0)
        self.embed_career_status = EmbeddingLayerNorm(max(dic['career_status'].values())+1, 8, padding_idx=0)
        self.embed_enterprise_attribute = EmbeddingLayerNorm(max(dic['enterprise_attribute'].values())+1, 16, padding_idx=0)
        self.embed_business = EmbeddingLayerNorm(max(dic['business'].values())+1, 32, padding_idx=0)
        self.embed_occupation = EmbeddingLayerNorm(max(dic['occupation'].values())+1, 16, padding_idx=0)
        self.embed_job = EmbeddingLayerNorm(max(dic['job'].values())+1, 16, padding_idx=0)
        self.embed_title = EmbeddingLayerNorm(max(dic['title'].values())+1, 16, padding_idx=0)
        self.embed_marriage_status = EmbeddingLayerNorm(max(dic['marriage_status'].values())+1, 8, padding_idx=0)

        self.embed_enterprise = EmbeddingLayerNorm(max(dic['enterprise'].values())+1, 128, padding_idx=0)

        self.embed_addr_postal = EmbeddingLayerNorm(max(dic['postal_address'].values())+1, 128, padding_idx=0)
        self.embed_addr_permanent = EmbeddingLayerNorm(max(dic['permanent_residence_address'].values())+1, 128, padding_idx=0)
        self.embed_addr_living = EmbeddingLayerNorm(max(dic['living_address'].values())+1, 128, padding_idx=0)
        self.embed_addr_enterprise = EmbeddingLayerNorm(max(dic['enterprise_address'].values())+1, 128, padding_idx=0)
        # self.embed_addr_spouse_enterprise = EmbeddingLayerNorm(max(dic['title'].values()), 128, padding_idx=0)

        self.lstm_query = nn.LSTM(
            input_size=103,
            hidden_size=hidden_size_query,
            num_layers=num_layers_query,
            batch_first=True,
            dropout=dropout if num_layers_query > 1 else 0
        )

        self.attention_query = nn.Sequential(
            nn.Linear(hidden_size_query, hidden_size_query),
            nn.Tanh(),
            nn.Linear(hidden_size_query, 1),
            nn.Softmax(dim=1)
        )

        self.lstm_loan = nn.LSTM(
            input_size=218,
            hidden_size=hidden_size_loan,
            num_layers=num_layers_loan,
            batch_first=True,
            dropout=dropout if num_layers_loan > 1 else 0
        )

        self.attention_loan = nn.Sequential(
            nn.Linear(hidden_size_loan, hidden_size_loan),
            nn.Tanh(),
            nn.Linear(hidden_size_loan, 1),
            nn.Softmax(dim=1)
        )

        self.layer_norm_q = nn.LayerNorm(query_event_dim)
        self.layer_norm_l = nn.LayerNorm(loan_event_dim)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1049, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)

        )

    def forward(self, x, dic):
        (lst_query_org, lst_query_org_type, lst_query_reason, lst_day_diff_cx, lst_sin_month, lst_cos_month_cx,
         lst_sin_week_cx, lst_cos_week_cx, lst_sin_day_cx, lst_cos_day_cx,
         lst_forg_code, lst_forg_type, lst_frepay_type, lst_frepay_freq, lst_floan_type, lst_floan_amount,
         lst_fterms, lst_facc_type, lst_fcreditline, lst_fcurrency_type, lst_day_diff_loan, lst_sin_month_loan,
         lst_cos_month_loan, lst_sin_week_loan, lst_cos_week_loan, lst_sin_day_loan, lst_cos_day_loan,
         business_status, gender, education, degree, age, nationality, living_status,
         career_status, enterprise_attribute, business, occupation, job, title, marriage_status,
         postal_address, permanent_residence_address, living_address, enterprise, enterprise_address) = preprocess(x)

        vec_query_org = self.embed_query_org(
            get_index_fill(lst_query_org, dic['lst_query_org'], split_type='', max_len=self.query_event_dim))
        vec_query_reason = self.embed_query_reason(
            get_index_fill(lst_query_reason, dic['lst_query_reason'], split_type='', max_len=self.query_event_dim))
        vec_query_org_type = self.embed_query_org_type(
            get_index_fill(lst_query_org_type, dic['lst_query_org_type'], split_type='', max_len=self.query_event_dim))

        vec_loan_org = self.embed_loan_org(
            get_index_fill(lst_forg_code, dic['lst_forg_code'], split_type='', max_len=self.loan_event_dim))
        vec_loan_org_type = self.embed_loan_org_type(
            get_index_fill(lst_forg_type, dic['lst_forg_type'], split_type='', max_len=self.loan_event_dim))
        vec_loan_repay_type = self.embed_loan_repay_type(
            get_index_fill(lst_frepay_type, dic['lst_frepay_type'], split_type='', max_len=self.loan_event_dim))
        vec_loan_repay_freq = self.embed_loan_repay_freq(
            get_index_fill(lst_frepay_freq, dic['lst_frepay_freq'], split_type='', max_len=self.loan_event_dim))
        vec_loan_loan_type = self.embed_loan_loan_type(
            get_index_fill(lst_floan_type, dic['lst_floan_type'], split_type='', max_len=self.loan_event_dim))
        vec_loan_acc_type = self.embed_loan_acc_type(
            get_index_fill(lst_facc_type, dic['lst_facc_type'], split_type='', max_len=self.loan_event_dim))
        vec_loan_currency_type = self.embed_loan_currency_type(
            get_index_fill(lst_fcurrency_type, dic['lst_fcurrency_type'], split_type='', max_len=self.loan_event_dim))

        vec_business_status = self.embed_business_status(get_index_value(business_status, dic['business_status']))
        vec_nationality = self.embed_nationality(get_index_value(nationality, dic['nationality']))
        vec_education = self.embed_education(get_index_value(education, dic['education']))
        vec_degree = self.embed_degree(get_index_value(degree, dic['degree']))
        vec_gender = self.embed_gender(get_index_value(gender, dic['gender']))
        vec_living_status = self.embed_living_status(get_index_value(living_status, dic['living_status']))
        vec_career_status = self.embed_career_status(get_index_value(career_status, dic['career_status']))
        vec_enterprise_attribute = self.embed_enterprise_attribute(get_index_value(enterprise_attribute, dic['enterprise_attribute']))
        vec_business = self.embed_business(get_index_value(business, dic['business']))
        vec_occupation = self.embed_occupation(get_index_value(occupation, dic['occupation']))
        vec_job = self.embed_job(get_index_value(job, dic['job']))
        vec_title = self.embed_title(get_index_value(title, dic['title']))
        vec_marriage_status = self.embed_marriage_status(get_index_value(marriage_status, dic['marriage_status']))

        vec_enterprise, _ = torch.max(torch.transpose(self.embed_enterprise(
            get_index_list(enterprise, dic['enterprise'], split_type='jieba')), 0, 1), dim=1)
        vec_addr_postal, _ = torch.max(torch.transpose(self.embed_addr_postal(
            get_index_list(postal_address, dic['postal_address'], split_type='jieba')), 0, 1), dim=1)
        vec_addr_permanent, _ = torch.max(torch.transpose(self.embed_addr_permanent(
            get_index_list(permanent_residence_address, dic['permanent_residence_address'], split_type='jieba')), 0, 1), dim=1)
        vec_addr_living, _ = torch.max(torch.transpose(self.embed_addr_living(
            get_index_list(living_address, dic['living_address'], split_type='jieba')), 0, 1), dim=1)
        vec_addr_enterprise, _ = torch.max(torch.transpose(self.embed_addr_enterprise(
            get_index_list(enterprise_address, dic['enterprise_address'], split_type='jieba')), 0, 1), dim=1)

        # vec_spouse_enterprise = self.embed_addr_spouse_enterprise(enterprise_address)

        # 64 + 16 + 16 + 1*7 = 103
        x_query = torch.cat([vec_query_org,
                             vec_query_org_type,
                             vec_query_reason,
                             torch.tensor(pad_sequence(lst_day_diff_cx, self.query_event_dim, pad=-99)).reshape(self.query_event_dim,1),
                             torch.tensor(pad_sequence(lst_sin_month, self.query_event_dim, pad=-99)).reshape(self.query_event_dim,1),
                             torch.tensor(pad_sequence(lst_cos_month_cx, self.query_event_dim, pad=-99)).reshape(self.query_event_dim,1),
                             torch.tensor(pad_sequence(lst_sin_week_cx, self.query_event_dim, pad=-99)).reshape(self.query_event_dim,1),
                             torch.tensor(pad_sequence(lst_cos_week_cx, self.query_event_dim, pad=-99)).reshape(self.query_event_dim,1),
                             torch.tensor(pad_sequence(lst_sin_day_cx, self.query_event_dim, pad=-99)).reshape(self.query_event_dim,1),
                             torch.tensor(pad_sequence(lst_cos_day_cx, self.query_event_dim, pad=-99)).reshape(self.query_event_dim,1),],
                            dim=1).float()

        #  64 + 32 + 32 + 16 + 32 + 16 + 16 + 1*10 = 218
        x_loan = torch.cat([vec_loan_org,
                            vec_loan_org_type,
                            vec_loan_repay_type,
                            vec_loan_repay_freq,
                            vec_loan_loan_type,
                            vec_loan_acc_type,
                            vec_loan_currency_type,
                            torch.tensor(pad_sequence(lst_floan_amount, self.loan_event_dim, pad=-99)).reshape(self.loan_event_dim,1),
                            torch.tensor(pad_sequence(lst_fterms, self.loan_event_dim, pad=-99)).reshape(self.loan_event_dim,1),
                            torch.tensor(pad_sequence(lst_fcreditline, self.loan_event_dim, pad=-99)).reshape(self.loan_event_dim,1),
                            torch.tensor(pad_sequence(lst_day_diff_loan, self.loan_event_dim, pad=-99)).reshape(self.loan_event_dim,1),
                            torch.tensor(pad_sequence(lst_sin_month_loan, self.loan_event_dim, pad=-99)).reshape(self.loan_event_dim,1),
                            torch.tensor(pad_sequence(lst_cos_month_loan, self.loan_event_dim, pad=-99)).reshape(self.loan_event_dim,1),
                            torch.tensor(pad_sequence(lst_sin_week_loan, self.loan_event_dim, pad=-99)).reshape(self.loan_event_dim,1),
                            torch.tensor(pad_sequence(lst_cos_week_loan, self.loan_event_dim, pad=-99)).reshape(self.loan_event_dim,1),
                            torch.tensor(pad_sequence(lst_sin_day_loan, self.loan_event_dim, pad=-99)).reshape(self.loan_event_dim,1),
                            torch.tensor(pad_sequence(lst_cos_day_loan, self.loan_event_dim, pad=-99)).reshape(self.loan_event_dim,1)],
                           dim=1).float()

        #  64 + 32 + 32 + 16 + 32 + 16 + 16 + 1*10 = 209
        x_info = torch.cat([vec_business_status, vec_nationality, vec_education, vec_gender, vec_degree,
                            vec_career_status, vec_enterprise_attribute, vec_business, vec_occupation, vec_job,
                            vec_title, vec_marriage_status, vec_living_status,
                            torch.tensor([age])],
                           dim=-1).float()
        # 128 * 5
        x_addr = torch.cat([vec_enterprise, vec_addr_postal, vec_addr_permanent, vec_addr_living,
                            vec_addr_enterprise], dim=-1).float()

        lstm_out_q, (hn, cn) = self.lstm_query(x_query)
        attention_weights_q = self.attention_query(lstm_out_q)
        vector_q = torch.sum(attention_weights_q * lstm_out_q, dim=1)
        # q = self.layer_norm_q(vector_q)

        lstm_out_l, (hn, cn) = self.lstm_loan(x_loan)
        attention_weights_l = self.attention_loan(lstm_out_l)
        vector_l = torch.sum(attention_weights_l * lstm_out_l, dim=1)
        # l = self.layer_norm_l(vector_l)

        vec = torch.cat([vector_q, vector_l, x_info, x_addr], dim=-1)

        out = self.fc(vec)

        return out


def eval_model(model, loader_set, criterion):
    res = []
    for k, v in loader_set.items():
        loss = 0
        preds = []
        true = []
        if k != 's':
            with torch.no_grad():
                for batch_X_cpu, batch_y_cpu in v:
                    batch_X = batch_X_cpu.to(device, non_blocking=True)
                    batch_y = batch_y_cpu.to(device, non_blocking=True)
                    outputs = model(batch_X)
                    v_loss = criterion(outputs, batch_y)
                    loss += v_loss.item()

                    predicted = torch.softmax(outputs, dim=1)[:, 1]
                    preds.extend(predicted.cpu().numpy())
                    true.extend(batch_y.cpu().numpy())

            fpr_off, tpr_off, _ = roc_curve(true, preds)
            v_ks = abs(fpr_off - tpr_off).max()
            v_auc = max(auc(fpr_off, tpr_off), 1 - auc(fpr_off, tpr_off))
            res.append({'data_set': k, 'avg_loss': loss / len(v), 'ks': v_ks, 'auc': v_auc})

    return res


def constraint_checker(eval_res, rule, threshold):
    threshold = min(threshold, eval_res.loc[eval_res['data_set'] == 'oot', 'ks'].values * 0.2)
    if rule == 1:
        case_1 = 1 if abs(eval_res.loc[eval_res['data_set'] == 'train', 'ks'].values[0] -
                          eval_res.loc[eval_res['data_set'] == 'oot', 'ks'].values[0]) <= threshold else 0
        case_2 = 1 if abs(eval_res.loc[eval_res['data_set'] == 'train', 'ks'].values[0] -
                          eval_res.loc[eval_res['data_set'] == 'valid', 'ks'].values[0]) <= threshold else 0
        case_3 = 1 if abs(eval_res.loc[eval_res['data_set'] == 'oot', 'ks'].values[0] -
                          eval_res.loc[eval_res['data_set'] == 'valid', 'ks'].values[0]) <= threshold else 0
        return case_1 * case_2 * case_3

    elif rule == 2:
        pass


def early_stop(epoch, model, loss, best_loss, constraint_checker, patience_counter, patience, eval_res, rule, threshold, file_path):

    if loss > 0.142:
        torch.save(model.state_dict(), file_path + 'model_{}.pth'.format(str(epoch)))

    if loss > best_loss and constraint_checker(eval_res, rule, threshold):
        best_loss = loss
        patience_counter = 0
        torch.save(model.state_dict(), file_path + 'best_model.pth')  # 保存最佳模型
    else:
        patience_counter += 1

    print('loss: ', loss, ' best_loss: ', best_loss,
          ' constraint_checker: ', constraint_checker(eval_res, rule, threshold),
          ' patience_counter: ', patience_counter)

    if patience_counter >= patience:
        return 1, best_loss, patience_counter
    else:
        return 0, best_loss, patience_counter


def get_training_metrics(param, process, file_path):
    plt.figure(figsize=(12, 4))

    length = int(min(process.shape[0]/len(process['data_set'].unique()), param['num_epochs']))
    plt.subplot(1, 2, 1)
    plt.plot([i for i in range(length)], process[process['data_set'] == 'train']['avg_loss'], label='Train Loss')
    plt.plot([i for i in range(length)], process[process['data_set'] == 'valid']['avg_loss'], label='Validation Loss')
    plt.plot([i for i in range(length)], process[process['data_set'] == 'oot']['avg_loss'], label='Test Loss')
    plt.title('Training & Validation & Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot([i for i in range(length)], process[process['data_set'] == 'train']['ks'], label='Train KS')
    plt.plot([i for i in range(length)], process[process['data_set'] == 'valid']['ks'], label='Validation KS')
    plt.plot([i for i in range(length)], process[process['data_set'] == 'oot']['ks'], label='Test KS')
    plt.title('Training & Validation & Test KS')
    plt.xlabel('Epoch')
    plt.ylabel('KS')
    plt.legend()

    plt.tight_layout()
    plt.savefig(file_path + 'training_metrics.png')
    plt.show()


def get_fusion_test(data, col, ex_lst, labels, mons):
    print(col)
    res = bi_train(data[data['label'] >= 0][['score', col] + ex_lst], dep='label', exclude=ex_lst, model_switch=[0, 0, 1, 0, 0])
    data['score_fution_' + col] = res['model_lgb'].predict_proba(data[['score', col]])[:, -1]
    ks_vintage_org = get_vintage(data, mons, col, labels, 'ks')
    ks_vintage_fusion = get_vintage(data, mons, 'score_fution_' + col, labels, 'ks')
    diff = ks_vintage_fusion.replace('', np.nan) - ks_vintage_org.replace('', np.nan)
    print(ks_vintage_org)
    print(ks_vintage_fusion)
    print(diff)


def train_model(model, loader_set, criterion, optimizer, scheduler, num_epochs, rule, threshold, file_path, patience=10):
    process = []
    best_loss = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for batch_X, batch_y in loader_set['train']:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        eval_res = pd.DataFrame(eval_model(model, loader_set, criterion))
        eval_res['epoch'] = epoch
        process.append(eval_res)

        # 学习率调度
        scheduler.step(eval_res.loc[eval_res['data_set'] == 'valid', 'avg_loss'].values[0])

        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}] ')
            print(eval_res)
            print('')

        f, best_loss, patience_counter = early_stop(epoch, model, eval_res.loc[eval_res['data_set'] == 'oot', 'ks'].values[0], best_loss, constraint_checker,
                                                    patience_counter, patience, eval_res, rule, threshold, file_path)
        if f:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    return pd.concat(process), epoch + 1


def predict_score(data, model, batch_size=50000):
    X_tensor = torch.FloatTensor(list(data['features'])).to(device)
    y_tensor = torch.LongTensor(data['label'].values).to(device)

    data_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for batch_X, batch_y in tqdm(data_loader):
            predicted = torch.softmax(model(batch_X), dim=1)[:, 1]
            preds.extend(predicted.cpu().numpy())

    return preds


def train_model_clean(model, loader_set, criterion, optimizer, scheduler, num_epochs, rule, threshold, file_path, patience=10, accumulation_steps=5):
    process = []
    best_loss = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for batch_idx, (batch_X_cpu, batch_y_cpu) in enumerate(loader_set['train']):

            batch_X = batch_X_cpu.to(device, non_blocking=True)
            batch_y = batch_y_cpu.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            outputs = model(batch_X)
            
            loss = criterion(outputs.squeeze(), batch_y.squeeze())
            # loss = loss / accumulation_steps
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪
            
            # if (batch_idx + 1) % accumulation_steps == 0:

            #     optimizer.step()
            #     optimizer.zero_grad()
            
            #     print(loss)
            #     print(f"Epoch {epoch}, 更新参数 at batch {batch_idx}")
        
            optimizer.step()
            train_loss += loss.item()

            if batch_idx % 100 == 0:
                print(loss)
                emergency_memory_cleanup()

        model.eval()
        eval_res = pd.DataFrame(eval_model(model, loader_set, criterion))
        eval_res['epoch'] = epoch
        process.append(eval_res)

        # 学习率调度
        scheduler.step(eval_res.loc[eval_res['data_set'] == 'valid', 'avg_loss'].values[0])

        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}] ')
            print(eval_res)
            print('')

        f, best_loss, patience_counter = early_stop(epoch, model, eval_res.loc[eval_res['data_set'] == 'oot', 'ks'].values[0], best_loss, constraint_checker,
                                                    patience_counter, patience, eval_res, rule, threshold, file_path)
        if f:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    return pd.concat(process), epoch + 1
