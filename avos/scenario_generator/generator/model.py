import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily


def _init_weights(module, nlayer=1):
    """GPT-2 / nanoGPT style weight initialization with depth-scaled residual projections."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class MCG_block(nn.Module):
    def __init__(self, hidden_dim):
        super(MCG_block, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, inp, context, mask):
        context = context.unsqueeze(1)
        mask = mask.unsqueeze(-1)

        inp = self.MLP(inp)
        inp = inp * context
        inp = inp.masked_fill(mask == 0, -1e9)
        context = torch.max(inp, dim=1)[0]
        return inp, context


class CG_stacked(nn.Module):
    def __init__(self, stack_num, hidden_dim):
        super(CG_stacked, self).__init__()
        self.CGs = nn.ModuleList()
        self.stack_num = stack_num
        for i in range(stack_num):
            self.CGs.append(MCG_block(hidden_dim))

    def forward(self, inp, context, mask):

        inp_, context_ = self.CGs[0](inp, context, mask)
        for i in range(1, self.stack_num):
            inp, context = self.CGs[i](inp_, context_, mask)
            inp_ = (inp_ * i + inp) / (i + 1)
            context_ = (context_ * i + context) / (i + 1)
        return inp_, context_


class MLP(nn.Module):
    def __init__(self, dims):
        super(MLP, self).__init__()

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlp(x)
        return x


def clones(module, N):
    """Product N identical layers."""
    # print("clones!")
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    """Compute Scaled Dot Product Attention"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4) # create 4 linear layers
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        # (batch_size, seq_length, d_model)

        # 1) Do all the linear projections in batch from d_model => h * d_k
        query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))]
        query, key, value = [x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2) for x in (query, key, value)]
        # (batch_size, h, seq_length, d_k)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](x)


def pos2posemb(pos, num_pos_feats=128, temperature=10000):
    """
    Copied from https://github.com/OpenDriveLab/UniAD/blob/main/projects/mmdet3d_plugin/models/utils/functional.py
    Convert 2D position into positional embeddings.

    Args:
        pos (torch.Tensor): Input N-D position tensor.
        num_pos_feats (int, optional): Number of positional features. Default is 128.
        temperature (int, optional): Temperature factor for positional embeddings. Default is 10000.

    Returns:
        torch.Tensor: Positional embeddings tensor.
    """
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    D = pos.shape[-1]
    pos_dims = []
    for i in range(D):
        pos_dim_i = pos[..., i, None] / dim_t
        pos_dim_i = torch.stack((pos_dim_i[..., 0::2].sin(), pos_dim_i[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_dims.append(pos_dim_i)
    posemb = torch.cat(pos_dims, dim=-1)
    return posemb


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe[:, :x.size(1)]


class ScenarioGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ablation = config['DATA']['ABLATION']
        self.ablation_variable_start_idx = config['DATA']['ABLATION_VARIABLE_START_IDX']
        self.ablation_variable_end_idx = config['DATA']['ABLATION_VARIABLE_END_IDX']

        d_model = config['MODEL']['HIDDEN_DIM'] * 2
        object_code_dim = config['DATA']['OBJECT_CODE_DIM']
        self.pos_encoding_dim = config['MODEL']['POS_ENCODING_DIM']
        query_num = config['MODEL']['QUERY_NUM']
        # interaction_code_dim = config['DATA']['INTERACTION_CODE_DIM']

        nhead = config['MODEL']['NHEAD']
        nlayer = config['MODEL']['NLAYER']
        ff_dim = config['MODEL']['FF_DIM']
        activation = config['MODEL']['ACTIVATION']
        dropout = config['MODEL']['DROPOUT']

        self.use_one_gmm = config['MODEL']['USE_ONE_GMM']
        self.use_all_locations = config['LOSS']['USE_ALL_LOCATIONS']

        self.hidden_dim = config['MODEL']['HIDDEN_DIM']

        # Split object input mode
        self.split_object_input = config['DATA'].get('SPLIT_OBJECT_INPUT', False)

        if self.split_object_input:
            self.ablation_field = config['DATA'].get('ABLATION_FIELD', 'none')

            object_pose_dim = config['DATA']['OBJECT_POSE_DIM']
            num_obj_types = config['DATA']['NUM_OBJ_TYPES']
            num_action_trends = config['DATA']['NUM_ACTION_TRENDS']
            num_action_types = config['DATA']['NUM_ACTION_TYPES']
            code_emb_dim = config['DATA']['CODE_EMB_DIM']
            action_step = config['PREPROCESS']['ACTION_STEP']

            # Pose branch: pos2posemb + MLP
            self.pose_query_encoding = nn.Sequential(
                nn.Linear(object_pose_dim * self.pos_encoding_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )

            # Code branch: embeddings for categorical features
            self.obj_type_emb = nn.Embedding(num_obj_types, code_emb_dim)
            self.action_trend_emb = nn.Embedding(num_action_trends, code_emb_dim)
            self.action_type_emb = nn.Embedding(num_action_types, code_emb_dim)

            # Code branch: linear projection for continuous features (speed_range, size_range)
            self.cont_proj = nn.Linear(2, code_emb_dim)

            # Code fusion: type_emb(1) + cont_proj(1) + trend_embs(5) + type_embs(5) = 12 * code_emb_dim
            code_total_dim = code_emb_dim * (1 + 1 + action_step + action_step)
            self.code_fusion = nn.Sequential(
                nn.Linear(code_total_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.object_code_embedding = nn.Sequential(
                nn.Linear(object_code_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            )

            self.object_query_encoding = nn.Sequential(
                nn.Linear(self.pos_encoding_dim*query_num, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )

        self.code_encoding = PositionalEncoding(d_model)

        self.type_embedding = nn.Embedding(34, self.hidden_dim)
        self.state_embedding = nn.Embedding(6, self.hidden_dim)
        self.map_po_encoding = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim*2),
            nn.LayerNorm(self.hidden_dim*2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim*2, self.hidden_dim)
        )

        self.CG_map = CG_stacked(5, self.hidden_dim)

        self.learnable_query = nn.Parameter(torch.randn(1, query_num, d_model))

        layer_cfg = {'d_model': d_model, 'nhead': nhead, 'dim_feedforward': ff_dim, 'dropout': dropout, 'activation': activation, 'batch_first': True}
        decoder_layer = nn.TransformerDecoderLayer(**layer_cfg)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=nlayer)

        self.mlp_dim = config['MODEL']['MLP_DIM']
        self.attr_gmm_k = config['MODEL']['ATTR_GMM_K']
        if self.use_one_gmm:
            self.init_head = MLP([d_model, self.mlp_dim, self.attr_gmm_k * (1+4+10)])   # 4*(4+1)/2
        else:
            self.bbox_head = MLP([d_model, self.mlp_dim, self.attr_gmm_k*(1+2+3)])
            self.speed_hdn_head = MLP([d_model, self.mlp_dim, self.attr_gmm_k*(1+1+1)])
            self.speed_head = MLP([d_model, self.mlp_dim, self.attr_gmm_k*(1+1+1)])

        self.m_k = config['MODEL']['M_K']
        self.step = config['MODEL']['STEP']
        self.future_prob_head = MLP([d_model, self.mlp_dim, self.mlp_dim//2, self.m_k])
        self.future_pos_head = MLP([d_model, self.mlp_dim, self.mlp_dim//2, self.m_k*self.step*2])
        self.future_hdn_head = MLP([d_model, self.mlp_dim, self.mlp_dim//2, self.m_k*self.step])
        self.future_vel_head = MLP([d_model, self.mlp_dim, self.mlp_dim//2, self.m_k*self.step*2])

        self._initialize_weights(nlayer)

    def _initialize_weights(self, nlayer):
        self.apply(lambda m: _init_weights(m, nlayer))

        # Depth-scaled init for Transformer residual projections (GPT-2 style)
        residual_std = 0.02 / math.sqrt(2 * nlayer)
        for layer in self.decoder.layers:
            if hasattr(layer, 'linear2'):
                nn.init.normal_(layer.linear2.weight, mean=0.0, std=residual_std)
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'out_proj'):
                nn.init.normal_(layer.self_attn.out_proj.weight, mean=0.0, std=residual_std)
            if hasattr(layer, 'multihead_attn') and hasattr(layer.multihead_attn, 'out_proj'):
                nn.init.normal_(layer.multihead_attn.out_proj.weight, mean=0.0, std=residual_std)

        nn.init.normal_(self.learnable_query, mean=0.0, std=0.02)

    def map_data_encoding(self, map_data):
        map_po = map_data[..., :4]
        map_type = map_data[..., 4].to(int)
        map_state = map_data[..., 5].to(int)

        map_type_embedding = self.type_embedding(map_type)
        map_state_embedding = self.state_embedding(map_state)
        map_po_encoding = self.map_po_encoding(map_po)

        map_data_encoding = map_po_encoding + map_type_embedding + map_state_embedding

        return map_data_encoding

    def map_feature_extract(self, map_encoding, map_mask, object_context):
        # map information fusion with CG block
        map_encoding, map_context = self.CG_map(map_encoding, object_context, map_mask)
        # map context feature
        map_feature = torch.cat([map_encoding, map_context.unsqueeze(1).repeat(1, map_encoding.shape[1], 1)], dim=-1)

        return map_feature, map_context

    def output_to_distri(self, para, n):
        if n == 1:      # one-dimensional normal distribution
            loc = para[..., 0]  # mean
            scale = torch.exp(para[..., 1])  # std
            distri = torch.distributions.Normal(loc, scale)
            return distri

        elif n == 2:    # two-dimensional normal distribution (keep existing implementation)
            loc = para[..., :2]  # mean (mu_x, mu_y)
            tril = para[..., 2]  #
            diag = para[..., 3:]  # ln(std)

            sigma_1 = torch.exp(diag[..., 0])
            sigma_2 = torch.exp(diag[..., 1])
            rho = torch.tanh(tril)

            cov = (
                torch.stack(
                    [sigma_1 ** 2, rho * sigma_1 * sigma_2, rho * sigma_1 * sigma_2, sigma_2 ** 2],
                    dim=-1).view(*loc.shape[:-1], 2, 2)
            )

            distri = torch.distributions.multivariate_normal.MultivariateNormal(loc=loc, covariance_matrix=cov)
            return distri

        else:           # n-dimensional multivariate normal distribution (n>2)
            loc = para[..., :n]
            cov_params = para[..., n:n + n * (n + 1) // 2]

            # build lower-triangular matrix L
            L = torch.zeros(*para.shape[:-1], n, n, device=para.device, dtype=para.dtype)
            tril_indices = torch.tril_indices(n, n, offset=0)
            L[..., tril_indices[0], tril_indices[1]] = cov_params

            # use softplus to ensure positive diagonal entries
            diag_indices = torch.arange(n, device=para.device)
            L[..., diag_indices, diag_indices] = torch.nn.functional.softplus(L[..., diag_indices, diag_indices])

            # add regularization to the diagonal
            reg = 1e-3  # use a larger regularization term
            L[..., diag_indices, diag_indices] = L[..., diag_indices, diag_indices] + reg

            # use scale_tril directly instead of covariance_matrix
            distri = torch.distributions.multivariate_normal.MultivariateNormal(
                loc=loc,
                scale_tril=L,  # use the Cholesky factor instead of the full covariance matrix
                validate_args=False
            )

            return distri


    def gmm_modeling(self, result, scenario_feature):
        if self.use_one_gmm:
            init_out = self.init_head(scenario_feature).view([*scenario_feature.shape[:-1], self.attr_gmm_k, -1])
            init_weight_logit = init_out[..., 0]
            init_param = init_out[..., 1:]
            init_distri = self.output_to_distri(init_param, 4)
            init_weight = Categorical(logits=init_weight_logit, validate_args=False)
            result['pred_init'] = (MixtureSameFamily(init_weight, init_distri, validate_args=False))
        else:
            bbox_out = self.bbox_head(scenario_feature).view([*scenario_feature.shape[:-1], self.attr_gmm_k, -1])
            bbox_weight_logit = bbox_out[..., 0]
            bbox_param = bbox_out[..., 1:]
            bbox_distri = self.output_to_distri(bbox_param, 2)
            bbox_weight = Categorical(logits=bbox_weight_logit, validate_args=False)
            result['pred_bbox'] = (MixtureSameFamily(bbox_weight, bbox_distri, validate_args=False))

            speed_hdn_out = self.speed_hdn_head(scenario_feature).view([*scenario_feature.shape[:-1], self.attr_gmm_k, -1])
            speed_hdn_weight_logit = speed_hdn_out[..., 0]
            speed_hdn_param = speed_hdn_out[..., 1:]
            speed_hdn_distri = self.output_to_distri(speed_hdn_param, 1)
            speed_hdn_weight = Categorical(logits=speed_hdn_weight_logit, validate_args=False)
            result['pred_speed_hdn_rel_dir'] = (MixtureSameFamily(speed_hdn_weight, speed_hdn_distri, validate_args=False))

            speed_out = self.speed_head(scenario_feature).view([*scenario_feature.shape[:-1], self.attr_gmm_k, -1])
            speed_weight_logit = speed_out[..., 0]
            speed_param = speed_out[..., 1:]
            speed_distri = self.output_to_distri(speed_param, 1)
            speed_weight = Categorical(logits=speed_weight_logit, validate_args=False)
            result['pred_speed'] = (MixtureSameFamily(speed_weight, speed_distri, validate_args=False))

        return result

    def motion_pred(self, result, scenario_feature):
        batch_size = scenario_feature.shape[0]

        future_prob_pred = self.future_prob_head(scenario_feature)
        result['pred_future_prob'] = future_prob_pred

        future_pos_pred = self.future_pos_head(scenario_feature).view(batch_size, -1, self.m_k, self.step, 2)
        future_pos_pred = future_pos_pred.cumsum(dim=-2)
        result['pred_future_pos'] = future_pos_pred

        future_hdn_pred = self.future_hdn_head(scenario_feature).view(batch_size, -1, self.m_k, self.step, 1)
        future_hdn_pred = future_hdn_pred.cumsum(dim=-2)
        result['pred_future_hdn'] = future_hdn_pred

        future_vel_pred = self.future_vel_head(scenario_feature).view(batch_size, -1, self.m_k, self.step, 2)
        result['pred_future_vel'] = future_vel_pred

    def forward(self, batch):
        object_code = batch['object_code']
        object_mask = batch['object_mask']

        map_data = batch['map']
        map_mask = batch['map_mask']
        lane_mask = batch['lane_mask']

        batch_size = object_code.shape[0]
        device = object_code.device

        # Object Encoder
        if self.split_object_input:
            object_pose = batch['object_pose']              # (B, N, 4)
            object_code_split = batch['object_code_split']  # (B, N, 13)

            # Pose branch: sinusoidal positional encoding + MLP
            pose_encoding = pos2posemb(object_pose, self.pos_encoding_dim)
            pose_encoding = self.pose_query_encoding(pose_encoding)  # (B, N, d_model)

            # Code branch - always compute all features, then zero-out ablated one
            # object_code_split layout: [obj_type, speed_range, size_range, trend*5, atype*5]

            # 1) obj_type embedding
            type_feat = self.obj_type_emb(object_code_split[..., 0].long())     # (B, N, emb)

            # 2) continuous: speed_range(col1) + size_range(col2)
            cont_input = object_code_split[..., 1:3]                            # (B, N, 2)
            if self.ablation and self.ablation_field == 'speed_range':
                cont_input = cont_input.clone()
                cont_input[..., 0] = 0
            elif self.ablation and self.ablation_field == 'size_range':
                cont_input = cont_input.clone()
                cont_input[..., 1] = 0
            cont_feat = self.cont_proj(cont_input)                              # (B, N, emb)

            # 3) action trend embeddings (5 steps, cols 3-7)
            trend_feats = []
            for s in range(5):
                trend_feats.append(self.action_trend_emb(object_code_split[..., 3 + s].long()))
            trend_feat = torch.cat(trend_feats, dim=-1)                         # (B, N, 5*emb)

            # 4) action type embeddings (5 steps, cols 8-12)
            atype_feats = []
            for s in range(5):
                atype_feats.append(self.action_type_emb(object_code_split[..., 8 + s].long()))
            atype_feat = torch.cat(atype_feats, dim=-1)                         # (B, N, 5*emb)

            # Zero-out ablated categorical features
            if self.ablation:
                if self.ablation_field == 'obj_type':
                    type_feat = torch.zeros_like(type_feat)
                elif self.ablation_field == 'actions_trend':
                    trend_feat = torch.zeros_like(trend_feat)
                elif self.ablation_field == 'actions_type':
                    atype_feat = torch.zeros_like(atype_feat)

            # Fuse all code features
            code_feat = torch.cat([type_feat, cont_feat, trend_feat, atype_feat], dim=-1)
            code_encoding = self.code_fusion(code_feat)                         # (B, N, d_model)

            # Combine pose + code + learnable query
            learnable_query_encoding = self.learnable_query.repeat(batch_size, 1, 1)
            query_encoding = learnable_query_encoding + pose_encoding + code_encoding
        else:
            if self.ablation:
                object_code = torch.cat([object_code[..., :self.ablation_variable_start_idx], object_code[..., self.ablation_variable_end_idx+1:]], dim=-1)

            object_query_encoding = pos2posemb(object_code, self.pos_encoding_dim)
            object_query_encoding = self.object_query_encoding(object_query_encoding)
            learnable_query_encoding = self.learnable_query.repeat(batch_size, 1, 1)
            query_encoding = learnable_query_encoding + object_query_encoding

        # Map Encoder
        map_encoding = self.map_data_encoding(map_data)
        empty_context = torch.ones([batch_size, map_encoding.shape[-1]], device=device, dtype=torch.float32)
        map_feature, _ = self.map_feature_extract(map_encoding, map_mask, empty_context)
        lane_feature = map_feature[:, :batch['lane_mask'].shape[1], :]    # TODO lane_mask 1024?

        # Generative Transformer
        scenario_feature = self.decoder(
            tgt=query_encoding,
            memory=lane_feature,
            tgt_key_padding_mask=~object_mask,
            memory_key_padding_mask=~lane_mask
        )

        # ############################################# Scenario Decoder ############################################# #
        result = dict()

        # Init MLP
        self.gmm_modeling(result, scenario_feature)

        # Future MLP
        self.motion_pred(result, scenario_feature)

        return result
