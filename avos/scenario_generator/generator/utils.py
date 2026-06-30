import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch.utils.data.dataloader import default_collate


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


def fc_collate_fn(batch):
    result_batch = {}

    for key in batch[0].keys():
        result_batch[key] = default_collate([item[key] for item in batch])

    return result_batch


def format_labels(batch):
    batch_size = batch['object'].shape[0]
    labels = []

    for i in range(batch_size):
        label = {}

        # decide whether to predict the ego vehicle
        start_idx = 0   # if self.pred_ego else 1

        object_mask = batch['object_mask'][i]

        # gt vector label for each actor in the scenario
        label['labels'] = batch['object_lane_index'][i, object_mask][start_idx:]
        label['all_labels'] = batch['gt_lane_index'].transpose(1, 2)[i, object_mask][start_idx:]
        label['pos'] = batch['gt_long_lat'][i, label['labels']]
        label['speed'] = batch['gt_speed'][i, label['labels']]
        label['bbox'] = batch['gt_bbox'][i, label['labels']]
        label['speed_hdn_rel_dir'] = batch['gt_speed_hdn_rel_dir'][i, label['labels']]
        label['hdn_lane_rel_dir'] = batch['gt_hdn_lane_rel_dir'][i, label['labels']]

        label['init'] = torch.cat([label['speed'].unsqueeze(-1), label['speed_hdn_rel_dir'].unsqueeze(-1), label['bbox']], dim=-1)

        future_mask = batch['all_object_mask'][i][1:, object_mask]
        future_mask = future_mask[:, start_idx:]
        future_mask = torch.swapaxes(future_mask, 0, 1).unsqueeze(-1).repeat(1, 1, 2)
        label['future_mask'] = future_mask

        future_pos = batch['future_pos'][i][1:, start_idx:]
        label['future_pos'] = torch.swapaxes(future_pos, 0, 1).float()

        future_hdn = batch['future_hdn'][i][1:, start_idx:]
        label['future_hdn'] = torch.swapaxes(future_hdn, 0, 1).float()

        future_vel = batch['future_vel'][i][1:, start_idx:]
        label['future_vel'] = torch.swapaxes(future_vel, 0, 1).float()

        future_lat_disp = batch['future_lat_disp'][i][1:, start_idx:]
        label['future_lat_disp'] = torch.swapaxes(future_lat_disp, 0, 1).float()

        future_nearest_lane = batch['lane_based_rep'][i][1:, start_idx:, 5:]
        label['future_nearest_lane'] = torch.swapaxes(future_nearest_lane, 0, 1).float()

        # sort objects according to distance
        dist = torch.norm(batch['object'][i, object_mask][start_idx:, :2], dim=1)
        sort_idx = torch.argsort(dist)
        for attr in label:
            label[attr] = label[attr][sort_idx]
        label['dist'] = dist[sort_idx]

        labels.append(label)

    batch['labels'] = labels

    return batch


def rotate(x, y, angle):
  if isinstance(x, torch.Tensor):
      other_x_trans = torch.cos(angle) * x - torch.sin(angle) * y
      other_y_trans = torch.cos(angle) * y + torch.sin(angle) * x
      output_coords = torch.stack((other_x_trans, other_y_trans), axis=-1)

  else:
      other_x_trans = np.cos(angle) * x - np.sin(angle) * y
      other_y_trans = np.cos(angle) * y + np.sin(angle) * x
      output_coords = np.stack((other_x_trans, other_y_trans), axis=-1)
  return output_coords


def get_object_pos_from_vec(object_lane, pos, speed, speed_hdn_rel_dir, hdn_lane_rel_dir, bbox):
    x1, y1, x2, y2 = object_lane[:, 0], object_lane[:, 1], object_lane[:, 2], object_lane[:, 3]
    x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2

    lane_len = 5
    # lane_len = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    lane_dir = torch.atan2(y2 - y1, x2 - x1)

    long_pos = lane_len * pos[..., 0]
    lat_pos = lane_len * pos[..., 1]

    coord = rotate(lat_pos, long_pos, -np.pi / 2 + lane_dir)

    coord[:, 0] += x_center
    coord[:, 1] += y_center

    hdn = hdn_lane_rel_dir + lane_dir
    speed_dir = speed_hdn_rel_dir + hdn

    vel = torch.stack([torch.cos(speed_dir) * speed, torch.sin(speed_dir) * speed], axis=-1)

    valid_object_num = vel.shape[0]

    type = Tensor([[1]]).repeat(valid_object_num, 1).to(coord.device)
    object = torch.cat([coord, vel, hdn.unsqueeze(1), bbox, type], dim=-1)

    vec_based_info = torch.cat([pos, speed.unsqueeze(-1),
                                speed_hdn_rel_dir.unsqueeze(-1),
                                hdn_lane_rel_dir.unsqueeze(-1),
                                object_lane], dim=-1)

    return [object, vec_based_info]


def convert_pred_to_pos(future_pos, future_prob, init_pos, init_hdn):
    num_valid_objects = future_pos.shape[0]

    pos = []
    rel_pos = []
    for i in range(num_valid_objects):
        init_pos_i = init_pos[i][np.newaxis, :]
        motion_index = np.argmax(future_prob[i])
        pos_i = future_pos[i, motion_index]
        rel_pos_i = np.concatenate([np.zeros((1, 2)), pos_i], axis=0)
        rel_pos_i = rel_pos_i[:, np.newaxis, :]

        pos_i = rotate(pos_i[..., 0], pos_i[..., 1], init_hdn[i])

        pos_i = pos_i + init_pos_i
        pos_i = np.concatenate([init_pos_i, pos_i], axis=0)
        pos_i = pos_i[:, np.newaxis, :]

        pos.append(pos_i)
        rel_pos.append(rel_pos_i)

    pos = np.concatenate(pos, axis=1)
    rel_pos = np.concatenate(rel_pos, axis=1)

    return pos, rel_pos


def cal_rel_dir(dir1, dir2):
  dist = dir1 - dir2

  while not np.all(dist >= 0):
    dist[dist < 0] += np.pi * 2
  while not np.all(dist < np.pi * 2):
    dist[dist >= np.pi * 2] -= np.pi * 2

  dist[dist > np.pi] -= np.pi * 2
  return dist


def convert_pred_to_hdn_vel(future_hdn, future_vel, future_prob, init_hdn):
    num_valid_objects = future_hdn.shape[0]

    hdn = []
    vel = []
    for i in range(num_valid_objects):
        motion_index = np.argmax(future_prob[i])
        hdn_i = future_hdn[i, motion_index]
        vel_i = future_vel[i, motion_index]

        hdn_i = cal_rel_dir(hdn_i, -init_hdn[i])
        vel_i = rotate(vel_i[..., 0], vel_i[..., 1], init_hdn[i])

        hdn_i = hdn_i[:, np.newaxis, :]
        vel_i = vel_i[:, np.newaxis, :]

        hdn.append(hdn_i)
        vel.append(vel_i)

    hdn = np.concatenate(hdn, axis=1)
    vel = np.concatenate(vel, axis=1)

    return hdn, vel


def get_object_type_and_subtype(all_type):
    obj_type = []
    obj_subtype = []

    for i in range(len(all_type)):
        all_type_i = all_type[i]
        if all_type_i == 8:
            obj_type.append('VEHICLE')
            obj_subtype.append('UNKNOWN')
        elif all_type_i == 12:
            obj_type.append('CYCLIST')
            obj_subtype.append('UNKNOWN')
        elif all_type_i == 13:
            obj_type.append('PEDESTRIAN')
            obj_subtype.append('PEDESTRIAN')
        else:
            raise ValueError(f"Unknown type: {all_type_i}")

    return obj_type, obj_subtype


def process_scene(config, preds, batch):
    batch_size = len(batch['object'])
    use_one_gmm = config['MODEL']['USE_ONE_GMM']

    outputs = []
    for i in range(batch_size):
        output = {}

        start_idx = 0
        # sort objects according to distance
        object_i = batch['object'][i].cpu()
        object_mask_i = batch['object_mask'][i].cpu()
        object_i[~object_mask_i, :2] = 1e6

        dist = torch.norm(object_i[start_idx:, :2], dim=1)
        sort_idx = torch.argsort(dist)

        num_valid_objects = torch.sum(object_mask_i)
        valid_objects_idx = torch.arange(num_valid_objects)

        if use_one_gmm:
            init = preds['pred_init'].sample()[i, valid_objects_idx].detach().cpu()

            speed = torch.clip(init[:, 0], min=0.0, max=40.0)
            speed_hdn_rel_dir = torch.clip(init[:, 1], min=-torch.pi/2, max=torch.pi/2)
            bbox = torch.clip(init[:, 2:4], min=0.1, max=20.0)
        else:
            speed = torch.clip(preds['pred_speed'].sample()[i, valid_objects_idx].detach().cpu(), min=0.0, max=40.0)
            speed_hdn_rel_dir = torch.clip(preds['pred_speed_hdn_rel_dir'].sample()[i, valid_objects_idx].detach().cpu(), min=-torch.pi/2, max=torch.pi/2)
            bbox = torch.clip(preds['pred_bbox'].sample()[i, valid_objects_idx].detach().cpu(), min=0.1, max=20.0)

        # all_objects = get_object_pos_from_vec(pos, speed, hdn, bbox)

        # # get prob for each existing object
        # probs = []
        # for valid_object_idx in valid_objects_idx:
        #     prob = pred_logits[i, valid_object_idx][object_mask_i].softmax(-1)
        #     probs.append(prob)

        # TODO add map !

        object_i = object_i[sort_idx]

        output['type'], output['subtype'] = get_object_type_and_subtype(object_i[valid_objects_idx, -2])

        hdn = object_i[valid_objects_idx, 6]
        speed_dir = speed_hdn_rel_dir + hdn
        vel = torch.stack([torch.cos(speed_dir) * speed, torch.sin(speed_dir) * speed], axis=-1)

        init_pos = object_i[valid_objects_idx, 0:2].numpy()
        # init_vel = torch.column_stack([speed, torch.zeros(len(speed))]).cpu().numpy()       # TODO init_vel depend on speed_hdn_rel_dir
        init_vel = vel.numpy()
        init_acc = None
        init_hdn = object_i[valid_objects_idx, 6].numpy()
        init_size = bbox.numpy()

        output['init_pos'] = init_pos
        output['init_vel'] = init_vel
        output['init_acc'] = init_acc
        output['init_hdn'] = init_hdn
        output['init_size'] = init_size

        future_pos = preds['pred_future_pos'][i, valid_objects_idx].cpu().numpy()
        future_prob = preds['pred_future_prob'][i, valid_objects_idx].cpu().numpy()

        output['complete_pos'], output['complete_rel_pos'] = convert_pred_to_pos(future_pos, future_prob, init_pos, init_hdn)
        # exceed_bound = self._exceed_boundary(bound_i, center_i, output['traj'])
        future_hdn = preds['pred_future_hdn'][i, valid_objects_idx].cpu().numpy()
        future_vel = preds['pred_future_vel'][i, valid_objects_idx].cpu().numpy()
        output['future_hdn'], output['future_vel'] = convert_pred_to_hdn_vel(future_hdn, future_vel, future_prob, init_hdn)

        outputs.append(output)

    return outputs


@torch.no_grad()
def match(config, outputs, labels, object_mask):
    batch_size, num_objects, _ = outputs['pred_future_prob'].shape

    use_all_locations = config['LOSS']['USE_ALL_LOCATIONS']

    if use_all_locations:
        out_prob = outputs['pred_logits'][..., 0, :].clone()
    else:
        out_prob = outputs['pred_logits'].clone()
    out_prob = out_prob.softmax(-1)
    out_prob[~object_mask] = 0.0
    out_prob = out_prob.flatten(0, 1)  # [batch_size * num_objects, num_classes]

    # Also concat the target labels and boxes
    tgt_ids = torch.cat([v['labels'] for v in labels]).long()

    # Compute the classification cost. Contrary to the loss, we don't use the NLL,
    # but approximate it in 1 - proba[target class].
    # The 1 is a constant that doesn't change the matching, it can be ommitted.
    cost_class = -out_prob[:, tgt_ids]

    cost_class = cost_class.view(batch_size, num_objects, -1).cpu()

    sizes = [len(v['labels']) for v in labels]
    indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_class.split(sizes, -1))]

    return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def normalize_angle(angle):
  if isinstance(angle, torch.Tensor):
    while not torch.all(angle >= 0):
      angle[angle < 0] += np.pi * 2
    while not torch.all(angle < np.pi * 2):
      angle[angle >= np.pi * 2] -= np.pi * 2
    return angle

  else:
    while not np.all(angle >= 0):
      angle[angle < 0] += np.pi * 2
    while not np.all(angle < np.pi * 2):
      angle[angle >= np.pi * 2] -= np.pi * 2

    return angle
