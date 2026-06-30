import copy
import numpy as np
import torch


def get_ped_action_code(pos, vel, acc, hdn):
    kMaxSpeedForStationary = 0.2                    # (m/s)
    kMaxDisplacementForStationary = 0.3             # (m)
    kMeanAcceleartionForXcelerate = 0.5             # (m/s^2)

    pos = pos - pos[0]
    pos = rotate(pos[:, 0], pos[:, 1], -hdn[0])

    disp = np.linalg.norm(pos[-1] - pos[0])
    max_speed = np.max(np.linalg.norm(vel, axis=1))
    mean_long_acc = np.mean(acc[:, 0])

    # stop: 3, decelerate: 4, keep speed: 5, accelerate: 6
    if (max_speed < kMaxSpeedForStationary) and (disp < kMaxDisplacementForStationary):
        return 1                        # stop
    else:
        if mean_long_acc > kMeanAcceleartionForXcelerate:
            return 7                    # accelerate
        elif mean_long_acc < -kMeanAcceleartionForXcelerate:
            return 5                    # decelerate
        else:
            return 6                    # keep speed


def get_veh_cyc_action_code(pos, vel, acc, hdn, type):
    if type == 8:
        kMaxSpeedForStationary = 2.0                # (m/s)
        kMaxDisplacementForStationary = 3.0         # (m)
    elif type == 12:
        kMaxSpeedForStationary = 1.0                # (m/s)
        kMaxDisplacementForStationary = 1.5         # (m)
    else:
        raise ValueError(f'Invalid object type: {type}')
    kMaxLateralDisplacementForStraight = 2.5        # (m)
    kMinLongitudinalDisplacementForUTurn = 0.0      # (m)
    kMaxAbsHeadingDiffForStraight = np.pi / 6.0     # (rad)
    kMeanAcceleartionForXcelerate = 0.5             # (m/s^2)

    pos = pos - pos[0]
    pos = rotate(pos[:, 0], pos[:, 1], -hdn[0])

    disp = np.linalg.norm(pos[-1] - pos[0])
    long_disp = pos[-1, 0] - pos[0, 0]
    lat_disp = pos[-1, 1] - pos[0, 1]
    max_speed = np.max(np.linalg.norm(vel, axis=1))
    mean_long_acc = np.mean(acc[:, 0])
    hdn_diff = hdn[-1] - hdn[0]

    # left U-intersection: 0, left intersection: 1, left lane-change / left through: 2,
    # stop: 3, decelerate: 4, keep speed: 5, accelerate: 6,
    # right lane-change/ right through: 7, right intersection: 8, right U-intersection: 9

    if (max_speed < kMaxSpeedForStationary) and (disp < kMaxDisplacementForStationary):
        return 1                            # stop

    if abs(hdn_diff) < kMaxAbsHeadingDiffForStraight:
        if abs(lat_disp) < kMaxLateralDisplacementForStraight:
            if mean_long_acc > kMeanAcceleartionForXcelerate:
                return 7                    # accelerate
            elif mean_long_acc < -kMeanAcceleartionForXcelerate:
                return 5                    # decelerate
            else:
                return 6                    # keep speed
        return 4 if lat_disp < 0 else 8     # left lane-change, right lane-change

    if lat_disp < 0:
        if long_disp < kMinLongitudinalDisplacementForUTurn:
            return 10                       # right U-intersection
        return 9                            # right intersection

    if long_disp < kMinLongitudinalDisplacementForUTurn:
        return 2                            # left U-intersection
    return 3                                # left intersection


def get_actions_code(config, scenario):
    object_mask = scenario['object_mask']
    pos = scenario['future_pos'][:, object_mask]
    vel = scenario['future_vel'][:, object_mask]
    acc = scenario['future_acc'][:, object_mask]
    hdn = scenario['future_hdn'][:, object_mask]
    act = scenario['future_act'][:, object_mask]
    mask = scenario['all_object_mask'][:, object_mask]
    type = scenario['object'][:, 10][object_mask]

    action_step = config['PREPROCESS']['ACTION_STEP']
    use_action_from_data = config['PREPROCESS']['USE_ACTION_FROM_DATA']
    traj_step, valid_object_num, _ = pos.shape
    object_num = len(object_mask)

    sample_rate = traj_step // action_step

    actions_trend = np.zeros((object_num, action_step))
    actions_type = np.zeros((object_num, action_step))


    for object_i in range(valid_object_num):
        for step_j in range(action_step):
            if use_action_from_data:
                # action trend [1: start, 2: keep, 3: finish]
                # action type [1: stop,
                #              2: U-turn, 3: left turn, 4: left lane-change,
                #              5: decelerate, 6: keep speed, 7: accelerate,
                #              8: right lane-change, 9: right turn]

                act_i_t = act[sample_rate*step_j: sample_rate*(step_j+1), object_i]
                unique_act_i_t = np.unique(act_i_t)

                if 0 in act_i_t:
                    action_trend = 0
                    action_type = 0
                else:
                    if len(unique_act_i_t) == 1:
                        action_trend = 2
                        action_type = unique_act_i_t[0]
                    else:
                        # [2: U-turn > 8: right lane-change > 4: left lane-change > 9: right turn > 3: left turn >
                        #  7: accelerate > 5: decelerate > 6: keep speed > 1: stop]

                        action_prior = [2, 8, 4, 9, 3, 7, 5, 6, 1]
                        if action_prior.index(act_i_t[0]) < action_prior.index(act_i_t[-1]):
                            action_trend = 3
                            action_type = act_i_t[0]
                        else:
                            action_trend = 1
                            action_type = act_i_t[-1]

                actions_trend[object_i, step_j] = action_trend
                actions_type[object_i, step_j] = action_type

            else:
                pos_i_t = pos[sample_rate*step_j: sample_rate*(step_j+1), object_i, :]
                vel_i_t = vel[sample_rate*step_j: sample_rate*(step_j+1), object_i, :]
                acc_i_t = acc[sample_rate*step_j: sample_rate*(step_j+1), object_i, :]
                hdn_i_t = hdn[sample_rate*step_j: sample_rate*(step_j+1), object_i]
                mask_i_t = mask[sample_rate*step_j: sample_rate*(step_j+1), object_i]
                type_i = type[object_i]

                if np.sum(mask_i_t) != len(mask_i_t):
                    action_type = 0
                else:
                    if type_i == 8:
                        action_type = get_veh_cyc_action_code(pos_i_t, vel_i_t, acc_i_t, hdn_i_t, type_i)
                    elif type_i == 12:
                        action_type = get_veh_cyc_action_code(pos_i_t, vel_i_t, acc_i_t, hdn_i_t, type_i)
                    elif type_i == 13:
                        action_type = get_ped_action_code(pos_i_t, vel_i_t, acc_i_t, hdn_i_t)
                    else:
                        raise ValueError(f'Invalid object type: {type_i}')

                actions_type[object_i, step_j] = action_type

    return actions_trend, actions_type


def get_speed_range_code(config, scenario):
    base_speed = config['BASE_SPEED']

    vel = scenario['object'][:, 2:4]
    speed = np.linalg.norm(vel, axis=1)

    speed_range = np.array([int(speed[i] // base_speed) for i in range(len(vel))])

    return speed_range


def get_rel_dist_code(config, scenario):
    base_dist = config['BASE_DIST']

    rel_pos = scenario['object'][:, :2]
    dist = np.linalg.norm(rel_pos, axis=1)

    rel_dist_range = np.array([int(dist[i] // base_dist) + 1 for i in range(len(rel_pos))])
    rel_dist_range[0] = 0

    return rel_dist_range


def get_rel_dir_code(scenario):
    hdn_diff = scenario['object'][:, 6]
    hdn_diff = np.mod(hdn_diff + np.pi, 2*np.pi) - np.pi

    rel_dir = np.array([0] * len(hdn_diff))

    for i in range(len(rel_dir)):
        if hdn_diff[i] == 0:
            pass
        else:
            if -3 * np.pi / 4 < hdn_diff[i] < -np.pi / 4:
                rel_dir[i] = 4  # perpendicular down
            elif -np.pi / 4 < hdn_diff[i] < np.pi / 4:
                rel_dir[i] = 1  # parallel same
            elif np.pi / 4 < hdn_diff[i] < 3 * np.pi / 4:
                rel_dir[i] = 3  # perpendicular up
            else:
                rel_dir[i] = 2  # parallel opposite

    rel_dir[0] = 0

    return rel_dir


def get_rel_pos_code(scenario):
    x, y = scenario['object'][:, 0], scenario['object'][:, 1]
    rel_deg = np.arctan2(y, x)

    rel_pos = np.array([0] * len(rel_deg))

    for i in range(len(rel_deg)):
        if rel_deg[i] == 0:
            pass
        else:
            if -np.pi / 9 < rel_deg[i] < np.pi / 9:
                rel_pos[i] = 1  # front
            elif -np.pi / 2 < rel_deg[i] < -np.pi / 9:
                rel_pos[i] = 2  # front right
            elif -8 * np.pi / 9 < rel_deg[i] < -np.pi / 2:
                rel_pos[i] = 3  # back right
            elif np.pi / 9 < rel_deg[i] < np.pi / 2:
                rel_pos[i] = 6  # front left
            elif np.pi / 2 < rel_deg[i] < 8 * np.pi / 9:
                rel_pos[i] = 5  # back left
            else:
                rel_pos[i] = 4  # back

    rel_pos[0] = 0  # ego car

    return rel_pos


def get_obj_type_code(scenario):
    obj_type = copy.deepcopy(scenario['object'][:, 10])
    for i in range(len(obj_type)):
        if obj_type[i] == 0:
            pass
        else:
            if obj_type[i] == 8:
                obj_type[i] = 3
            elif obj_type[i] == 12:
                obj_type[i] = 2
            elif obj_type[i] == 13:
                obj_type[i] = 1
            else:
                raise ValueError(f'Invalid object type: {obj_type[i]}')

    obj_type[0] = 4

    return obj_type


def get_size_range_code(config, scenario):
    base_length= config['BASE_LENGTH']

    # size = scenario['object'][:, 7:9]
    # size_range = np.array([int(size[i] // base_size) for i in range(len(size))])

    length = scenario['object'][:, 7]
    length_range = np.array([int(length[i] // base_length) for i in range(len(length))])

    # size_code = np.array([0] * len(size))
    # for i in range(len(size)):
    #     if size[i, 0] == 0:
    #         pass
    #     else:
    #         if size[i, 0] <= 6:
    #             size_code[i] = 1
    #         elif 6 < size[i, 0] <= 12:
    #             size_code[i] = 2
    #         else:
    #             size_code[i] = 3

    return length_range

def get_category_text(config, scenario):
    only_language_input = config['MODE']['ONLY_LANGUAGE_INPUT']
    object_range = config['OBJECT_RANGE']
    split_object_input = config['DATA'].get('SPLIT_OBJECT_INPUT', False)

    obj_type = get_obj_type_code(scenario)
    if only_language_input: # only text
        rel_pos = get_rel_pos_code(scenario)
        rel_dir = get_rel_dir_code(scenario)
        rel_dist_range = get_rel_dist_code(config, scenario)
    else:   # text + user
        rel_pos_x = scenario['object'][:, 0] / object_range
        rel_pos_y = scenario['object'][:, 1] / object_range
        rel_hdn_sin = np.sin(scenario['object'][:, 6])  # output [-1, 1]
        rel_hdn_cos = np.cos(scenario['object'][:, 6])  # output [-1, 1]
        # rel_hdn = np.mod(scenario['object'][:, 6] / object_range, 2 * np.pi) / (2 * np.pi)
    speed_range = get_speed_range_code(config, scenario)
    size_range = get_size_range_code(config, scenario)
    actions_trend, actions_type = get_actions_code(config, scenario)

    if only_language_input:
        object_code = np.column_stack([obj_type, rel_pos, rel_dir, rel_dist_range, speed_range, size_range, actions_trend, actions_type])
    else:
        object_code = np.column_stack([rel_pos_x, rel_pos_y, rel_hdn_sin, rel_hdn_cos, obj_type, speed_range, size_range, actions_trend, actions_type])

    # ablation_variable_idx = config['DATA']['ABLATION_VARIABLE_IDX']
    # object_code = object_code[ablation_variable_idx+1:]
    # object_code = torch.cat([object_code[..., :ablation_variable_idx], object_code[..., ablation_variable_idx+1:]], dim=-1)

    start_idx = 0
    # sort objects according to distance
    object = scenario['object']
    object_mask = scenario['object_mask']
    object[~object_mask, :2] = 1e6

    dist = np.linalg.norm(object[start_idx:, :2], axis=1)
    sort_idx = np.argsort(dist)
    object_code_ = object_code[sort_idx]

    if split_object_input and not only_language_input:
        # Split into object_pose (normalized continuous) and object_code (categorical/discrete)
        object_pose = np.column_stack([rel_pos_x, rel_pos_y, rel_hdn_sin, rel_hdn_cos])
        object_code_only = np.column_stack([obj_type, speed_range, size_range, actions_trend, actions_type])
        # Sort both by distance
        object_pose_ = object_pose[sort_idx]
        object_code_only_ = object_code_only[sort_idx]
        return object_code_, object_pose_, object_code_only_

    return object_code_, None, None


def get_object_code(config, data):
    object_code, object_pose, object_code_split = get_category_text(config, data)
    data['object_code'] = object_code
    if object_pose is not None:
        data['object_pose'] = object_pose
    if object_code_split is not None:
        data['object_code_split'] = object_code_split




def cal_rel_dir(dir1, dir2):
    return np.mod(dir1 - dir2 + np.pi, 2 * np.pi) - np.pi


def get_traj(scenario):
    # rotate traj of each vehicle to the vehicle coordinate
    traj = scenario['gt_pos']
    traj = traj - traj[[0], :]
    init_hdn = scenario['gt_hdn'][0]
    traj = rotate(traj[..., 0], traj[..., 1], -init_hdn)

    return traj


def get_future_heading(scenario):
    future_hdn = copy.deepcopy(scenario['gt_hdn'])
    future_hdn = cal_rel_dir(future_hdn, scenario['gt_hdn'][0])

    return future_hdn


def get_gt(config, scenario):
    lane_num = scenario['lane'].shape[1]

    lane, object_lane_index, lane_based_rep, bbox, pos = (
        scenario['map'][:, :lane_num], scenario['object_lane_index'], scenario['lane_based_rep'],scenario['object'][..., 7:9], scenario['object_'][..., :2]
    )

    timestep, lane_num, _ = lane.shape
    gt_distribution = np.zeros([timestep, lane_num])
    gt_lane_based_coord = np.zeros([timestep, lane_num, 5])
    gt_bbox = np.zeros([timestep, lane_num, 2])
    for i in range(timestep):
        mask = scenario['object_mask'][i]
        index = object_lane_index[i][mask].astype(int)
        gt_distribution[i][index] = 1
        gt_lane_based_coord[i, index] = lane_based_rep[i, mask, :5]
        gt_bbox[i, index] = bbox[i, mask]
    scenario['gt_lane_index'] = object_lane_index
    scenario['gt_pos'] = pos
    scenario['gt_bbox'] = gt_bbox
    scenario['gt_distribution'] = gt_distribution
    scenario['gt_long_lat'] = gt_lane_based_coord[..., :2]
    scenario['gt_speed'] = gt_lane_based_coord[..., 2]
    scenario['gt_speed_hdn_rel_dir'] = gt_lane_based_coord[..., 3]
    scenario['gt_hdn_lane_rel_dir'] = gt_lane_based_coord[..., 4]
    scenario['gt_hdn'] = scenario['object_'][..., 6]
    scenario['future_pos'] = get_traj(scenario)
    scenario['future_hdn'] = get_future_heading(scenario)
    scenario['future_vel'] = scenario['object'][..., 2:4]
    scenario['future_acc'] = scenario['object'][..., 4:6]
    scenario['future_act'] = scenario['object'][..., 9]
    # scenario['future_lon_disp'] = lane_based_rep[..., 0]
    scenario['future_lat_disp'] = lane_based_rep[..., 1]

    return scenario


def process_map_inp(config, scenario):
    MAP_RANGE = config['MAP_RANGE']
    lane = copy.deepcopy(scenario['lane'])
    lane[..., :4] /= MAP_RANGE
    line = copy.deepcopy(scenario['line'])
    line[..., :4] /= MAP_RANGE
    edge = copy.deepcopy(scenario['edge'])
    edge[..., :4] /= MAP_RANGE
    other = copy.deepcopy(scenario['other'])
    other[..., :4] /= MAP_RANGE

    scenario['map'] = np.concatenate([lane, line, edge, other], axis=1)
    scenario['map_mask'] = np.concatenate([scenario['lane_mask'], scenario['line_mask'], scenario['edge_mask'], scenario['other_mask']], axis=1)

    return scenario


def get_lane_based_rep(config, scenario):
    THRES = config['THRES']
    MAX_OBJECT_NUM = config['MAX_OBJECT_NUM']

    object = scenario['object_']
    _object = scenario['object']
    object_mask = scenario['object_mask']

    lane = scenario['lane']
    lane_mask = scenario['lane_mask']

    lane_x = (lane[..., 0] + lane[..., 2]) / 2
    lane_y = (lane[..., 1] + lane[..., 3]) / 2

    object_x = object[..., 0]
    object_y = object[..., 1]

    timestep, lane_num = lane_x.shape
    _, object_num = object_x.shape

    lane_x = np.repeat(lane_x.reshape(timestep, -1, lane_num), axis=1, repeats=object_num)
    lane_y = np.repeat(lane_y.reshape(timestep, -1, lane_num), axis=1, repeats=object_num)

    object_x = np.repeat(object_x.reshape(timestep, object_num, -1), axis=-1, repeats=lane_num)
    object_y = np.repeat(object_y.reshape(timestep, object_num, -1), axis=-1, repeats=lane_num)

    dist = np.sqrt((lane_x - object_x) ** 2 + (lane_y - object_y) ** 2)

    lane_mask = np.repeat(lane_mask.reshape(timestep, -1, lane_num), axis=1, repeats=object_num)

    dist[lane_mask == 0] = 10e5

    lane_index = np.argmin(dist, -1)
    min_dist_to_lane = np.min(dist, -1)
    min_dist_mask = min_dist_to_lane < THRES

    # Find the lane segment that is closest to each object at each time step.
    selected_lane = np.take_along_axis(lane, lane_index.reshape(timestep, object_num, -1), axis=1)

    # object_ = scenario['object_']
    # object_x_ = object_[..., 0]
    # object_y_ = object_[..., 1]
    # object_x_ = np.repeat(object_x_.reshape(timestep, object_num, -1), axis=-1, repeats=lane_num)
    # object_y_ = np.repeat(object_y_.reshape(timestep, object_num, -1), axis=-1, repeats=lane_num)
    # dist_ = np.sqrt((lane_x - object_x_) ** 2 + (lane_y - object_y_) ** 2)
    # dist_[lane_mask == 0] = 10e5
    # lane_index_ = np.argmin(dist_, -1)
    # min_dist_to_lane_ = np.min(dist_, -1)
    # min_dist_mask_ = min_dist_to_lane_ < THRES
    # selected_lane_ = np.take_along_axis(lane, lane_index_.reshape(timestep, object_num, -1), axis=1)

    vx, vy = object[..., 2], object[..., 3]
    speed = np.sqrt(vx ** 2 + vy ** 2)
    low_speed = speed < 0.1

    speed_dir = np.arctan2(vy, vx)
    x1, y1, x2, y2 = selected_lane[..., 0], selected_lane[..., 1], selected_lane[..., 2], selected_lane[..., 3]
    lane_dir = np.arctan2(y2 - y1, x2 - x1)
    hdn_dir = object[..., 6]

    speed_hdn_rel_dir = cal_rel_dir(speed_dir, hdn_dir)
    hdn_lane_rel_dir = cal_rel_dir(hdn_dir, lane_dir)

    # low speed lead to wrong speed direction, remove it
    speed_hdn_rel_dir[low_speed] = 0

    speed_hdn_rel_dir_mask = np.abs(speed_hdn_rel_dir) < np.pi / 6
    hdn_lane_rel_dir_mask = np.abs(hdn_lane_rel_dir) < np.pi / 4

    object_x = object[..., 0]
    object_y = object[..., 1]
    lane_x = (x1 + x2) / 2
    lane_y = (y1 + y2) / 2

    object_to_lane_x = object_x - lane_x
    object_to_lane_y = object_y - lane_y

    coord = rotate(object_to_lane_x, object_to_lane_y, np.pi / 2 - lane_dir)

    lane_len = 5
    # lane_len = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # lane_len = np.clip(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), a_min=4.5, a_max=5.5)

    # Lateral and longitudinal position percentage of the object relative to the lane segment center
    lat_perc = np.clip(coord[..., 0], a_min=-lane_len / 2, a_max=lane_len / 2) / lane_len
    long_perc = np.clip(coord[..., 1], a_min=-lane_len / 2, a_max=lane_len / 2) / lane_len

    # ignore other masks for future agents (to support out-of-range agent prediction)
    total_mask = object_mask
    # for the first frame, use all masks to filter out off-road agents
    # total_mask[0, :] = (min_dist_mask * object_mask * speed_hdn_rel_dir_mask * hdn_lane_rel_dir_mask)[0, :]

    # av is always valid
    total_mask[:, 0] = 1
    total_mask = total_mask.astype(bool)

    timestep, object_num, object_dim = object.shape
    object_ = np.zeros((timestep, MAX_OBJECT_NUM, object_dim))
    _object_ = np.zeros((timestep, MAX_OBJECT_NUM, object_dim))
    object_mask_ = np.zeros((timestep, MAX_OBJECT_NUM)).astype(bool)

    # the_lane = np.take_along_axis(lane, )
    info = np.concatenate([
        lane_index.reshape(timestep, object_num, -1),
        long_perc.reshape(timestep, object_num, -1),
        lat_perc.reshape(timestep, object_num, -1),
        speed.reshape(timestep, object_num, -1),
        speed_hdn_rel_dir.reshape(timestep, object_num, -1),
        hdn_lane_rel_dir.reshape(timestep, object_num, -1),
        selected_lane
    ], -1)
    info_ = np.zeros([timestep, MAX_OBJECT_NUM, info.shape[-1]])

    start_mask = total_mask[0]
    for i in range(timestep):
        object_i = object[i][start_mask]
        _object_i = _object[i][start_mask]
        info_i = info[i][start_mask]

        step_mask = total_mask[i]
        valid_mask = step_mask[start_mask]

        object_i = object_i[:MAX_OBJECT_NUM]
        _object_i = _object_i[:MAX_OBJECT_NUM]
        info_i = info_i[:MAX_OBJECT_NUM]

        valid_num = object_i.shape[0]
        object_i = np.pad(object_i, ((0, MAX_OBJECT_NUM-valid_num), (0, 0)))
        _object_i = np.pad(_object_i, ((0, MAX_OBJECT_NUM-valid_num), (0, 0)))
        info_i = np.pad(info_i, ((0, MAX_OBJECT_NUM-valid_num), (0, 0)))

        object_[i] = object_i
        _object_[i] = _object_i
        info_[i] = info_i
        object_mask_[i, :valid_num] = valid_mask[:valid_num]

    scenario['lane_based_rep'] = info_[..., 1:]     # [long percent, lat percent, speed, speed_hdn_rel_dir, hdn_lane_rel_dir, selected_lane (dim5-10)]
    scenario['object_lane_index'] = info_[..., 0].astype(int)
    scenario['object'] = object_
    scenario['object_'] = _object_
    scenario['object_mask'] = object_mask_

    return scenario


def process_map_element(map_element, MAP_ELEMENT_NUM, MAP_RANGE):
    # map element: lane, line, edge, other
    map_element_mask = (abs(map_element[..., 0]) < MAP_RANGE) * (abs(map_element[..., 1]) < MAP_RANGE)

    map_element_id = np.unique(map_element[..., -2])

    vec_ls = list()
    vec_mask_ls = list()
    vec_id_ls = list()
    timestep, _, map_dim = map_element.shape
    vec_dim = 6
    for id in map_element_id:
        id_set = map_element[..., -2] == id
        # po: polyline or polygon or point
        po = map_element[id_set].reshape(timestep, -1, map_dim)
        po_mask = map_element_mask[id_set].reshape(timestep, -1)

        vector_id = np.ones((timestep, po.shape[1]-1, 1)) * id
        vector = np.zeros((timestep, po.shape[1]-1, vec_dim))

        vector[..., :2] = po[:, :-1, :2]    # start points
        vector[..., 2:4] = po[:, 1:, :2]    # end points
        vector[..., 4] = po[:, 1:, 2]       # type
        vector[..., 5] = po[:, 1:, 4]       # traffic light
        vector_mask = po_mask[:, :-1] * po_mask[:, 1:]
        vector[vector_mask == 0] = 0
        vec_ls.append(vector)
        vec_mask_ls.append(vector_mask)
        vec_id_ls.append(vector_id)

    vector = np.concatenate(vec_ls, axis=1) if vec_ls else np.zeros((timestep, 0, vec_dim))
    vector_mask = np.concatenate(vec_mask_ls, axis=1) if vec_mask_ls else np.zeros((timestep, 0), dtype=bool)
    vector_id = np.concatenate(vec_id_ls, axis=1) if vec_id_ls else np.zeros((timestep, 0, 1))

    all_vector = np.zeros([timestep, MAP_ELEMENT_NUM, vec_dim])
    all_vector_mask = np.zeros([timestep, MAP_ELEMENT_NUM])
    all_vector_id = np.zeros([timestep, MAP_ELEMENT_NUM, 1])

    # sort by AV-to-vector distance
    for t in range(timestep):
        vector_mask_t = vector_mask[t]

        # valid vector
        vector_t = vector[t][vector_mask_t]
        vector_id_t = vector_id[t][vector_mask_t]

        dist2 = ((vector_t[..., 0] + vector_t[..., 2]) / 2) ** 2 + ((vector_t[..., 1] + vector_t[..., 3]) / 2) ** 2
        idx = np.argsort(dist2)

        vector_t = vector_t[idx]
        vector_mask_t = np.ones(vector_t.shape[0])
        vector_id_t = vector_id_t[idx]

        vector_t = vector_t[:MAP_ELEMENT_NUM]
        vector_mask_t = vector_mask_t[:MAP_ELEMENT_NUM]
        vector_id_t = vector_id_t[:MAP_ELEMENT_NUM]

        vector_t = np.pad(vector_t, ([0, MAP_ELEMENT_NUM-vector_t.shape[0]], [0, 0]))
        vector_mask_t = np.pad(vector_mask_t, ([0, MAP_ELEMENT_NUM-vector_mask_t.shape[0]]))
        vector_id_t = np.pad(vector_id_t, ([0, MAP_ELEMENT_NUM-vector_id_t.shape[0]], [0, 0]))

        all_vector[t] = vector_t
        all_vector_mask[t] = vector_mask_t
        all_vector_id[t] = vector_id_t

    return all_vector, all_vector_mask.astype(bool), all_vector_id.astype(int)


def get_control_lane_state_code(state):
    state_ls = list()
    colors = state['color']
    for color in colors:
        if color == 'RED':
            state_ls.append(0)
        elif color == 'YELLOW':
            state_ls.append(1)
        elif color == 'GREEN':
            state_ls.append(2)
        elif color == 'FLASHING':
            state_ls.append(3)
        elif color == 'UNKNOWN':
            state_ls.append(4)
        elif color is None:
            state_ls.append(5)
        else:
            raise ValueError(f"Unknown color: {color}")

    return np.array(state_ls)


def process_map(static_map, dynamic_map_states, LANE_NUM, LINE_NUM, EDGE_NUM, OTHER_NUM, MAP_RANGE, MAX_TIMESTEP):
    map = np.zeros((*static_map.shape[:-1], 5))
    map[..., :4] = static_map

    static_map_id = static_map[..., -1]

    timestep = static_map.shape[0]

    if dynamic_map_states:
        for dynamic_map_id in dynamic_map_states:
            dynamic_map_state = dynamic_map_states[dynamic_map_id]

            control_lane_id = dynamic_map_state['id']
            control_lane_state = get_control_lane_state_code(dynamic_map_state['state'])[0: MAX_TIMESTEP]

            idx1, idx2 = np.where(static_map_id == int(control_lane_id))
            map[idx1, idx2, -1] = np.repeat([control_lane_state], len(idx1)/timestep, axis=1)

    map_type = map[0, :, 2]
    lane_ind = (map_type == 18) + (map_type == 19) + (map_type == 20) + (map_type == 21) + (map_type == 22) + (map_type == 23) + \
               (map_type == 24) + (map_type == 25) + (map_type == 26)
    line_ind = (map_type == 0) + (map_type == 1) + (map_type == 2) + (map_type == 3) + (map_type == 4) + (map_type == 5) + \
               (map_type == 6) + (map_type == 7) + (map_type == 8) + (map_type == 9) + (map_type == 10) + (map_type == 11) + \
               (map_type == 12) + (map_type == 13) + (map_type == 14) + (map_type == 15) + (map_type == 16) + (map_type == 17)
    edge_ind = (map_type == 27) + (map_type == 28) + (map_type == 29)
    other_ind = (map_type == 30) + (map_type == 31) + (map_type == 32) + (map_type == 33)

    lane, lane_mask, lane_id = process_map_element(map[:, lane_ind], LANE_NUM, MAP_RANGE)
    line, line_mask, _ = process_map_element(map[:, line_ind], LINE_NUM, MAP_RANGE)
    edge, edge_mask, _ = process_map_element(map[:, edge_ind], EDGE_NUM, MAP_RANGE)
    other, other_mask, _ = process_map_element(map[:, other_ind], OTHER_NUM, MAP_RANGE)

    return lane, lane_mask, lane_id, line, line_mask, edge, edge_mask, other, other_mask


def process_object(object, OBJECT_RANGE):
    # "object" is based on ego coordinate's first frame
    # "object_" is object_based on ego coordinate's all frame

    object_ = copy.deepcopy(object)

    timestep, _, _ = object.shape

    ego_track = object[:, 0]
    ego_pos = ego_track[0, :2].reshape(1, 1, -1)
    ego_hdn = ego_track[[0], [6]]

    ego_track_ = object_[:, 0]
    ego_pos_ = ego_track_[:, :2].reshape(timestep, 1,  -1)
    ego_hdn_ = ego_track_[:, [6]]

    object[..., :2] -= ego_pos
    object[..., :2] = rotate(object[..., 0], object[..., 1], -ego_hdn)
    object[..., 6] -= ego_hdn

    object_[..., :2] -= ego_pos_
    object_[..., :2] = rotate(object_[..., 0], object_[..., 1], -ego_hdn_)
    object_[..., 6] -= ego_hdn_

    object_valid_mask = object[..., -1]
    # is_vehicle_mask = object[..., 9] <= 7
    object_range_mask = (abs(object[..., 0]) < OBJECT_RANGE) * (abs(object[..., 1]) < OBJECT_RANGE)

    # use object range mask only for the first frame allow object to be out of range in the future frames
    object_valid_mask[0, :] *= object_range_mask[0, :]
    object_mask = object_valid_mask.astype(bool)

    return object, object_, object_mask # is_vehicle_mask


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


def transform_object_coordinate(object_tracks, MAX_RANGE):
    ego_track = object_tracks[:, 0]
    ego_pos = ego_track[0, :2].reshape(1, 1, -1)
    ego_hdn = ego_track[[0], [2]]

    object_tracks[..., :2] -= ego_pos
    object_tracks[..., :2] = rotate(object_tracks[..., 0], object_tracks[..., 1], -ego_hdn)
    object_tracks[..., 4] -= ego_hdn

    object_valid_mask = object_tracks[..., -1]
    object_range_mask = (abs(object_tracks[..., 0]) < MAX_RANGE) * (abs(object_tracks[..., 1]) < MAX_RANGE)
    object_mask = object_valid_mask * object_range_mask

    return object_tracks, object_mask


def transform_map_coordinate(object_tracks, map_features):
    # every frame is different
    timestep = object_tracks.shape[0]

    ego_track = object_tracks[:, 0]
    ego_pos = ego_track[:, :2].reshape(timestep, 1,  -1)

    map_features = map_features.reshape(1, map_features.shape[0], map_features.shape[1])
    map_features = np.repeat(map_features, timestep, axis=0)
    map_features[..., :2] -= ego_pos

    ego_x = map_features[..., 0]
    ego_y = map_features[..., 1]
    ego_heading = ego_track[:, [6]]

    map_features[..., :2] = rotate(ego_x, ego_y, -ego_heading)

    return map_features


def polyline_interpolate(xyz: np.ndarray, step: float, method: int) -> np.ndarray:
    """
    Interpolate points based on cumulative distances from the first one. Two modes are available:
    INTER_METER: interpolate using step as a meter value over cumulative distances (variable len result-v2.0-v4)
    INTER_ENSURE_LEN: interpolate using a variable step such that we always get step values
    Args:
        xyz (np.ndarray): XYZ coords
        step (float): param for the interpolation
        method (InterpolationMethod): method to use to interpolate

    Returns:
        np.ndarray: the new interpolated coordinates
    """
    cum_dist = np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=-1))
    cum_dist = np.insert(cum_dist, 0, 0)

    if method == 1:
        step = int(step)
        assert step > 1, "step must be at least 2 with INTER_ENSURE_LEN"
        steps = np.linspace(cum_dist[0], cum_dist[-1], step)

    elif method == 0:
        assert step > 0, "step must be greater than 0 with INTER_FIXED"
        steps = np.arange(cum_dist[0], cum_dist[-1], step)
    else:
        raise NotImplementedError(f"interpolation method should be in 0 or 1")

    xyz_inter = np.empty((len(steps), 3), dtype=xyz.dtype)
    xyz_inter[:, 0] = np.interp(steps, xp=cum_dist, fp=xyz[:, 0])
    xyz_inter[:, 1] = np.interp(steps, xp=cum_dist, fp=xyz[:, 1])
    xyz_inter[:, 2] = np.interp(steps, xp=cum_dist, fp=xyz[:, 2])

    xyz_inter = np.vstack([xyz_inter, xyz[-1, :]])

    return xyz_inter


def get_map_type_code(map_type, map_subtype, map_length):
    road_line_subtype_dict = {
        'DASHED_WHITE': 0, 'DASHED_YELLOW': 1, 'DASHED_UNKNOWN': 2,
        'SOLID_WHITE': 3, 'SOLID_YELLOW': 4, 'SOLID_BLUE': 5, 'SOLID_UNKNOWN': 6,
        'DOUBLE_DASH_WHITE': 7, 'DOUBLE_DASH_YELLOW': 8, 'DOUBLE_SOLID_WHITE': 9, 'DOUBLE_SOLID_YELLOW': 10,
        'DOUBLE_SOLID_UNKNOWN': 11,
        'DASH_SOLID_WHITE': 12, 'DASH_SOLID_YELLOW': 13,
        'SOLID_DASH_WHITE': 14, 'SOLID_DASH_YELLOW': 15,
        'VIRTUAL': 16, 'UNKNOWN': 17
    }
    road_lane_subtype_dict = {
        'URBAN_ALL': 18, 'URBAN_VEHICLE': 19, 'URBAN_BUS': 20, 'URBAN_BIKE': 21, 'URBAN_PEDESTRIAN': 22,
        'URBAN_UNKNOWN': 23,
        'HIGHWAY': 24, 'FREEWAY': 25, 'UNKNOWN': 26
    }
    road_edge_subtype_dict = {
        'BOUNDARY': 27, 'MEDIAN': 28, 'UNKNOWN': 29
    }
    other_subtype_dict = {
        'CROSSWALK': 30, 'SPEED_BUMP': 31, 'WALKWAY': 32, 'STOP_SIGN': 33
    }
    if map_type == 'ROAD_LINE':
        return np.array([road_line_subtype_dict[map_subtype]] * map_length)
    elif map_type == 'ROAD_LANE':
        return np.array([road_lane_subtype_dict[map_subtype]] * map_length)
    elif map_type == 'ROAD_EDGE':
        return np.array([road_edge_subtype_dict[map_subtype]] * map_length)
    elif map_type == 'OTHER':
        return np.array([other_subtype_dict[map_subtype]] * map_length)
    else:
        raise ValueError(f"Unknown map type: {map_type}")


def get_object_type_code(object_type, object_subtype, object_valid):
    scenario_length = len(object_valid)

    vehicle_subtype_dict = {'CAR': 1, 'TRUCK': 2, 'VAN': 3, 'BUS': 4, 'TRAM': 5, 'TRAILER': 6, 'EMERGENCY': 7,
                            'UNKNOWN': 8}
    cyclist_subtype_dict = {'BICYCLIST': 9, 'TRICYCLIST': 10, 'MOTORCYCLIST': 11, 'UNKNOWN': 12}
    pedestrian_subtype_dict = {'PEDESTRIAN': 13}
    animal_subtype_dict = {'ANIMAL': 14}
    static_subtype_dict = {'BARRIER': 15, 'TRAFFIC_CONE': 16, 'CONSTRUCTION': 17, 'UNKNOWN': 18}
    unknown_subtype_dict = {'UNKNOWN': 19}

    if object_type == 'VEHICLE':
        return np.array([vehicle_subtype_dict[object_subtype]] * scenario_length)
    elif object_type == 'CYCLIST':
        return np.array([cyclist_subtype_dict[object_subtype]] * scenario_length)
    elif object_type == 'PEDESTRIAN':
        return np.array([pedestrian_subtype_dict[object_subtype]] * scenario_length)
    elif object_type == 'ANIMAL':
        return np.array([animal_subtype_dict[object_subtype]] * scenario_length)
    elif object_type == 'STATIC':
        return np.array([static_subtype_dict[object_subtype]] * scenario_length)
    elif object_type == 'UNKNOWN':
        return np.array([unknown_subtype_dict[object_subtype]] * scenario_length)
    else:
        raise ValueError(f"Unknown object type: {object_type}")


def process(config, scenario):
    data = dict()

    object_tracks = scenario['object_tracks']
    map_features = scenario['map_features']
    dynamic_map_states = scenario['dynamic_map_states']
    extra_information = scenario['extra_information']

    av_id = extra_information['sdc_id']

    scenario_length = len(object_tracks[av_id]['state']['valid'])

    object_tracks_array = np.empty((0, scenario_length, 12))
    # av track first
    object_type = get_object_type_code(
        object_tracks[av_id]['type'],
        object_tracks[av_id]['subtype'],
        object_tracks[av_id]['state']['valid']
    )

    object_track_array = np.column_stack([object_tracks[av_id]['state']['global_center'][:, :2],
                                          object_tracks[av_id]['state']['local_velocity'],
                                          object_tracks[av_id]['state']['local_acceleration'],
                                          object_tracks[av_id]['state']['heading'],
                                          object_tracks[av_id]['state']['size'][:, :2],
                                          object_tracks[av_id]['state'].get('action', np.zeros(scenario_length)),
                                          object_type,
                                          object_tracks[av_id]['state']['valid']]).reshape(1, scenario_length, -1)
    object_tracks_array = np.row_stack([object_tracks_array, object_track_array])
    for object_id in object_tracks:
        if object_id != av_id:
            object_track = object_tracks[object_id]
            object_state = object_track['state']

            object_type = get_object_type_code(
                object_track['type'],
                object_track['subtype'],
                object_state['valid']
            )

            object_track_array = np.column_stack([object_state['global_center'][:, :2],
                                                  object_state['local_velocity'],
                                                  object_state['local_acceleration'],
                                                  object_state['heading'],
                                                  object_state['size'][:, :2],
                                                  object_state.get('action', np.zeros(scenario_length)),
                                                  object_type,
                                                  object_state['valid']]).reshape(1, scenario_length, -1)
            object_tracks_array = np.row_stack([object_tracks_array, object_track_array])

    object_tracks_array = object_tracks_array.transpose(1, 0, 2)

    map_preprocess = config['PREPROCESS']['MAP']

    map_features_array = np.empty((0, 4))
    for map_id in map_features:
        map_feature = map_features[map_id]
        if 'polyline' in map_feature.keys():
            map_po = polyline_interpolate(map_feature['polyline'], 1, 0)[:, 0:2] if map_preprocess else map_feature['polyline'][:, 0:2]
            map_length = map_po.shape[0]
            map_type = get_map_type_code(map_feature['type'], map_feature['subtype'], map_length)
            map_feature_array = np.column_stack((map_po, map_type, [int(map_id)] * map_length))
        elif 'polygon' in map_feature.keys():
            map_po = map_feature['polygon'][:, 0:2] if map_preprocess else map_feature['polygon'][:, 0:2]
            map_length = map_po.shape[0]
            map_type = get_map_type_code(map_feature['type'], map_feature['subtype'], map_length)
            map_feature_array = np.column_stack((map_po, map_type, [int(map_id)] * map_length))
        elif 'position' in map_feature.keys():
            map_po = map_feature['position'][0:2].reshape(1, -1)
            map_length = map_po.shape[0]
            map_type = get_map_type_code(map_feature['type'], map_feature['subtype'], map_length)
            map_feature_array = np.column_stack((map_po, map_type, [int(map_id)] * map_length))
        else:
            raise ValueError(f"Unknown map feature key: {map_feature.keys()}")

        map_features_array = np.row_stack([map_features_array, map_feature_array])

    MAX_TIMESTEP = config['MAX_TIMESTEP']

    OBJECT_RANGE = config['OBJECT_RANGE']
    MAP_RANGE = config['MAP_RANGE']

    LANE_NUM = config['LANE_NUM']
    LINE_NUM = config['LINE_NUM']
    EDGE_NUM = config['EDGE_NUM']
    OTHER_NUM = config['OTHER_NUM']

    object_tracks_array = object_tracks_array[0: MAX_TIMESTEP]

    data['static_map'] = transform_map_coordinate(object_tracks_array, map_features_array)

    scenario = dict()

    scenario['object'], scenario['object_'], scenario['object_mask'] = process_object(
        object_tracks_array,
        OBJECT_RANGE
    )
    (scenario['lane'], scenario['lane_mask'], scenario['lane_id'],
     scenario['line'], scenario['line_mask'],
     scenario['edge'], scenario['edge_mask'],
     scenario['other'], scenario['other_mask']) = process_map(
        data['static_map'],
        dynamic_map_states,
        LANE_NUM, LINE_NUM, EDGE_NUM, OTHER_NUM, MAP_RANGE, MAX_TIMESTEP
    )

    scenario = get_lane_based_rep(config, scenario)
    scenario = process_map_inp(config, scenario)
    scenario = get_gt(config, scenario)

    num_timestep = scenario['object'].shape[0]

    scenario_ls = []
    for i in range(num_timestep):
        dic = {}
        for k, v in scenario.items():
            dic[k] = v[i]
        scenario_ls.append(dic)

    future_attrs = ['gt_pos', 'gt_lane_index', 'future_pos', 'future_hdn', 'future_vel', 'future_acc', 'future_act', 'future_lat_disp', 'lane_based_rep']
    for attr in future_attrs:
        scenario_ls[0][attr] = scenario[attr]
    scenario_ls[0]['all_object_mask'] = scenario['object_mask']

    return scenario_ls[0]
