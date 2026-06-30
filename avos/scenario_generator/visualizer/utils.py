import copy
import cv2
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random

from matplotlib.patches import Rectangle, Polygon
from matplotlib.transforms import Affine2D
from scipy.interpolate import interp1d
from tqdm import tqdm


AGENT_LABEL_FONT_SIZE = 5
AGENT_LABEL_ID_OFFSET = 1.5
AGENT_LABEL_SPEED_OFFSET = 0.5
AGENT_LABEL_LON_ACC_OFFSET = -0.5
AGENT_LABEL_LAT_ACC_OFFSET = -1.5
AGENT_LABEL_ACTION_OFFSET = -2.5
AGENT_LABEL_VISIBLE_RANGE = 50.0
ACTION_LABELS = {
    0: "-",
    1: "stop",
    2: "U-turn",
    3: "L-turn",
    4: "L-change",
    5: "decel",
    6: "keep",
    7: "accel",
    8: "R-change",
    9: "R-turn",
    10: "R-U-turn",
}


def is_vehicle_agent(object_type):
    return str(object_type).upper() == 'VEHICLE'


def action_label_at_frame(state, frame_index):
    action = state.get('action')
    if action is None or frame_index >= len(action):
        return "-"
    try:
        if not np.isfinite(action[frame_index]):
            return "-"
        return ACTION_LABELS.get(int(action[frame_index]), "-")
    except (TypeError, ValueError):
        return "-"


class ScenarioVisualization(object):
    def __init__(self, ts, sdc_id):
        self.ts = ts
        self.sdc_id = sdc_id

    def rotate(self, x, y, angle):
        other_x_trans = np.cos(angle) * x - np.sin(angle) * y
        other_y_trans = np.cos(angle) * y + np.sin(angle) * x
        output_coords = np.stack((other_x_trans, other_y_trans), axis=-1)

        return output_coords

    def preprocess(self, data):
        object_tracks = data['object_tracks']
        map_features = data['map_features']
        dynamic_map_states = data['dynamic_map_states']

        sdc_p0 = copy.deepcopy(object_tracks[self.sdc_id]['state']['global_center'][0, :])
        sdc_h0 = copy.deepcopy(object_tracks[self.sdc_id]['state']['heading'][0])
        for object_id in object_tracks:
            object_track = object_tracks[object_id]
            object_state = object_track['state']

            p = object_state['global_center']
            valid = object_state['valid']

            p[valid, 0:2] -= sdc_p0[0:2]
            object_state['global_center'][valid, 0:2] = self.rotate(p[:, 0], p[:, 1], -sdc_h0)[valid]
            object_state['heading'][valid] -= sdc_h0

        for map_id in map_features:
            map_feature = map_features[map_id]
            if 'polyline' in list(map_feature.keys()):
                p = map_feature['polyline']
                p[:, 0:2] -= sdc_p0[0:2]
                map_feature['polyline'][:, 0:2] = self.rotate(p[:, 0], p[:, 1], -sdc_h0)
            elif 'polygon' in list(map_feature.keys()):
                p = map_feature['polygon']
                p[:, 0:2] -= sdc_p0[0:2]
                map_feature['polygon'][:, 0:2] = self.rotate(p[:, 0], p[:, 1], -sdc_h0)
            else:
                p = map_feature['position']
                p[0:2] -= sdc_p0[0:2]
                map_feature['position'][0:2] = self.rotate(p[0], p[1], -sdc_h0)

        for dynamic_map_id in dynamic_map_states:
            dynamic_map_state = dynamic_map_states[dynamic_map_id]
            p = dynamic_map_state['stop_point']
            p[0:2] -= sdc_p0[0:2]
            dynamic_map_state['stop_point'][0:2] = self.rotate(p[0], p[1], -sdc_h0)

        return data

    def plot_object_tracks_with_future_traj(self, data, ax, sdc_id, num_frame=50, text=False, plot=True):
        if not plot:
            return
        object_tracks = data['object_tracks']
        for object_id in object_tracks:
            object_track = object_tracks[object_id]

            object_state = object_track['state']
            object_type = object_track['type']

            valid = object_state['valid']
            if not valid[0]:
                continue

            p = object_state['global_center'][valid][0: num_frame]
            v = object_state['local_velocity'][valid][0: num_frame]
            a = object_state['local_acceleration'][valid][0: num_frame]
            h = object_state['heading'][valid][0: num_frame]
            s = object_state['size'][valid][0: num_frame]

            object_color = get_random_color(object_id)

            # plot init state
            ax.add_patch(Rectangle(
                xy=(p[0, 0] - 0.5 * s[0, 0], p[0, 1] - 0.5 * s[0, 1]),
                width=s[0, 0],
                height=s[0, 1],
                transform=mpl.transforms.Affine2D().rotate_deg_around(p[0, 0], p[0, 1], h[0] * 180 / np.pi) + ax.transData,
                alpha=0.8,
                linewidth=0.5,
                facecolor=object_color if object_id != sdc_id else 'red',
                edgecolor="black",
                label=object_type,
                zorder=1
            ))
            if object_type == 'VEHICLE':
                point_a = (p[0, 0] + s[0, 0] * np.cos(h[0]) / 2, p[0, 1] + s[0, 0] * np.sin(h[0]) / 2)
                point_b = (p[0, 0] + s[0, 0] * np.cos(h[0]) / 3 - s[0, 0] * np.sin(h[0]) / (6 * np.sqrt(3)),
                           p[0, 1] + s[0, 0] * np.sin(h[0]) / 3 + s[0, 0] * np.cos(h[0]) / (6 * np.sqrt(3)))
                point_c = (p[0, 0] + s[0, 0] * np.cos(h[0]) / 3 + s[0, 0] * np.sin(h[0]) / (6 * np.sqrt(3)),
                           p[0, 1] + s[0, 0] * np.sin(h[0]) / 3 - s[0, 0] * np.cos(h[0]) / (6 * np.sqrt(3)))
                ax.add_patch(Polygon(
                    xy=(point_a, point_b, point_c),
                    linewidth=0.5,
                    alpha=0.8,
                    facecolor='none',
                    edgecolor="black",
                    zorder=1
                ))

            # plot future state
            ax.plot(
                p[1:, 0], p[1:, 1],
                color=object_color if object_id != sdc_id else 'red',
                zorder=0
            )
            ax.scatter(
                p[1:, 0], p[1:, 1],
                s=4,
                color=object_color if object_id != sdc_id else 'red',
                zorder=0
            )

            if text:
                show_acceleration_labels = is_vehicle_agent(object_type)
                ax.text(
                    x=p[0, 0],
                    y=p[0, 1] + AGENT_LABEL_ID_OFFSET,
                    s=object_id,
                    fontsize=AGENT_LABEL_FONT_SIZE,
                    verticalalignment="center",
                    horizontalalignment="center",
                    color="w",
                    zorder=2
                )
                ax.text(
                    x=p[0, 0],
                    y=p[0, 1] + AGENT_LABEL_SPEED_OFFSET,
                    s=str(round(np.sqrt(v[0, 0] ** 2 + v[0, 1] ** 2), 1)),
                    fontsize=AGENT_LABEL_FONT_SIZE,
                    verticalalignment="center",
                    horizontalalignment="center",
                    color="w",
                    zorder=2
                )
                if show_acceleration_labels:
                    ax.text(
                        x=p[0, 0],
                        y=p[0, 1] + AGENT_LABEL_LON_ACC_OFFSET,
                        s=str(round(a[0, 0], 1)),
                        fontsize=AGENT_LABEL_FONT_SIZE,
                        verticalalignment="center",
                        horizontalalignment="center",
                        color="w",
                        zorder=2
                    )
                    ax.text(
                        x=p[0, 0],
                        y=p[0, 1] + AGENT_LABEL_LAT_ACC_OFFSET,
                        s=str(round(a[0, 1], 1)),
                        fontsize=AGENT_LABEL_FONT_SIZE,
                        verticalalignment="center",
                        horizontalalignment="center",
                        color="w",
                        zorder=2
                    )

    def plot_tracks(self, data, t, ax, history_scenario_t, plot=True):
        if not plot:
            print("No tracks")
            return
        scenario_t = []
        tracks = data['object_tracks']
        for object_id in tracks:
            track = tracks[object_id]
            object_dict = {}

            # timestamp = scenario_visualization.ts
            # frame = np.array(range(len(self.ts)))
            frame = np.array(range(len(track['state']['valid'])))
            global_center_x = track['state']['global_center'][:, 0]
            global_center_y = track['state']['global_center'][:, 1]
            local_velocity_x = track['state']['local_velocity'][:, 0]
            local_velocity_y = track['state']['local_velocity'][:, 1]
            local_acceleration_x = track['state']['local_acceleration'][:, 0]
            local_acceleration_y = track['state']['local_acceleration'][:, 1]
            heading = track['state']['heading']
            length = track['state']['size'][:, 0]
            width = track['state']['size'][:, 1]
            valid = track['state']['valid']
            action_label = action_label_at_frame(track['state'], t)

            speed = np.sqrt(local_velocity_x ** 2 + local_velocity_y ** 2)
            acceleration = np.sqrt(local_acceleration_x ** 2 + local_acceleration_y ** 2)

            '''
            # filling miss data
            valid_frame = frame[valid]
            valid_global_center_x = global_center_x[valid]
            valid_global_center_y = global_center_y[valid]
            valid_heading = heading[valid]

            start_index = valid_frame[0]
            end_index = valid_frame[-1]

            if np.sum(valid) >= 4:
                f1 = interp1d(valid_frame, valid_global_center_x, kind='cubic')
                f2 = interp1d(valid_frame, valid_global_center_y, kind='cubic')
                f3 = interp1d(valid_frame, valid_heading, kind='cubic')
            elif np.sum(valid) == 3:
                f1 = interp1d(valid_frame, valid_global_center_x, kind='quadratic')
                f2 = interp1d(valid_frame, valid_global_center_y, kind='quadratic')
                f3 = interp1d(valid_frame, valid_heading, kind='quadratic')
            elif np.sum(valid) == 2:
                f1 = interp1d(valid_frame, valid_global_center_x, kind='linear')
                f2 = interp1d(valid_frame, valid_global_center_y, kind='linear')
                f3 = interp1d(valid_frame, valid_heading, kind='linear')
            else:
                pass

            filled_global_center_x = np.copy(global_center_x)
            filled_global_center_y = np.copy(global_center_y)
            filled_heading = np.copy(heading)

            interpolation_range = (frame >= start_index) & (frame <= end_index)
            missing_frame = frame[(valid == 0) & interpolation_range]

            if missing_frame.size > 0:
                filled_global_center_x[missing_frame] = f1(missing_frame)
                filled_global_center_y[missing_frame] = f2(missing_frame)
                filled_heading[missing_frame] = f3(missing_frame)

            local_velocity_x = np.zeros_like(frame)
            local_velocity_y = np.zeros_like(frame)
            local_acceleration_x = np.zeros_like(frame)
            local_acceleration_y = np.zeros_like(frame)

            timestamp = self.ts
            if np.sum(valid) >= 100:
                valid_local_velocity_x, valid_local_velocity_y, valid_local_acceleration_x, valid_local_acceleration_y, _, _, _ = (
                    compute_vel_acc_jerk_ang_vel(filled_global_center_x[interpolation_range],
                                                 filled_global_center_y[interpolation_range],
                                                 timestamp[interpolation_range],
                                                 filled_heading[interpolation_range]))

                local_velocity_x[interpolation_range] = valid_local_velocity_x
                local_velocity_y[interpolation_range] = valid_local_velocity_y
                local_acceleration_x[interpolation_range] = valid_local_acceleration_x
                local_acceleration_y[interpolation_range] = valid_local_acceleration_y

                acceleration = np.sqrt(local_acceleration_x ** 2 + local_acceleration_y ** 2)
                speed = np.sqrt(local_velocity_x ** 2 + local_velocity_y ** 2)
            else:
                speed = np.array([0]*len(timestamp))
                acceleration = np.array([0]*len(timestamp))
            '''

            if object_id == self.sdc_id:
                valid = valid[t]
                if valid:
                    object_dict['x_t'] = global_center_x[t]
                    object_dict['y_t'] = global_center_y[t]
                    object_dict['vx_t'] = local_velocity_x[t]
                    object_dict['vy_t'] = local_velocity_y[t]
                    object_dict['v_t'] = speed[t]
                    object_dict['ax_t'] = local_acceleration_x[t]
                    object_dict['ay_t'] = local_acceleration_y[t]
                    object_dict['a_t'] = acceleration[t]
                    object_dict['l'] = length[t]
                    object_dict['w'] = width[t]
                    object_dict['heading'] = heading[t]
                    object_dict['object_type'] = track['type']
                    object_dict['action_label'] = action_label
                    if len(object_id) > 5:
                        object_dict['id'] = object_id[-5:]
                    else:
                        object_dict['id'] = object_id
                    object_dict['color'] = 'red'
                    scenario_t.append(object_dict)
            else:
                valid = track['state']['valid'][t]
                if valid:
                    object_dict['x_t'] = global_center_x[t]
                    object_dict['y_t'] = global_center_y[t]
                    object_dict['vx_t'] = local_velocity_x[t]
                    object_dict['vy_t'] = local_velocity_y[t]
                    object_dict['v_t'] = speed[t]
                    object_dict['ax_t'] = local_acceleration_x[t]
                    object_dict['ay_t'] = local_acceleration_y[t]
                    object_dict['a_t'] = acceleration[t]
                    object_dict['l'] = length[t]
                    object_dict['w'] = width[t]
                    object_dict['heading'] = heading[t]
                    object_dict['object_type'] = track['type']
                    object_dict['action_label'] = action_label
                    if len(object_id) > 5:
                        object_dict['id'] = object_id[-5:]
                    else:
                        object_dict['id'] = object_id
                    # object_dict['color'] = get_random_color(int(object_id))
                    object_dict['color'] = get_random_color(object_id)
                    scenario_t.append(object_dict)

        history_scenario_t.append(scenario_t)

        if len(history_scenario_t) >= 1:
            for t in range(max(0, len(history_scenario_t)-20), len(history_scenario_t)):
                for i in range(len(history_scenario_t[t])):
                    ax.add_patch(Rectangle(
                        xy=(history_scenario_t[t][i]["x_t"] - 0.5 * history_scenario_t[t][i]["l"],
                            history_scenario_t[t][i]["y_t"] - 0.5 * history_scenario_t[t][i]["w"]),
                        width=history_scenario_t[t][i]["l"],
                        height=history_scenario_t[t][i]["w"],
                        transform=mpl.transforms.Affine2D().rotate_deg_around(history_scenario_t[t][i]["x_t"],
                                                                              history_scenario_t[t][i]["y_t"],
                                                                              history_scenario_t[t][i]["heading"] * 180 / np.pi)
                                  + ax.transData,
                        alpha=0.2-(len(history_scenario_t)-t)*0.01,
                        linewidth=0,
                        facecolor=history_scenario_t[t][i]["color"],
                        edgecolor="black",
                        label=history_scenario_t[t][i]["object_type"],
                        zorder=3
                    ))

        for i in range(len(scenario_t)):
            if scenario_t[i]["l"] > 0 and scenario_t[i]["w"] > 0:
                ax.add_patch(Rectangle(
                    xy=(scenario_t[i]["x_t"] - 0.5 * scenario_t[i]["l"],
                        scenario_t[i]["y_t"] - 0.5 * scenario_t[i]["w"]),
                    width=scenario_t[i]["l"],
                    height=scenario_t[i]["w"],
                    transform=mpl.transforms.Affine2D().rotate_deg_around(scenario_t[i]["x_t"],
                                                                          scenario_t[i]["y_t"],
                                                                          scenario_t[i]["heading"] * 180 / np.pi)
                              + ax.transData,
                    alpha=0.8,
                    linewidth=0.5,
                    facecolor=scenario_t[i]["color"],
                    edgecolor="black",
                    label=scenario_t[i]["object_type"],
                    zorder=4
                ))

                if scenario_t[i]["object_type"] == 'VEHICLE':
                    point_a = (scenario_t[i]["x_t"] + scenario_t[i]["l"] * np.cos(scenario_t[i]["heading"]) / 2,
                               scenario_t[i]["y_t"] + scenario_t[i]["l"] * np.sin(scenario_t[i]["heading"]) / 2)
                    point_b = (scenario_t[i]["x_t"] + scenario_t[i]["l"] * np.cos(scenario_t[i]["heading"]) / 3 - scenario_t[i]["l"] * np.sin(scenario_t[i]["heading"]) / (6 * np.sqrt(3)),
                               scenario_t[i]["y_t"] + scenario_t[i]["l"] * np.sin(scenario_t[i]["heading"]) / 3 + scenario_t[i]["l"] * np.cos(scenario_t[i]["heading"]) / (6 * np.sqrt(3)))
                    point_c = (scenario_t[i]["x_t"] + scenario_t[i]["l"] * np.cos(scenario_t[i]["heading"]) / 3 + scenario_t[i]["l"] * np.sin(scenario_t[i]["heading"]) / (6 * np.sqrt(3)),
                               scenario_t[i]["y_t"] + scenario_t[i]["l"] * np.sin(scenario_t[i]["heading"]) / 3 - scenario_t[i]["l"] * np.cos(scenario_t[i]["heading"]) / (6 * np.sqrt(3)))
                    ax.add_patch(Polygon(
                        xy=(point_a, point_b, point_c),
                        linewidth=0.5,
                        alpha=0.8,
                        facecolor='none',
                        edgecolor="black",
                        zorder=5
                    ))
            else:
                plt.scatter(scenario_t[i]["x_t"], scenario_t[i]["y_t"], s=5, color=get_random_color(scenario_t[i]["id"]))

            x_obj = scenario_t[i]["x_t"]
            y_obj = scenario_t[i]["y_t"]

            if data['extra_information']['sdc_id']:
                sdc_id = data['extra_information']['sdc_id']
                sdc_center_x = data['object_tracks'][sdc_id]['state']['global_center'][t, 0]
                sdc_center_y = data['object_tracks'][sdc_id]['state']['global_center'][t, 1]
                sdc_heading = data['object_tracks'][sdc_id]['state']['heading'][t]
                # print(sdc_center_x, sdc_center_y, sdc_heading_angle)
            else:
                # sdc is a virtual vehicle in 'else'
                time = int(data['extra_information']['scene_length'] / 2)
                sdc_id = data['extra_information']['sdc_id']
                sdc_center_x = np.mean(np.array([data['object_tracks'][object_id]['state']['global_center'][time, 0]
                                                 for object_id in data['object_tracks']
                                                 if data['object_tracks'][object_id]['state']['valid'][time]]))
                sdc_center_y = np.mean(np.array([data['object_tracks'][object_id]['state']['global_center'][time, 1]
                                                 for object_id in data['object_tracks']
                                                 if data['object_tracks'][object_id]['state']['valid'][time]]))
                sdc_heading = 0

            x_obj_local, y_obj_local = global_to_local(x_obj, y_obj, sdc_center_x, sdc_center_y, sdc_heading)

            if (
                -AGENT_LABEL_VISIBLE_RANGE <= x_obj_local <= AGENT_LABEL_VISIBLE_RANGE
                and -AGENT_LABEL_VISIBLE_RANGE <= y_obj_local <= AGENT_LABEL_VISIBLE_RANGE
            ):
                plt.text(x=scenario_t[i]["x_t"], y=scenario_t[i]["y_t"] + AGENT_LABEL_ID_OFFSET, s=scenario_t[i]["id"],
                         fontsize=AGENT_LABEL_FONT_SIZE, verticalalignment="center", horizontalalignment="center", color="w", zorder=5, clip_on=True)

                plt.text(x=scenario_t[i]["x_t"], y=scenario_t[i]["y_t"] + AGENT_LABEL_SPEED_OFFSET, s=str(round(scenario_t[i]["v_t"], 1)),
                         fontsize=AGENT_LABEL_FONT_SIZE, verticalalignment="center", horizontalalignment="center", color="w", zorder=5, clip_on=True)

                if is_vehicle_agent(scenario_t[i]["object_type"]):
                    plt.text(x=scenario_t[i]["x_t"], y=scenario_t[i]["y_t"] + AGENT_LABEL_LON_ACC_OFFSET, s=str(round(scenario_t[i]["ax_t"], 1)),
                             fontsize=AGENT_LABEL_FONT_SIZE, verticalalignment="center", horizontalalignment="center", color="w", zorder=5, clip_on=True)

                    plt.text(x=scenario_t[i]["x_t"], y=scenario_t[i]["y_t"] + AGENT_LABEL_LAT_ACC_OFFSET, s=str(round(scenario_t[i]["ay_t"], 1)),
                             fontsize=AGENT_LABEL_FONT_SIZE, verticalalignment="center", horizontalalignment="center", color="w", zorder=5, clip_on=True)
                    action_offset = AGENT_LABEL_ACTION_OFFSET
                else:
                    action_offset = AGENT_LABEL_LON_ACC_OFFSET

                plt.text(
                    x=scenario_t[i]["x_t"],
                    y=scenario_t[i]["y_t"] + action_offset,
                    s=f"act:{scenario_t[i]['action_label']}",
                    fontsize=AGENT_LABEL_FONT_SIZE,
                    verticalalignment="center",
                    horizontalalignment="center",
                    color="w",
                    zorder=5,
                    clip_on=True,
                )
                # plt.text(x=scenario_t[i]["x_t"], y=scenario_t[i]["y_t"] + 1.2, s=scenario_t[i]["id"],
                #          fontsize=4, verticalalignment="center", horizontalalignment="center", color="w", zorder=5, clip_on=True)
                #
                # plt.text(x=scenario_t[i]["x_t"], y=scenario_t[i]["y_t"], s=str(round(scenario_t[i]["v_t"], 1)),
                #          fontsize=4, verticalalignment="center", horizontalalignment="center", color="w", zorder=5, clip_on=True)
                #
                # plt.text(x=scenario_t[i]["x_t"], y=scenario_t[i]["y_t"] - 1.2, s=str(round(scenario_t[i]["a_t"], 1)),
                #          fontsize=4, verticalalignment="center", horizontalalignment="center", color="w", zorder=5, clip_on=True)
            else:
                plt.text(x=scenario_t[i]["x_t"], y=scenario_t[i]["y_t"], s=scenario_t[i]["id"],
                         fontsize=AGENT_LABEL_FONT_SIZE, verticalalignment="center", horizontalalignment="center", color="w", zorder=5, clip_on=True)

        return sdc_center_x, sdc_center_y

    def plot_dynamic_map_states(self, ax, data, t, plot=True):
        if not plot:
            print("No dynamic map states")
            return

        map_features = data['map_features']
        dynamic_map_states = data['dynamic_map_states']
        for dynamic_map_id in dynamic_map_states:
            dynamic_map_state = dynamic_map_states[dynamic_map_id]
            # if dynamic_map_state['state']['color'][t] == 'GREEN':
            #     x, y = dynamic_map_state['stop_point'][0], dynamic_map_state['stop_point'][1]
            #     ax.scatter(x, y, s=10, edgecolor="black", linewidth=0.5, c="green", zorder=3)
            # elif dynamic_map_state['state']['color'][t] == 'RED':
            #     x, y = dynamic_map_state['stop_point'][0], dynamic_map_state['stop_point'][1]
            #     ax.scatter(x, y, s=10, edgecolor="black", linewidth=0.5, c="red", zorder=3)
            # elif dynamic_map_state['state']['color'][t] == 'YELLOW':
            #     x, y = dynamic_map_state['stop_point'][0], dynamic_map_state['stop_point'][1]
            #     ax.scatter(x, y, s=10, edgecolor="black", linewidth=0.5, c="yellow", zorder=3)
            # elif dynamic_map_state['state']['color'][t] == 'FLASHING':
            #     x, y = dynamic_map_state['stop_point'][0], dynamic_map_state['stop_point'][1]
            #     ax.scatter(x, y, s=10, edgecolor="black", linewidth=0.5, c="blue", zorder=3)
            # elif dynamic_map_state['state']['color'][t] == 'UNKNOWN':
            #     x, y = dynamic_map_state['stop_point'][0], dynamic_map_state['stop_point'][1]
            #     ax.scatter(x, y, s=10, edgecolor="black", linewidth=0.5, c="black", zorder=3)
            # else:
            #     pass
            if dynamic_map_state['state']['color'][t] == 'GREEN':
                x, y = [row[0] for row in map_features[dynamic_map_id]['polyline']], [row[1] for row in map_features[dynamic_map_id]['polyline']]
                ax.plot(x, y, c="green", linewidth=1, zorder=1)
            elif dynamic_map_state['state']['color'][t] == 'RED':
                x, y = [row[0] for row in map_features[dynamic_map_id]['polyline']], [row[1] for row in map_features[dynamic_map_id]['polyline']]
                ax.plot(x, y, c="red", linewidth=1, zorder=1)
            elif dynamic_map_state['state']['color'][t] == 'YELLOW':
                x, y = [row[0] for row in map_features[dynamic_map_id]['polyline']], [row[1] for row in map_features[dynamic_map_id]['polyline']]
                ax.plot(x, y, c="yellow", linewidth=1, zorder=1)
            elif dynamic_map_state['state']['color'][t] == 'FLASHING':
                x, y = [row[0] for row in map_features[dynamic_map_id]['polyline']], [row[1] for row in map_features[dynamic_map_id]['polyline']]
                ax.plot(x, y, c="blue", linewidth=1, zorder=1)
            elif dynamic_map_state['state']['color'][t] == 'UNKNOWN':
                x, y = [row[0] for row in map_features[dynamic_map_id]['polyline']], [row[1] for row in map_features[dynamic_map_id]['polyline']]
                ax.plot(x, y, c="black", linewidth=1, zorder=1)
            else:
                pass

    def plot_map_features(self, data, fig, ax, result_path, plot=True, extra_plot=False):
        if not plot:
            return

        map_features = data['map_features']
        for map_feature_id in map_features:
            map_feature = map_features[map_feature_id]
            if map_feature['subtype'] == 'DASHED_WHITE':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                plt.plot(x, y, c='white', linestyle=(0, (4, 6)), linewidth=1)
            elif map_feature['subtype'] == 'SOLID_WHITE':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                plt.plot(x, y, c='white', linewidth=1)
            elif map_feature['subtype'] == 'DOUBLE_SOLID_WHITE':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                x1, y1, x2, y2 = calculate_parallel_curves(x, y)
                plt.plot(x1, y1, c='white', linestyle=(0, (4, 6)), linewidth=1)
                plt.plot(x1, y1, c='white', linestyle=(0, (4, 6)), linewidth=1)
            elif map_feature['subtype'] == 'DASHED_YELLOW':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                plt.plot(x, y, c='yellow', linestyle=(0, (4, 6)), linewidth=1)
            elif map_feature['subtype'] == 'DOUBLE_DASH_WHITE':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                x1, y1, x2, y2 = calculate_parallel_curves(x, y)
                plt.plot(x1, y1, c='white', linestyle=(0, (4, 6)), linewidth=1)
                plt.plot(x2, y2, c='white', linestyle=(0, (4, 6)), linewidth=1)
            elif map_feature['subtype'] == 'DOUBLE_DASH_YELLOW':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                x1, y1, x2, y2 = calculate_parallel_curves(x, y)
                plt.plot(x1, y1, c='yellow', linestyle=(0, (4, 6)), linewidth=1)
                plt.plot(x2, y2, c='yellow', linestyle=(0, (4, 6)), linewidth=1)
            elif map_feature['subtype'] == 'SOLID_YELLOW':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                plt.plot(x, y, c='yellow', linewidth=1)
            elif map_feature['subtype'] == 'SOLID_BLUE':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                plt.plot(x, y, c='blue', linewidth=1)
            elif map_feature['subtype'] == 'DOUBLE_SOLID_YELLOW':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                x1, y1, x2, y2 = calculate_parallel_curves(x, y)
                plt.plot(x1, y1, c='yellow', linewidth=1)
                plt.plot(x2, y2, c='yellow', linewidth=1)
            elif map_feature['subtype'] == 'DASH_SOLID_YELLOW':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                x1, y1, x2, y2 = calculate_parallel_curves(x, y)
                plt.plot(x1, y1, c='yellow', linestyle=(0, (4, 6)), linewidth=1)
                plt.plot(x2, y2, c='yellow', linewidth=1)
            elif map_feature['subtype'] == 'SOLID_DASH_YELLOW':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                x1, y1, x2, y2 = calculate_parallel_curves(x, y)
                plt.plot(x1, y1, c='yellow', linewidth=1)
                plt.plot(x2, y2, c='yellow', linestyle=(0, (4, 6)), linewidth=1)
            elif map_feature['subtype'] == 'DASH_SOLID_WHITE':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                x1, y1, x2, y2 = calculate_parallel_curves(x, y)
                plt.plot(x1, y1, c='white', linestyle=(0, (4, 6)), linewidth=1)
                plt.plot(x2, y2, c='white', linewidth=1)
            elif map_feature['subtype'] == 'SOLID_DASH_WHITE':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                x1, y1, x2, y2 = calculate_parallel_curves(x, y)
                plt.plot(x1, y1, c='white', linewidth=1)
                plt.plot(x2, y2, c='white', linestyle=(0, (4, 6)), linewidth=1)
            elif map_feature['subtype'] == 'VIRTUAL':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                plt.plot(x, y, c="black", linewidth=1, alpha=0.2)
                pass
            elif map_feature['subtype'] == 'BOUNDARY':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                plt.plot(x, y, c="black", linewidth=1)
            elif map_feature['subtype'] == 'MEDIAN':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                plt.plot(x, y, c="green", linewidth=1)
            elif map_feature['type'] == 'ROAD_LINE' and map_feature['subtype'] == 'UNKNOWN':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                plt.plot(x, y, c="black", linewidth=1, alpha=0.5)
            elif map_feature['type'] == 'ROAD_EDGE' and map_feature['subtype'] == 'UNKNOWN':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                plt.plot(x, y, c="black", linewidth=1)
            else:
                pass

            if map_feature['subtype'] == 'STOP_LINE':
                # x, y = [row[0] for row in map_feature['polygon']], [row[1] for row in map_feature['polygon']]
                # x.append(map_feature['polygon'][0][0])
                # y.append(map_feature['polygon'][0][1])
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                plt.plot(x, y, c="black", linewidth=1, zorder=8)

            if map_feature['subtype'] == 'CROSSWALK':
                x, y = [row[0] for row in map_feature['polygon']], [row[1] for row in map_feature['polygon']]
                x.append(map_feature['polygon'][0][0])
                y.append(map_feature['polygon'][0][1])
                plt.plot(x, y, c="white", linewidth=1, zorder=6)
            elif map_feature['subtype'] == 'SPEED_BUMP':
                x, y = [row[0] for row in map_feature['polygon']], [row[1] for row in map_feature['polygon']]
                x.append(map_feature['polygon'][0][0])
                y.append(map_feature['polygon'][0][1])
                plt.plot(x, y, c="yellow", linewidth=1)
            elif map_feature['subtype'] == 'DRIVEWAY':
                x, y = [row[0] for row in map_feature['polygon']], [row[1] for row in map_feature['polygon']]
                x.append(map_feature['polygon'][0][0])
                y.append(map_feature['polygon'][0][1])
                plt.plot(x, y, c="blue", linewidth=1)   # TODO driveway's color is blue ?
            elif map_feature['subtype'] == 'SIDEWALK':
                x, y = [row[0] for row in map_feature['polygon']], [row[1] for row in map_feature['polygon']]
                x.append(map_feature['polygon'][0][0])
                y.append(map_feature['polygon'][0][1])
                plt.plot(x, y, c="cyan", linewidth=1)
            elif map_feature['subtype'] == 'KEEPOUT':
                x, y = [row[0] for row in map_feature['polygon']], [row[1] for row in map_feature['polygon']]
                x.append(map_feature['polygon'][0][0])
                y.append(map_feature['polygon'][0][1])
                plt.plot(x, y, c="red", linewidth=1)
            # elif map_feature['type'] == 'JUNCTION':
            #     x, y = [row[0] for row in map_feature['polygon']], [row[1] for row in map_feature['polygon']]
            #     x.append(map_feature['polygon'][0][0])
            #     y.append(map_feature['polygon'][0][1])
            #     plt.plot(x, y, c="white", linewidth=1)
            else:
                pass

            # if map_feature['type'] == 'STOP_SIGN' or map_feature['type'] == 'YIELD_SIGN':       # TODO YIELD_SIGN
            #     x, y = map_feature['position'][0], map_feature['position'][1]
            #     plt.scatter(x, y, s=10, c="black", edgecolor="black", linewidth=0.5)

            if extra_plot:
                if map_feature['subtype'] == 'URBAN_VEHICLE':
                    x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                    plt.plot(x, y, c="purple", linewidth=1, zorder=1, alpha=0.1)
                elif map_feature['subtype'] == 'FREEWAY' or map_feature['type'] == 'NO_RESTRICTION':    # TODO NO_RESTRICTION ?
                    x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                    plt.plot(x, y, c="blue", linewidth=1, zorder=1, alpha=0.1)
                elif map_feature['subtype'] == 'HIGHWAY':
                    x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                    plt.plot(x, y, c="green", linewidth=1, zorder=1, alpha=0.1)
                elif map_feature['subtype'] == 'URBAN_BIKE':
                    x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                    plt.plot(x, y, c="orange", linewidth=1, zorder=1, alpha=0.1)
                elif map_feature['subtype'] == 'URBAN_UNSTRUCTURE':        # TODO URBAN_UNSTRUCTURE ?
                    x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                    plt.plot(x, y, c="black", linewidth=1, zorder=1, alpha=0.1)
                elif map_feature['subtype'] == 'URBAN_UNKNOWN':
                    x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                    plt.plot(x, y, c="black", linewidth=1, zorder=1, alpha=0.1)
                else:
                    pass

        os.makedirs(f"{result_path}/temp", exist_ok=True)

        return pickle.dump((fig, ax), open(f"{result_path}/temp/map_features.pkl", 'wb'))


    def plot_map_features_without_save(self, data, ax, plot=True, extra_plot=True):
        if not plot:
            print("No map features")
            return

        map_features = data['map_features']
        for map_feature_id in map_features:
            map_feature = map_features[map_feature_id]
            if map_feature['subtype'] == 'DASHED_WHITE':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                ax.plot(x, y, c='white', linestyle=(0, (4, 6)), linewidth=1)
            elif map_feature['subtype'] == 'SOLID_WHITE':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                ax.plot(x, y, c='white', linewidth=1)
            elif map_feature['subtype'] == 'DOUBLE_SOLID_WHITE':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                x1, y1, x2, y2 = calculate_parallel_curves(x, y)
                ax.plot(x1, y1, c='white', linestyle=(0, (4, 6)), linewidth=1)
                ax.plot(x1, y1, c='white', linestyle=(0, (4, 6)), linewidth=1)
            elif map_feature['subtype'] == 'DASHED_YELLOW':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                ax.plot(x, y, c='yellow', linestyle=(0, (4, 6)), linewidth=1)
            elif map_feature['subtype'] == 'DOUBLE_DASH_WHITE':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                x1, y1, x2, y2 = calculate_parallel_curves(x, y)
                ax.plot(x1, y1, c='white', linestyle=(0, (4, 6)), linewidth=1)
                ax.plot(x2, y2, c='white', linestyle=(0, (4, 6)), linewidth=1)
            elif map_feature['subtype'] == 'DOUBLE_DASH_YELLOW':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                x1, y1, x2, y2 = calculate_parallel_curves(x, y)
                ax.plot(x1, y1, c='yellow', linestyle=(0, (4, 6)), linewidth=1)
                ax.plot(x2, y2, c='yellow', linestyle=(0, (4, 6)), linewidth=1)
            elif map_feature['subtype'] == 'SOLID_YELLOW':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                ax.plot(x, y, c='yellow', linewidth=1)
            elif map_feature['subtype'] == 'SOLID_BLUE':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                ax.plot(x, y, c='blue', linewidth=1)
            elif map_feature['subtype'] == 'DOUBLE_SOLID_YELLOW':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                x1, y1, x2, y2 = calculate_parallel_curves(x, y)
                ax.plot(x1, y1, c='yellow', linewidth=1)
                ax.plot(x2, y2, c='yellow', linewidth=1)
            elif map_feature['subtype'] == 'DASH_SOLID_YELLOW':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                x1, y1, x2, y2 = calculate_parallel_curves(x, y)
                ax.plot(x1, y1, c='yellow', linestyle=(0, (4, 6)), linewidth=1)
                ax.plot(x2, y2, c='yellow', linewidth=1)
            elif map_feature['subtype'] == 'SOLID_DASH_YELLOW':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                x1, y1, x2, y2 = calculate_parallel_curves(x, y)
                ax.plot(x1, y1, c='yellow', linewidth=1)
                ax.plot(x2, y2, c='yellow', linestyle=(0, (4, 6)), linewidth=1)
            elif map_feature['subtype'] == 'DASH_SOLID_WHITE':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                x1, y1, x2, y2 = calculate_parallel_curves(x, y)
                ax.plot(x1, y1, c='white', linestyle=(0, (4, 6)), linewidth=1)
                ax.plot(x2, y2, c='white', linewidth=1)
            elif map_feature['subtype'] == 'SOLID_DASH_WHITE':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                x1, y1, x2, y2 = calculate_parallel_curves(x, y)
                ax.plot(x1, y1, c='white', linewidth=1)
                ax.plot(x2, y2, c='white', linestyle=(0, (4, 6)), linewidth=1)
            elif map_feature['subtype'] == 'BOUNDARY':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                ax.plot(x, y, c="black", linewidth=1)
            elif map_feature['subtype'] == 'MEDIAN':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                ax.plot(x, y, c="green", linewidth=1)
            elif map_feature['type'] == 'ROAD_LINE' and map_feature['subtype'] == 'UNKNOWN':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                ax.plot(x, y, c="black", linewidth=1, alpha=0.5)
            # elif map_feature['type'] == 'ROAD_LINE_VIRTUAL':
            #     x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
            #     ax.plot(x, y, c="lightgray", linewidth=1, alpha=0.5)
            elif map_feature['type'] == 'ROAD_EDGE' and map_feature['subtype'] == 'UNKNOWN':
                x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                ax.plot(x, y, c="black", linewidth=1)
            # elif map_feature['subtype'] == 'VIRTUAL': # ROAD_LINE  VIRTUAL
            #     x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
            #     ax.plot(x, y, c="red", linewidth=1)
            else:
                pass

            if map_feature['type'] == 'STOP_LINE':
                x, y = [row[0] for row in map_feature['polygon']], [row[1] for row in map_feature['polygon']]
                x.append(map_feature['polygon'][0][0])
                y.append(map_feature['polygon'][0][1])
                ax.plot(x, y, c="black", linewidth=1, zorder=2)

            if map_feature['subtype'] == 'CROSSWALK':
                x, y = [row[0] for row in map_feature['polygon']], [row[1] for row in map_feature['polygon']]
                x.append(map_feature['polygon'][0][0])
                y.append(map_feature['polygon'][0][1])
                ax.plot(x, y, c="white", linewidth=1, zorder=6)
            elif map_feature['subtype'] == 'SPEED_BUMP':
                x, y = [row[0] for row in map_feature['polygon']], [row[1] for row in map_feature['polygon']]
                x.append(map_feature['polygon'][0][0])
                y.append(map_feature['polygon'][0][1])
                ax.plot(x, y, c="yellow", linewidth=1)
            elif map_feature['subtype'] == 'DRIVEWAY':
                x, y = [row[0] for row in map_feature['polygon']], [row[1] for row in map_feature['polygon']]
                x.append(map_feature['polygon'][0][0])
                y.append(map_feature['polygon'][0][1])
                ax.plot(x, y, c="blue", linewidth=1)   # TODO driveway's color is blue ?
            elif map_feature['subtype'] == 'SIDEWALK':
                x, y = [row[0] for row in map_feature['polygon']], [row[1] for row in map_feature['polygon']]
                x.append(map_feature['polygon'][0][0])
                y.append(map_feature['polygon'][0][1])
                ax.plot(x, y, c="cyan", linewidth=1)
            elif map_feature['subtype'] == 'KEEPOUT':
                x, y = [row[0] for row in map_feature['polygon']], [row[1] for row in map_feature['polygon']]
                x.append(map_feature['polygon'][0][0])
                y.append(map_feature['polygon'][0][1])
                ax.plot(x, y, c="red", linewidth=1)
            # elif map_feature['type'] == 'JUNCTION':
            #     x, y = [row[0] for row in map_feature['polygon']], [row[1] for row in map_feature['polygon']]
            #     x.append(map_feature['polygon'][0][0])
            #     y.append(map_feature['polygon'][0][1])
            #     ax.plot(x, y, c="white", linewidth=1)
            else:
                pass

            # if map_feature['type'] == 'STOP_SIGN' or map_feature['type'] == 'YIELD_SIGN':       # TODO YIELD_SIGN
            #     x, y = map_feature['position'][0], map_feature['position'][1]
            #     ax.scatter(x, y, s=10, c="black", edgecolor="black", linewidth=0.5)

            if extra_plot:
                if map_feature['subtype'] == 'URBAN_VEHICLE':
                    x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                    ax.plot(x, y, c="purple", linewidth=1, zorder=1, alpha=0.1)
                elif map_feature['subtype'] == 'FREEWAY' or map_feature['type'] == 'NO_RESTRICTION':    # TODO NO_RESTRICTION ?
                    x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                    ax.plot(x, y, c="blue", linewidth=1, zorder=1, alpha=0.1)
                elif map_feature['subtype'] == 'HIGHWAY':
                    x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                    ax.plot(x, y, c="green", linewidth=1, zorder=1, alpha=0.1)
                elif map_feature['subtype'] == 'URBAN_BIKE':
                    x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                    ax.plot(x, y, c="orange", linewidth=1, zorder=1, alpha=0.1)
                elif map_feature['subtype'] == 'URBAN_UNSTRUCTURE':        # TODO URBAN_UNSTRUCTURE ?
                    x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                    ax.plot(x, y, c="black", linewidth=1, zorder=1, alpha=0.1)
                elif map_feature['subtype'] == 'UNKNOWN':
                    x, y = [row[0] for row in map_feature['polyline']], [row[1] for row in map_feature['polyline']]
                    ax.plot(x, y, c="black", linewidth=1, zorder=1, alpha=0.1)
                else:
                    pass

    def plot_traffic_event_pair(self, data, traffic_events, t, plot=False):
        if plot:
            for idx, traffic_event in traffic_events.iterrows():
                object_tracks = data['object_tracks']

                leader_id = str(int(traffic_event['leader id']))
                follower_id = str(int(traffic_event['follower id']))
                first_frame = traffic_event['first frame']
                last_frame = traffic_event['last frame']

                if first_frame <= t <= last_frame:
                    x = (object_tracks[leader_id]['state']['center_x'][t], object_tracks[follower_id]['state']['center_x'][t])
                    y = (object_tracks[leader_id]['state']['center_y'][t], object_tracks[follower_id]['state']['center_y'][t])
                    plt.plot(x, y, c="black", linewidth=1, zorder=3)
                else:
                    pass

    def plot_agent_map_interaction(self, ax, data, t, labeling, plot=False):
        if plot:
            valid_for_road_segment = [True if i == 'road segment' else False for i in labeling['Region']]
            valid_for_other = [True if i == 'other' else False for i in labeling['Region']]

            object_tracks = data['object_tracks']
            map_features = data['map_features']
            extra_information = data['extra_information']

            sdc_id = extra_information['sdc_id']

            ax.scatter(object_tracks[sdc_id]['state']['center_x'][valid_for_road_segment],
                        object_tracks[sdc_id]['state']['center_y'][valid_for_road_segment],
                        color='blue', zorder=3, s=1)
            ax.scatter(object_tracks[sdc_id]['state']['center_x'][valid_for_other],
                        object_tracks[sdc_id]['state']['center_y'][valid_for_other],
                        color='green', zorder=3, s=1)

            for map_feature_id in map_features:
                map_feature = map_features[map_feature_id]
                map_feature_type = map_feature['type']

                polyline_flag = True if 'polyline' in list(map_feature.keys()) else False
                if polyline_flag:
                    map_feature_polyline = map_feature['polyline']

                    polyline_center = map_feature_polyline[len(map_feature_polyline) // 2, :]
                    if map_feature_type == 'ROAD_LANE':
                        # ax.text(polyline_center[0], polyline_center[1], map_feature_id, clip_on=True)
                        pass
                    else:
                        pass

            sdc_center_x = data['object_tracks'][data['extra_information']['sdc_id']]['state']['center_x'][t]
            sdc_center_y = data['object_tracks'][data['extra_information']['sdc_id']]['state']['center_y'][t]

            ax.text(sdc_center_x-40, sdc_center_y+12,
                    " Type of region: [1, 0, 1] \n Number of vehicle lanes: [3, 0, 2] \n "
                    "Number of left vehicle lanes: [2, 0, 1] \n Number of right vehicle lanes: [0, 0, 0] \n "
                    "Current lane id: '144' \n Lane ids: [['183', '188', '187'], [], ['267', '268']]",
                    color='r', clip_on=True, zorder=7,
                    bbox=dict(
                        boxstyle="round",  # rounded box
                        facecolor="white",  # background color
                        edgecolor="gray",  # border color
                        alpha=0.5,  # opacity (0-1)
                        linewidth=1  # border width
                    ))

    def plot_map_interaction(self, ax, data, frame, plot=False):    # TODO add labeling dataset information
        if plot:
            sdc_center_x = data['object_tracks'][data['extra_information']['sdc_id']]['state']['center_x'][frame]
            sdc_center_y = data['object_tracks'][data['extra_information']['sdc_id']]['state']['center_y'][frame]

            ax.text(sdc_center_x-40, sdc_center_y+24,
                    " Lane ids: [['183', '188', '187'], [], ['267', '268']] \n "
                    "Lane width: [[3.50, 3.50, 3.50], [], [3.75, 3.75]]",
                    color='r', clip_on=True, zorder=7,
                    bbox=dict(
                        boxstyle="round",  # rounded box
                        facecolor="white",  # background color
                        edgecolor="gray",  # border color
                        alpha=0.5,  # opacity (0-1)
                        linewidth=1  # border width
                    ))
            pass


def calculate_parallel_curves(x, y):
    # actual curve offset distance (0.3 m)
    offset = 0.3

    # compute curve gradient (dy/dx)
    dy_dx = np.gradient(y, x)

    # compute normal vector
    normal_x = -dy_dx
    normal_y = np.ones_like(dy_dx)

    # normalize normal vector
    length = np.sqrt(normal_x ** 2 + normal_y ** 2)
    normal_x /= length
    normal_y /= length

    # compute per-point offset (0.3 m spacing)
    dx = offset * normal_x
    dy = offset * normal_y

    # generate two parallel curves
    x1 = x + dx
    y1 = y + dy
    x2 = x - dx
    y2 = y - dy
    return x1, y1, x2, y2


# Generate random colors using the given seed
def get_random_color(seed):
    random.seed(seed)
    random_color1 = random.random()
    random_color2 = random.random()
    random_color3 = random.random()

    random_color1 = 0.3 + 0.4 * random_color1
    random_color2 = 0.3 + 0.4 * random_color2
    random_color3 = 0.3 + 0.4 * random_color3

    return (random_color1, random_color2, random_color3)


def compute_vel_acc_jerk_ang_vel(center_x, center_y, timestamp, heading):
    # Ensure input arrays are numpy arrays
    center_x = np.array(center_x)
    center_y = np.array(center_y)
    heading = np.unwrap(np.array(heading))  # Unwrap the heading to handle discontinuities
    timestamp = np.array(timestamp)

    N = len(center_x)
    velocity_x = np.zeros(N)
    velocity_y = np.zeros(N)
    acceleration_x = np.zeros(N)
    acceleration_y = np.zeros(N)
    jerk_x = np.zeros(N)
    jerk_y = np.zeros(N)
    angular_velocity = np.zeros(N)

    # Compute time steps (assuming variable time steps)
    h = np.diff(timestamp)

    # First derivative (velocity and angular velocity)
    for i in range(N):
        if i == 0:
            # Forward difference (first order)
            h_i = timestamp[i + 1] - timestamp[i]
            velocity_x[i] = (center_x[i + 1] - center_x[i]) / h_i
            velocity_y[i] = (center_y[i + 1] - center_y[i]) / h_i
            angular_velocity[i] = (heading[i + 1] - heading[i]) / h_i
        elif i == 1:
            # Forward difference (second order)
            h_i = timestamp[i + 1] - timestamp[i]
            h_im1 = timestamp[i] - timestamp[i - 1]
            h_total = h_i + h_im1
            velocity_x[i] = (-center_x[i + 2] + 4 * center_x[i + 1] - 3 * center_x[i]) / (2 * h_total)
            velocity_y[i] = (-center_y[i + 2] + 4 * center_y[i + 1] - 3 * center_y[i]) / (2 * h_total)
            angular_velocity[i] = (-heading[i + 2] + 4 * heading[i + 1] - 3 * heading[i]) / (2 * h_total)
        elif i == N - 2:
            # Backward difference (second order)
            h_i = timestamp[i] - timestamp[i - 1]
            h_im1 = timestamp[i - 1] - timestamp[i - 2]
            h_total = h_i + h_im1
            velocity_x[i] = (3 * center_x[i] - 4 * center_x[i - 1] + center_x[i - 2]) / (2 * h_total)
            velocity_y[i] = (3 * center_y[i] - 4 * center_y[i - 1] + center_y[i - 2]) / (2 * h_total)
            angular_velocity[i] = (3 * heading[i] - 4 * heading[i - 1] + heading[i - 2]) / (2 * h_total)
        elif i == N - 1:
            # Backward difference (first order)
            h_im1 = timestamp[i] - timestamp[i - 1]
            velocity_x[i] = (center_x[i] - center_x[i - 1]) / h_im1
            velocity_y[i] = (center_y[i] - center_y[i - 1]) / h_im1
            angular_velocity[i] = (heading[i] - heading[i - 1]) / h_im1
        else:
            # Central difference (fourth-order)
            h_total = timestamp[i + 2] - timestamp[i - 2]
            h_avg = h_total / 4  # Average h
            velocity_x[i] = (-center_x[i + 2] + 8 * center_x[i + 1] - 8 * center_x[i - 1] + center_x[i - 2]) / (
                        12 * h_avg)
            velocity_y[i] = (-center_y[i + 2] + 8 * center_y[i + 1] - 8 * center_y[i - 1] + center_y[i - 2]) / (
                        12 * h_avg)
            angular_velocity[i] = (-heading[i + 2] + 8 * heading[i + 1] - 8 * heading[i - 1] + heading[i - 2]) / (
                        12 * h_avg)

    # Second derivative (acceleration)
    for i in range(N):
        if i == 0 or i == 1:
            # Forward difference (second-order)
            h_i = timestamp[i + 1] - timestamp[i]
            h_ip1 = timestamp[i + 2] - timestamp[i + 1]
            acceleration_x[i] = (center_x[i + 2] - 2 * center_x[i + 1] + center_x[i]) / (h_i * h_ip1)
            acceleration_y[i] = (center_y[i + 2] - 2 * center_y[i + 1] + center_y[i]) / (h_i * h_ip1)
        elif i == N - 2 or i == N - 1:
            # Backward difference (second-order)
            h_im1 = timestamp[i] - timestamp[i - 1]
            h_im2 = timestamp[i - 1] - timestamp[i - 2]
            acceleration_x[i] = (center_x[i] - 2 * center_x[i - 1] + center_x[i - 2]) / (h_im1 * h_im2)
            acceleration_y[i] = (center_y[i] - 2 * center_y[i - 1] + center_y[i - 2]) / (h_im1 * h_im2)
        else:
            # Central difference (fourth-order)
            h_total_sq = ((timestamp[i + 2] - timestamp[i - 2]) / 4) ** 2
            acceleration_x[i] = (-center_x[i + 2] + 16 * center_x[i + 1] - 30 * center_x[i] + 16 * center_x[i - 1] -
                                 center_x[i - 2]) / (12 * h_total_sq)
            acceleration_y[i] = (-center_y[i + 2] + 16 * center_y[i + 1] - 30 * center_y[i] + 16 * center_y[i - 1] -
                                 center_y[i - 2]) / (12 * h_total_sq)

    # Third derivative (jerk)
    for i in range(N):
        if i <= 2:
            # Forward difference (third-order)
            h_i = timestamp[i + 1] - timestamp[i]
            h_ip1 = timestamp[i + 2] - timestamp[i + 1]
            h_ip2 = timestamp[i + 3] - timestamp[i + 2]
            jerk_x[i] = (-center_x[i + 3] + 3 * center_x[i + 2] - 3 * center_x[i + 1] + center_x[i]) / (
                        h_i * h_ip1 * h_ip2)
            jerk_y[i] = (-center_y[i + 3] + 3 * center_y[i + 2] - 3 * center_y[i + 1] + center_y[i]) / (
                        h_i * h_ip1 * h_ip2)
        elif i >= N - 3:
            # Backward difference (third-order)
            h_im1 = timestamp[i] - timestamp[i - 1]
            h_im2 = timestamp[i - 1] - timestamp[i - 2]
            h_im3 = timestamp[i - 2] - timestamp[i - 3]
            jerk_x[i] = (center_x[i] - 3 * center_x[i - 1] + 3 * center_x[i - 2] - center_x[i - 3]) / (
                        h_im1 * h_im2 * h_im3)
            jerk_y[i] = (center_y[i] - 3 * center_y[i - 1] + 3 * center_y[i - 2] - center_y[i - 3]) / (
                        h_im1 * h_im2 * h_im3)
        else:
            # Central difference (fourth-order)
            h_total_cubed = ((timestamp[i + 2] - timestamp[i - 2]) / 4) ** 3
            jerk_x[i] = (-center_x[i + 3] + 8 * center_x[i + 2] - 13 * center_x[i + 1] + 13 * center_x[i - 1] - 8 *
                         center_x[i - 2] + center_x[i - 3]) / (8 * h_total_cubed)
            jerk_y[i] = (-center_y[i + 3] + 8 * center_y[i + 2] - 13 * center_y[i + 1] + 13 * center_y[i - 1] - 8 *
                         center_y[i - 2] + center_y[i - 3]) / (8 * h_total_cubed)

    # Now convert the velocity, acceleration, and jerk to the vehicle coordinate system
    velocity_local_x = np.zeros(N)
    velocity_local_y = np.zeros(N)
    acceleration_local_x = np.zeros(N)
    acceleration_local_y = np.zeros(N)
    jerk_local_x = np.zeros(N)
    jerk_local_y = np.zeros(N)

    for i in range(N):
        # Rotation matrix for transforming global to local coordinates
        cos_theta = np.cos(heading[i])
        sin_theta = np.sin(heading[i])
        rotation_matrix = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])

        # Convert velocity
        velocity_global = np.array([velocity_x[i], velocity_y[i]])
        velocity_local = np.dot(rotation_matrix, velocity_global)
        velocity_local_x[i], velocity_local_y[i] = velocity_local

        # Convert acceleration
        acceleration_global = np.array([acceleration_x[i], acceleration_y[i]])
        acceleration_local = np.dot(rotation_matrix, acceleration_global)
        acceleration_local_x[i], acceleration_local_y[i] = acceleration_local

        # Convert jerk
        jerk_global = np.array([jerk_x[i], jerk_y[i]])
        jerk_local = np.dot(rotation_matrix, jerk_global)
        jerk_local_x[i], jerk_local_y[i] = jerk_local

    return velocity_local_x, velocity_local_y, acceleration_local_x, acceleration_local_y, jerk_local_x, jerk_local_y, angular_velocity


def normalize_angle_pi(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi


def calculate_acceleration(timestamp, heading, global_velocity_x, global_velocity_y):
    global_acceleration_x = [0 for i in timestamp]
    global_acceleration_y = [0 for i in timestamp]

    for t1 in range(len(timestamp)):
        if t1 == 0:
            global_acceleration_x[t1] = (global_velocity_x[2] - global_velocity_x[0]) / (timestamp[2] - timestamp[0])
            global_acceleration_y[t1] = (global_velocity_y[2] - global_velocity_y[0]) / (timestamp[2] - timestamp[0])
        elif t1 == len(timestamp)-1:
            global_acceleration_x[t1] = (global_velocity_x[len(timestamp) - 1] - global_velocity_x[len(timestamp) - 3]) / \
                                        (timestamp[len(timestamp) - 1] - timestamp[len(timestamp) - 3])
            global_acceleration_y[t1] = (global_velocity_y[len(timestamp) - 1] - global_velocity_y[len(timestamp) - 3]) / \
                                        (timestamp[len(timestamp) - 1] - timestamp[len(timestamp) - 3])
        else:
            global_acceleration_x[t1] = (global_velocity_x[t1+1] - global_velocity_x[t1-1]) / (timestamp[t1+1] - timestamp[t1-1])
            global_acceleration_y[t1] = (global_velocity_y[t1+1] - global_velocity_y[t1-1]) / (timestamp[t1+1] - timestamp[t1-1])

        local_acceleration_x = global_acceleration_x * np.cos(heading) + global_acceleration_y * np.sin(heading)
        local_acceleration_y = global_acceleration_y * np.cos(heading) - global_acceleration_x * np.sin(heading)

    acceleration = np.sqrt(np.power(global_acceleration_x, 2) + np.power(global_acceleration_y, 2))
    return (acceleration, local_acceleration_x, local_acceleration_y)


def calculate_speed(global_velocity_x, global_velocity_y):
    speed = np.sqrt(np.power(global_velocity_x, 2) +
                    np.power(global_velocity_y, 2))
    return speed

def global_to_local(x_b, y_b, x_m, y_m, theta_m):
    """
    Convert a background vehicle global position into the ego vehicle local frame.

    Args:
    x_b, y_b - background vehicle global position
    x_m, y_m - ego vehicle global position
    theta_m - ego vehicle heading (radians)

    Returns:
    x_local, y_local - lateral and longitudinal distances in the ego vehicle local frame
    """
    # compute position offset
    delta_x = x_b - x_m
    delta_y = y_b - y_m

    # compute position in the local frame
    x_local = np.cos(theta_m) * delta_x + np.sin(theta_m) * delta_y
    y_local = -np.sin(theta_m) * delta_x + np.cos(theta_m) * delta_y

    return x_local, y_local


def scenario_visualization(
    data,
    traffic_events,
    scenario_id,
    result_path,
    dataset_name,
    prompt_name="",
    model=None,
    idx=None,
    draw_signals=False,
    frame_dpi=600,
):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('gray')
    ax.set_facecolor('gray')

    scenario_visualization = ScenarioVisualization(
        ts=data['extra_information']['timestamp'],
        sdc_id=data['extra_information']['sdc_id'],
    )
    scenario_visualization.plot_map_features(data, fig, ax, result_path, plot=True, extra_plot=True)

    history_scenario_t = []
    random_color_ls = []

    for t in tqdm(range(len(scenario_visualization.ts))):

        fig, ax = pickle.load(open(f"{result_path}/temp/map_features.pkl", 'rb'))

        sdc_center_x, sdc_center_y = scenario_visualization.plot_tracks(data, t, ax, history_scenario_t, plot=True)
        if draw_signals:
            scenario_visualization.plot_dynamic_map_states(ax, data, t, plot=True)
        if len(traffic_events):
            scenario_visualization.plot_traffic_event_pair(data, traffic_events, t, plot=True)

        # remove x/y axes
        ax.axis('off')
        # trans = Affine2D().rotate_deg_around(sdc_center_x, sdc_center_y, -sdc_heading_angle) + ax.transData
        # ax.set_transform(trans)
        plt.tight_layout()
        ax.set_xlim(sdc_center_x - 50, sdc_center_x + 50)
        ax.set_ylim(sdc_center_y - 50, sdc_center_y + 50)

        # plt.savefig("figure/" + str(t) + ".png", dpi=300, bbox_inches="tight", facecolor="gray")
        os.makedirs(f"{result_path}/temp/figure", exist_ok=True)
        plt.savefig(f"{result_path}/temp/figure/{t}.png", dpi=frame_dpi, bbox_inches="tight", facecolor="gray")
        # plt.show()
        plt.close()

    '''
    # # # # # # # images to video # # # # # # #
    # jpg -> avi
    img_array = []
    for name in range(data['extra_information']['scene_length']):
        image_filename = f"{result_path}/temp/figure/{name}.png"
        img = cv2.imread(image_filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    os.makedirs(f"{result_path}/{dataset_name}", exist_ok=True)
    video_save_name = f"{result_path}/{dataset_name}/{scenario_id}.mp4"
    out = cv2.VideoWriter(video_save_name, cv2.VideoWriter_fourcc(*'mp4v'), data['extra_information']['sampling_rate'], size)   # H264, XVID
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print('camera video made success')
    filelist = glob.glob(os.path.join(f"{result_path}/temp/figure/", "*.png"))
    for f in filelist:
        os.remove(f)
    '''

    import glob

    fps = data['extra_information']['sampling_rate']
    scene_length = data['extra_information']['scene_length']

    os.makedirs(f"{result_path}/{dataset_name}", exist_ok=True)
    video_name_parts = [str(scenario_id)]
    if prompt_name:
        video_name_parts.append(str(prompt_name))
    if model is not None:
        video_name_parts.append(str(model))
    if idx is not None:
        video_name_parts.append(str(idx))
    video_save_name = f"{result_path}/{dataset_name}/{'-'.join(video_name_parts)}.mp4"

    try:
        import imageio

        with imageio.get_writer(video_save_name, fps=fps, macro_block_size=None) as writer:
            for name in range(scene_length):
                image_filename = f"{result_path}/temp/figure/{name}.png"
                writer.append_data(imageio.imread(image_filename))
    except (ImportError, ValueError):
        first_frame = cv2.imread(f"{result_path}/temp/figure/0.png")
        if first_frame is None:
            raise FileNotFoundError(f"Could not read {result_path}/temp/figure/0.png")
        height, width, _ = first_frame.shape
        out = cv2.VideoWriter(
            video_save_name,
            cv2.VideoWriter_fourcc(*'mp4v'),
            float(fps),
            (width, height),
        )
        if not out.isOpened():
            raise RuntimeError(f"Could not open video writer for {video_save_name}")
        out.write(first_frame)
        for name in range(1, scene_length):
            image_filename = f"{result_path}/temp/figure/{name}.png"
            img = cv2.imread(image_filename)
            if img is None:
                raise FileNotFoundError(f"Could not read {image_filename}")
            out.write(img)
        out.release()

    print('camera video made success')

    # clean up temporary files
    filelist = glob.glob(os.path.join(f"{result_path}/temp/figure/", "*.png"))
    for f in filelist:
        os.remove(f)

    return video_save_name
