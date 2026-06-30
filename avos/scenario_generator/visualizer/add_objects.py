import os
import json
import numpy as np
import matplotlib
try:
    matplotlib.use('Qt5Agg')
except:
    matplotlib.use('TkAgg')
matplotlib.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms

from visualizer.utils import ScenarioVisualization


# ==============================================================================
# 2. Interactive scene editor (core logic updated)
# ==============================================================================
class SceneEditor:
    AGENT_CONFIG = {
        'ego': {'w': 5.0, 'h': 2.0, 'color': 'red', 'label': 'Ego Car (Main)'},
        'vehicle': {'w': 5.0, 'h': 2.0, 'color': 'yellow', 'label': 'Target Vehicle'},
        'cyclist': {'w': 2.0, 'h': 0.8, 'color': 'blue', 'label': 'Cyclist'},
        'pedestrian': {'w': 0.6, 'h': 0.6, 'color': 'green', 'label': 'Pedestrian'}
    }

    def __init__(self, ax, save_path=None):
        self.ax = ax
        self.save_path = save_path
        self.active_type = 'vehicle'

        # placement state
        self.is_placing = False
        self.center_pos = None  # center point during placement
        self.current_patches = []
        self.placed_agents = []

        # pan state
        self.is_panning = False
        self.pan_start_pos = None  # mouse position at pan start

        # event bindings
        canvas = ax.figure.canvas
        self.cid_press = canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_release = canvas.mpl_connect('button_release_event', self.on_release)  # add release event
        self.cid_move = canvas.mpl_connect('motion_notify_event', self.on_move)
        self.cid_key = canvas.mpl_connect('key_press_event', self.on_key)
        self.cid_scroll = canvas.mpl_connect('scroll_event', self.on_scroll)  # add scroll event

        # New: bind the window close event
        self.cid_close = canvas.mpl_connect('close_event', self.on_close)

        # initialize ego vehicle
        self.add_static_agent('ego', 0, 0, 0)
        self.update_title()

    # New: callback on close
    def on_close(self, event):
        if self.save_path:
            print(f"\n>>> Window closed; automatically saving image to: {self.save_path}")

            # Optional: temporarily clear the title before saving for a cleaner image, or keep it as is
            # self.ax.set_title("")

            # perform save
            # Note: use self.ax.figure to access the canvas
            try:
                self.ax.figure.savefig(self.save_path, facecolor='gray', bbox_inches='tight')
                print(">>> Save completed.")
            except Exception as e:
                print(f">>> Save failed: {e}")
        else:
            print("\n>>> Window closed; no save path was specified, so saving is skipped.")

    def update_title(self):
        curr_label = self.AGENT_CONFIG[self.active_type]['label']
        title = (f"Scene Editor | Current: [{curr_label}]\n"
                 f"Left: Place | Right: Undo | Wheel: Zoom | Mid: Pan | [S]: Save Image\n"
                 f"Keys: 1:Car 2:Cyclist 3:Pedestrian")
        self.ax.set_title(title, fontsize=10, color='white')

    def get_geometry_patches(self, w, h, color, alpha):
        # 1. rectangle
        rect = patches.Rectangle(
            xy=(-w / 2, -h / 2), width=w, height=h,
            linewidth=0.5, facecolor=color, edgecolor="black", alpha=alpha, zorder=20
        )
        # 2. equilateral triangle arrow
        side_len = h * 0.6
        altitude = side_len * np.sqrt(3) / 2
        point_a = (w / 2, 0)
        point_b = (w / 2 - altitude, side_len / 2)
        point_c = (w / 2 - altitude, -side_len / 2)
        arrow = patches.Polygon(
            xy=[point_a, point_b, point_c],
            linewidth=0.5, facecolor='none', edgecolor="black", alpha=alpha, zorder=21
        )
        return rect, arrow

    def add_static_agent(self, type_key, x, y, heading_deg):
        cfg = self.AGENT_CONFIG[type_key]
        rect, arrow = self.get_geometry_patches(cfg['w'], cfg['h'], cfg['color'], alpha=0.9)
        t = transforms.Affine2D().rotate_deg(heading_deg).translate(x, y) + self.ax.transData
        rect.set_transform(t)
        arrow.set_transform(t)
        self.ax.add_patch(rect)
        self.ax.add_patch(arrow)
        self.placed_agents.append({'type': type_key, 'x': x, 'y': y, 'h': heading_deg, 'patches': [rect, arrow]})

    def on_key(self, event):
        if self.is_placing: return

        if event.key == 's':
            if self.save_path:
                print(f">>> Saving image to: {self.save_path}")
                # temporarily remove the title to save a clean image (optional)
                original_title = self.ax.get_title()
                self.ax.set_title("")

                # save image
                self.ax.figure.savefig(self.save_path, facecolor='gray', bbox_inches='tight')

                # restore title and notify
                self.ax.set_title(original_title + " [SAVED!]")
                self.ax.figure.canvas.draw()
                print(">>> Image Saved Successfully!")
            else:
                print(">>> No save path specified!")

        elif event.key == '1':
            self.active_type = 'vehicle'
        elif event.key == '2':
            self.active_type = 'cyclist'
        elif event.key == '3':
            self.active_type = 'pedestrian'

        if event.key != 's':
            self.update_title()

    # --- New: scroll-wheel zoom logic ---
    def on_scroll(self, event):
        if event.inaxes != self.ax: return

        # zoom factor (1.2 means 20% per step)
        base_scale = 1.2
        if event.button == 'up':  # scroll up -> zoom in
            scale_factor = 1 / base_scale
        elif event.button == 'down':  # scroll down -> zoom out
            scale_factor = base_scale
        else:
            return

        # get current view limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # compute mouse position relative to the current view (0.0-1.0)
        # this keeps zoom centered on the mouse position
        xdata, ydata = event.xdata, event.ydata
        x_left = xdata - xlim[0]
        x_right = xlim[1] - xdata
        y_bottom = ydata - ylim[0]
        y_top = ylim[1] - ydata

        # apply zoom
        self.ax.set_xlim([xdata - x_left * scale_factor, xdata + x_right * scale_factor])
        self.ax.set_ylim([ydata - y_bottom * scale_factor, ydata + y_top * scale_factor])

        self.ax.figure.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax: return

        # 1. middle button (Button 2): start panning
        if event.button == 2:
            self.is_panning = True
            # record the starting mouse position in pixel coordinates (event.x), since data coordinates change with the view
            self.pan_start_pos = (event.x, event.y)
            return

        # 2. left button (Button 1): placement logic
        if event.button == 1:
            if not self.is_placing:
                # first click: set position
                self.is_placing = True
                self.center_pos = (event.xdata, event.ydata)

                cfg = self.AGENT_CONFIG[self.active_type]
                rect, arrow = self.get_geometry_patches(cfg['w'], cfg['h'], cfg['color'], alpha=0.6)

                t = transforms.Affine2D().translate(*self.center_pos) + self.ax.transData
                rect.set_transform(t)
                arrow.set_transform(t)

                self.ax.add_patch(rect)
                self.ax.add_patch(arrow)
                self.current_patches = [rect, arrow]
                self.ax.figure.canvas.draw()
            else:
                # second click: set heading and finish
                self.is_placing = False
                dx = event.xdata - self.center_pos[0]
                dy = event.ydata - self.center_pos[1]
                heading_deg = np.degrees(np.arctan2(dy, dx))

                for p in self.current_patches: p.set_alpha(0.9)

                self.placed_agents.append({
                    'type': self.active_type, 'x': self.center_pos[0], 'y': self.center_pos[1],
                    'h': heading_deg, 'patches': self.current_patches
                })
                self.current_patches = []
                self.ax.figure.canvas.draw()

        # 3. right button (Button 3): undo
        elif event.button == 3:
            if self.is_placing:
                self.is_placing = False
                for p in self.current_patches: p.remove()
                self.current_patches = []
            elif self.placed_agents:
                last = self.placed_agents.pop()
                for p in last['patches']: p.remove()
                print(f"Removed agent: {last['type']}")
            self.ax.figure.canvas.draw()

    # --- New: mouse release event (ends panning) ---
    def on_release(self, event):
        if event.button == 2 and self.is_panning:
            self.is_panning = False
            self.pan_start_pos = None

    def on_move(self, event):
        if event.inaxes != self.ax: return

        # 1. handle panning first
        if self.is_panning and self.pan_start_pos:
            # compute pixel delta
            dx_pix = event.x - self.pan_start_pos[0]
            dy_pix = event.y - self.pan_start_pos[1]

            # convert pixel delta to data-coordinate delta
            # ax.transData.inverted() converts pixel coordinates back to data coordinates
            # only the scale matters here, so use the current view width
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

            # get Axes width and height in pixels
            bbox = self.ax.get_window_extent()
            width_pix, height_pix = bbox.width, bbox.height

            # compute movement ratio
            dx_data = -(dx_pix / width_pix) * (xlim[1] - xlim[0])
            dy_data = -(dy_pix / height_pix) * (ylim[1] - ylim[0])

            self.ax.set_xlim([xlim[0] + dx_data, xlim[1] + dx_data])
            self.ax.set_ylim([ylim[0] + dy_data, ylim[1] + dy_data])

            # update the start point for continuous dragging
            self.pan_start_pos = (event.x, event.y)
            self.ax.figure.canvas.draw_idle()
            return

        # 2. handle rotation preview during placement
        if self.is_placing and self.center_pos:
            dx = event.xdata - self.center_pos[0]
            dy = event.ydata - self.center_pos[1]
            angle_rad = np.arctan2(dy, dx)
            t = transforms.Affine2D().rotate(angle_rad).translate(*self.center_pos) + self.ax.transData
            for p in self.current_patches:
                p.set_transform(t)
            self.ax.figure.canvas.draw_idle()

    def get_result_data(self):
        return [{k: v for k, v in item.items() if k != 'patches'} for item in self.placed_agents]


def add_obj(data, args, scenario_id):
    # --- visualization class initialization ---
    # if using the real class: from xxx import ScenarioVisualization
    scenario_visualization = ScenarioVisualization(
        ts=data['extra_information']['timestamp'],
        sdc_id=data['extra_information']['sdc_id'],
    )

    # --- plot setup ---
    fig = plt.figure(figsize=(12, 12))
    ax = plt.gca()
    fig.patch.set_facecolor('gray')
    ax.set_facecolor('gray')

    # --- draw original data (non-blocking) ---
    data = scenario_visualization.preprocess(data)
    scenario_visualization.plot_map_features(data, fig, ax, args.output_path2, plot=True, extra_plot=True)
    # scenario_visualization.plot_dynamic_map_states(ax, data, t=0, plot=True)

    ax.add_patch(patches.Rectangle(xy=(-50, -50), width=100, height=100, linewidth=1, edgecolor='r', facecolor='none',
                                   linestyle='dashed'))

    # --- start interactive editor ---
    # set a reasonable view range; otherwise the plot may appear entirely gray
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_aspect('equal')
    ax.axis('off')

    save_file_path = os.path.join(args.output_path1, args.subset_name, scenario_id, 'llm_image.png')

    print(">>> Interactive mode started.")
    print(">>> Close the window after editing to save automatically.")
    print(">>> 1: vehicle (yellow), 2: cyclist (blue), 3: pedestrian (green)")

    editor = SceneEditor(ax, save_path=save_file_path)

    if not os.path.exists(os.path.join(args.output_path1, args.subset_name, scenario_id)):
        os.makedirs(os.path.join(args.output_path1, args.subset_name, scenario_id))

    plt.tight_layout()
    # plt.savefig(os.path.join(args.output_path, scenario_id, 'llm_image.png'))
    plt.show()  # block until the window is closed
    plt.close()

    # --- save data ---
    agents_list = editor.get_result_data()

    # build save dictionary
    save_payload = {
        "scenario_id": scenario_id,
        "timestamp": str(data['extra_information']['timestamp']),
        "agents": []
    }

    print(f"\n--- Saved results (including ego vehicle) ---")
    print(f"{'Type':<12} | {'X':<8} | {'Y':<8} | {'Heading(deg)':<8}")
    print("-" * 50)

    for ag in agents_list:
        # convert to Python float for JSON serialization compatibility
        entry = {
            "type": ag['type'],
            "x": float(ag['x']),
            "y": float(ag['y']),
            "heading_degrees": float(ag['h']),
            "heading_radians": float(np.radians(ag['h']))
        }
        save_payload["agents"].append(entry)
        print(f"{entry['type']:<12} | {entry['x']:<8.2f} | {entry['y']:<8.2f} | {entry['heading_degrees']:<8.2f}")

    with open(os.path.join(args.output_path1, args.subset_name, scenario_id, 'edited_agents.json'), 'w') as f:
        json.dump(save_payload, f, indent=4)

    return save_payload['agents']
