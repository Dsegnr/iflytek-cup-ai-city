from itertools import product
import numpy as np
import pandas as pd
import requests
import time
import hashlib
import urllib.parse
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib.animation as animation
from datetime import datetime
import json
import threading
import queue


# ----------------------Matplotlib中文字体配置----------------------
def setup_matplotlib_chinese():
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'PingFang SC', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("[初始化] Matplotlib中文字体配置完成")
    except Exception as e:
        print(f"[初始化] 中文字体配置警告：{str(e)}，将使用默认字体")


setup_matplotlib_chinese()

# ----------------------全局配置----------------------
API_CONFIG = {
    "host": "http://api.map.baidu.com",
    "path": "/traffic/v1/bound",
    "ak": "c4hUBLXvShVeLlkJ5VnODjLHa0ZkENy2",
    "sk": "WAwHCfjxdr1wUbRVmYkehv3v3GyIbegN",
    "params": {
        "bounds": "39.985151,116.392689;39.991003,116.404921",
        "coord_type": "bd09ll",
        "radius": 1500,
        "output": "json",
        "traffic_fields": {
            "road_name": "road_name",
            "status": "traffic_status",
            "speed": "avg_speed",
            "congestion_length": "lane_congestion_length",
            "update_time": "timestamp"
        }
    }
}

CONTROLLED_LANES = {
    "A": {
        "中文名称": "中关村路口",
        "location": {"lng": 116.3950, "lat": 39.9100},
        "map_xy": (116.3950, 39.9100),
        "lanes": {
            "A1": {"转向": "直行", "downstream_lanes": ["B1"], "宽度": 3.5, "道路名称": "中关村大街",
                   "lane_xy": (116.3935, 39.9100),
                   "traffic_light_pos": (116.3935, 39.9085),
                   "label_pos": (116.3942, 39.9112)},
            "A2": {"转向": "左转", "downstream_lanes": ["B2"], "宽度": 3.5, "道路名称": "中关村大街",
                   "lane_xy": (116.3965, 39.9100),
                   "traffic_light_pos": (116.3965, 39.9085),
                   "label_pos": (116.3972, 39.9112)}
        }
    },
    "B": {
        "中文名称": "海淀路口",
        "location": {"lng": 116.4020, "lat": 39.9130},
        "map_xy": (116.4020, 39.9130),
        "lanes": {
            "B1": {"转向": "直行", "downstream_lanes": ["D1"], "宽度": 3.5, "道路名称": "海淀大街",
                   "lane_xy": (116.4005, 39.9130),
                   "traffic_light_pos": (116.4005, 39.9115),
                   "label_pos": (116.4012, 39.9142)},
            "B2": {"转向": "左转", "downstream_lanes": [], "宽度": 3.5, "道路名称": "海淀大街",
                   "lane_xy": (116.4035, 39.9130),
                   "traffic_light_pos": (116.4035, 39.9115),
                   "label_pos": (116.4042, 39.9142)}
        }
    },
    "C": {
        "中文名称": "清华园路口",
        "location": {"lng": 116.3880, "lat": 39.9160},
        "map_xy": (116.3880, 39.9160),
        "lanes": {
            "C1": {"转向": "直行", "downstream_lanes": ["A1"], "宽度": 3.5, "道路名称": "中关村北大街",
                   "lane_xy": (116.3865, 39.9160),
                   "traffic_light_pos": (116.3865, 39.9145),
                   "label_pos": (116.3872, 39.9172)}
        }
    },
    "D": {
        "中文名称": "苏州街路口",
        "location": {"lng": 116.4090, "lat": 39.9090},
        "map_xy": (116.4090, 39.9090),
        "lanes": {
            "D1": {"转向": "直行", "downstream_lanes": [], "宽度": 3.5, "道路名称": "海淀大街",
                   "lane_xy": (116.4075, 39.9090),
                   "traffic_light_pos": (116.4075, 39.9075),
                   "label_pos": (116.4082, 39.9102)}
        }
    }
}

VIS_CONFIG = {
    "fig_size": (10, 7),
    "zoom": 0.003,
    "layers": {
        "road": {"color": "#8B4513", "linewidth": 9, "alpha": 0.7},
        "intersection": {"color": "#00008B", "size": 0.001, "alpha": 0.9},
        "lane": {"linewidth": 2, "edgecolor": "#000000", "alpha": 0.85},
        "text": {"fontsize": 7, "color": "#000000"},
        "congestion_label": {"fontsize": 6.5, "alpha": 0.95},
        "control_cmd": {"bg_color": "#FFF8DC", "text_color": "#DC143C", "fontsize": 6.5, "alpha": 0.95},
        "traffic_light": {"size": 0.0004},
        "hover_info": {"bg_color": "#FFFFE0", "text_color": "#000000", "fontsize": 8, "alpha": 0.95}
    },
    "congestion_style": {
        "轻度拥堵": {
            "color": "#90EE90",
            "border_color": "#228B22",
            "label": "🟢 轻度",
            "prob_color": "#228B22"
        },
        "中度拥堵": {
            "color": "#FFD700",
            "border_color": "#DAA520",
            "label": "🟡 中度",
            "prob_color": "#DAA520"
        },
        "重度拥堵": {
            "color": "#FF6347",
            "border_color": "#B22222",
            "label": "🔴 重度",
            "prob_color": "#B22222"
        },
        "无数据": {
            "color": "#D3D3D3",
            "border_color": "#808080",
            "label": "⚫ 无数据",
            "prob_color": "#696969"
        }
    },
    "traffic_light_color": {
        "绿灯": "#32CD32",
        "红灯": "#DC143C",
        "黄灯": "#FFD700",
        "off": "#F5F5F5"
    },
    "arrow_props": {
        "arrowstyle": "->",
        "connectionstyle": "arc3,rad=0.05",
        "linewidth": 1.8,
        "color": "#006400"
    },
    "update_interval": 60,
    "refresh_rate": 300  # 毫秒
}

CHINESE_MAPPING = {
    "intersection_id": "路口ID",
    "lane_id": "车道ID",
    "turn_type": "转向类型",
    "predicted_level": "拥堵等级",
    "heavy_congestion_prob": "重度拥堵概率",
    "light": "轻度拥堵",
    "moderate": "中度拥堵",
    "heavy": "重度拥堵",
    "straight": "直行",
    "left": "左转",
    "right": "右转",
    "路口ID": "路口中文名称",
    "绿灯调整指令": "绿灯调整指令"
}

MODEL_CONFIG = {
    "congestion_threshold": 0.7,
    "prediction_horizon": 10,
    "q_learning": {
        "episodes": 600,
        "alpha": 0.15,
        "gamma": 0.9,
        "epsilon": 0.1
    },
    "random_forest": {
        "n_estimators": 100,
        "random_state": 42
    },
    "lane_config": {
        "max_queue_per_lane": 0.05,
        "base_pass_per_second": 0.1
    },
    "real_data_config": {
        "data_path": "real_lane_traffic_data.csv",
        "seq_len": 8
    }
}


# ----------------------线程安全的数据管理器----------------------
class TrafficDataManager:
    def __init__(self):
        self.congestion_data = pd.DataFrame()
        self.control_cmds = pd.DataFrame()
        self.current_time = None
        self.lock = threading.Lock()
        self.data_version = 0
        self.update_callbacks = []

    def update_data(self, congestion_data, control_cmds=None, current_time=None):
        with self.lock:
            if congestion_data.empty:
                self.congestion_data = pd.DataFrame(
                    columns=["路口ID", "车道ID", "拥堵等级", "重度拥堵概率(%)", "转向类型"])
            else:
                self.congestion_data = congestion_data.copy(deep=True)
            self.control_cmds = control_cmds.copy(
                deep=True) if control_cmds is not None and not control_cmds.empty else pd.DataFrame()
            self.current_time = current_time
            self.data_version += 1

    def get_data(self):
        with self.lock:
            return (self.congestion_data.copy(deep=True),
                    self.control_cmds.copy(deep=True),
                    self.current_time,
                    self.data_version)

    def add_update_callback(self, callback):
        self.update_callbacks.append(callback)

    def notify_callbacks(self):
        for callback in self.update_callbacks:
            callback()


# ----------------------鼠标悬停交互模块----------------------
class HoverManager:
    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.hover_annotation = None
        self.current_hover_item = None
        self._setup_hover_events()

    def _setup_hover_events(self):
        """设置鼠标悬停事件监听"""
        self.visualizer.fig.canvas.mpl_connect("motion_notify_event", self._on_hover)

    def _on_hover(self, event):
        """鼠标移动事件处理"""
        if event.inaxes != self.visualizer.ax:
            self._hide_annotation()
            return

        found_item = None
        hover_text = ""

        # 检查是否悬停在车道上
        for lane_key, lane_dict in self.visualizer.lane_patches.items():
            patch = lane_dict['patch']
            if patch.contains_point([event.x, event.y]):
                found_item = lane_key
                hover_text = self._get_lane_hover_info(lane_key, lane_dict)
                break

        # 检查是否悬停在交通信号灯上
        if not found_item:
            for light_key, light_dict in self.visualizer.traffic_lights.items():
                if hasattr(light_dict['base'], 'contains_point') and light_dict['base'].contains_point(
                        [event.x, event.y]):
                    found_item = light_key
                    hover_text = self._get_traffic_light_hover_info(light_key, light_dict)
                    break

        if found_item and found_item != self.current_hover_item:
            self._show_annotation(event, hover_text, found_item)
        elif not found_item and self.current_hover_item:
            self._hide_annotation()

    def _get_lane_hover_info(self, lane_key, lane_dict):
        """获取车道悬停信息"""
        parts = lane_key.split('-')
        intersection_id = parts[0]
        lane_id = parts[1]

        lane_info = lane_dict['lane_info']
        intersection_name = self.visualizer.controlled_lanes[intersection_id]['中文名称']

        # 获取当前拥堵数据
        congestion_info = self._get_current_congestion_info(intersection_id, lane_id)

        info_lines = [
            f"📍 {intersection_name} - {lane_id}",
            f"🛣️ 道路: {lane_info['道路名称']}",
            f"↩️ 转向: {lane_info['转向']}",
            f"📏 宽度: {lane_info['宽度']}米",
            f"🚦 状态: {congestion_info['status']}",
            f"📊 拥堵等级: {congestion_info['level']}",
            f"🎯 重度拥堵概率: {congestion_info['heavy_prob']}%",
            f"🕒 更新时间: {congestion_info['update_time']}"
        ]

        return "\\n".join(info_lines)

    def _get_traffic_light_hover_info(self, light_key, light_dict):
        """获取交通信号灯悬停信息"""
        parts = light_key.split('-')
        intersection_id = parts[0]
        lane_id = parts[1]

        intersection_name = self.visualizer.controlled_lanes[intersection_id]['中文名称']
        current_state = light_dict.get('current_state', '未知')

        info_lines = [
            f"🚦 {intersection_name} - {lane_id} 信号灯",
            f"💡 当前状态: {current_state}",
            f"📍 位置: 经度 {light_dict['pos'][0]:.4f}, 纬度 {light_dict['pos'][1]:.4f}"
        ]

        return "\\n".join(info_lines)

    def _get_current_congestion_info(self, intersection_id, lane_id):
        """获取当前拥堵信息"""
        default_info = {
            'status': '无数据',
            'level': '未知',
            'heavy_prob': '0',
            'update_time': '--'
        }

        if hasattr(self.visualizer, 'current_congestion_data') and not self.visualizer.current_congestion_data.empty:
            df = self.visualizer.current_congestion_data
            row = df[(df['路口ID'] == intersection_id) & (df['车道ID'] == lane_id)]
            if not row.empty:
                return {
                    'status': '正常' if row.iloc[0]['拥堵等级'] != '重度拥堵' else '异常',
                    'level': row.iloc[0]['拥堵等级'],
                    'heavy_prob': f"{row.iloc[0]['重度拥堵概率(%)']:.1f}",
                    'update_time': datetime.now().strftime("%H:%M:%S")
                }

        return default_info

    def _show_annotation(self, event, text, item_key):
        """显示悬停注释"""
        if self.hover_annotation:
            self.hover_annotation.remove()

        self.hover_annotation = self.visualizer.ax.annotate(
            text,
            xy=(event.xdata, event.ydata),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=VIS_CONFIG["layers"]["hover_info"]["bg_color"],
                edgecolor="#888888",
                alpha=VIS_CONFIG["layers"]["hover_info"]["alpha"]
            ),
            fontsize=VIS_CONFIG["layers"]["hover_info"]["fontsize"],
            color=VIS_CONFIG["layers"]["hover_info"]["text_color"],
            zorder=1000
        )

        self.current_hover_item = item_key
        self.visualizer.fig.canvas.draw_idle()

    def _hide_annotation(self):
        """隐藏悬停注释"""
        if self.hover_annotation:
            self.hover_annotation.remove()
            self.hover_annotation = None
            self.current_hover_item = None
            self.visualizer.fig.canvas.draw_idle()


# ----------------------核心可视化模块（使用动画而非线程）----------------------
class TrafficGeoVisualizer:
    def __init__(self, controlled_lanes, vis_config, data_manager):
        self.controlled_lanes = controlled_lanes
        self.vis_config = vis_config
        self.data_manager = data_manager
        self.fig, self.ax = plt.subplots(figsize=vis_config["fig_size"])

        self.ax.set_title("车道级交通管控实时状态地图 (鼠标悬停查看详细信息)", fontsize=13, pad=12)
        self.ax.set_xlabel("经度（°E）", fontsize=9)
        self.ax.set_ylabel("纬度（°N）", fontsize=9)

        self._draw_road_network()
        self._draw_intersections()
        self.lane_patches = self._draw_lanes()
        self.traffic_lights = self._init_traffic_lights()
        self.dynamic_elements = {
            "congestion_labels": {},
            "control_cmds": {}
        }
        self.status_panel = self._init_status_panel()
        self.current_congestion_data = pd.DataFrame()
        self.last_data_version = 0

        self._adjust_map_extent()

        # 初始化鼠标悬停管理器
        self.hover_manager = HoverManager(self)

        # 设置动画更新
        self.ani = animation.FuncAnimation(
            self.fig, self._animate_update,
            interval=VIS_CONFIG["refresh_rate"],
            blit=False, cache_frame_data=False
        )

        plt.tight_layout()

    def _animate_update(self, frame):
        """动画更新函数 - 在主线程中安全执行"""
        congestion_data, control_cmds, current_time, current_version = self.data_manager.get_data()

        # 只有数据版本更新时才重绘
        if current_version > self.last_data_version:
            self.update_map(congestion_data, control_cmds, current_time)
            self.last_data_version = current_version

        return []

    def _adjust_map_extent(self):
        all_x = []
        all_y = []
        for cfg in self.controlled_lanes.values():
            all_x.append(cfg["map_xy"][0])
            all_y.append(cfg["map_xy"][1])
            for lane in cfg["lanes"].values():
                all_x.append(lane["lane_xy"][0])
                all_y.append(lane["lane_xy"][1])

        x_min, x_max = min(all_x) - self.vis_config["zoom"], max(all_x) + self.vis_config["zoom"]
        y_min, y_max = min(all_y) - self.vis_config["zoom"], max(all_y) + self.vis_config["zoom"]
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)

    def _draw_road_network(self):
        road_connections = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "A"), ("D", "B")]
        for (id1, id2) in road_connections:
            xy1 = self.controlled_lanes[id1]["map_xy"]
            xy2 = self.controlled_lanes[id2]["map_xy"]
            self.ax.plot(
                [xy1[0], xy2[0]], [xy1[1], xy2[1]],
                color=self.vis_config["layers"]["road"]["color"],
                linewidth=self.vis_config["layers"]["road"]["linewidth"],
                alpha=self.vis_config["layers"]["road"]["alpha"]
            )
            mid_x = (xy1[0] + xy2[0]) / 2
            mid_y = (xy1[1] + xy2[1]) / 2 + 0.0015
            road_name = self._get_road_name(id1, id2)
            self.ax.text(
                mid_x, mid_y, road_name,
                ha="center", va="bottom", fontsize=self.vis_config["layers"]["text"]["fontsize"],
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.6)
            )

    def _get_road_name(self, id1, id2):
        lanes1 = self.controlled_lanes[id1]["lanes"].values()
        lanes2 = self.controlled_lanes[id2]["lanes"].values()
        common_roads = set([lane["道路名称"] for lane in lanes1]) & set([lane["道路名称"] for lane in lanes2])
        return common_roads.pop() if common_roads else "无名道路"

    def _draw_intersections(self):
        for id, cfg in self.controlled_lanes.items():
            x, y = cfg["map_xy"]
            circle = plt.Circle(
                (x, y), self.vis_config["layers"]["intersection"]["size"],
                color=self.vis_config["layers"]["intersection"]["color"],
                alpha=self.vis_config["layers"]["intersection"]["alpha"]
            )
            self.ax.add_patch(circle)
            self.ax.text(
                x, y + 0.002, f"{cfg['中文名称']}\\n(ID:{id})",
                ha="center", va="bottom", fontsize=self.vis_config["layers"]["text"]["fontsize"],
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7)
            )

    def _draw_lanes(self):
        lane_patches = {}
        for id, cfg in self.controlled_lanes.items():
            for lane_id, lane_info in cfg["lanes"].items():
                x, y = lane_info["lane_xy"]
                lane_width = lane_info["宽度"] * 0.00035
                rect = Rectangle(
                    (x - lane_width / 2, y - 0.0007),
                    lane_width, 0.0014,
                    linewidth=self.vis_config["layers"]["lane"]["linewidth"],
                    edgecolor=self.vis_config["congestion_style"]["无数据"]["border_color"],
                    facecolor=self.vis_config["congestion_style"]["无数据"]["color"],
                    alpha=self.vis_config["layers"]["lane"]["alpha"]
                )
                self.ax.add_patch(rect)
                self.ax.text(
                    x, y, f"{lane_id}\\n{lane_info['转向']}",
                    ha="center", va="center", fontsize=self.vis_config["layers"]["text"]["fontsize"] - 0.5,
                    fontweight="bold", color="#000000"
                )
                self._add_turn_arrow(x, y, lane_info["转向"])
                lane_patches[f"{id}-{lane_id}"] = {
                    "patch": rect,
                    "lane_info": lane_info
                }
        return lane_patches

    def _add_turn_arrow(self, x, y, turn_type):
        arrow_len = 0.0005
        if turn_type == "直行":
            arrow = FancyArrowPatch(
                (x - arrow_len, y), (x + arrow_len, y),
                **self.vis_config["arrow_props"]
            )
        elif turn_type == "左转":
            arrow = FancyArrowPatch(
                (x, y - arrow_len / 2), (x - arrow_len, y + arrow_len / 2),
                **self.vis_config["arrow_props"]
            )
        elif turn_type == "右转":
            arrow = FancyArrowPatch(
                (x, y - arrow_len / 2), (x + arrow_len, y + arrow_len / 2),
                **self.vis_config["arrow_props"]
            )
        else:
            return
        self.ax.add_patch(arrow)

    def _init_traffic_lights(self):
        traffic_lights = {}
        for id, cfg in self.controlled_lanes.items():
            for lane_id, lane_info in cfg["lanes"].items():
                light_pos = lane_info["traffic_light_pos"]
                light_size = self.vis_config["layers"]["traffic_light"]["size"]
                base = Rectangle(
                    (light_pos[0] - light_size, light_pos[1] - 3 * light_size),
                    2 * light_size, 6 * light_size,
                    color="#696969", alpha=0.7
                )
                self.ax.add_patch(base)
                red_light = plt.Circle(
                    (light_pos[0], light_pos[1] + light_size),
                    light_size * 0.7,
                    color=self.vis_config["traffic_light_color"]["off"], alpha=0.5
                )
                green_light = plt.Circle(
                    (light_pos[0], light_pos[1] - light_size),
                    light_size * 0.7,
                    color=self.vis_config["traffic_light_color"]["off"], alpha=0.5
                )
                self.ax.add_patch(red_light)
                self.ax.add_patch(green_light)
                traffic_lights[f"{id}-{lane_id}"] = {
                    "base": base,
                    "red": red_light,
                    "green": green_light,
                    "pos": light_pos,
                    "current_state": None
                }
        return traffic_lights

    def _init_status_panel(self):
        return self.ax.text(
            0.02, 0.98, "", transform=self.ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="lightblue", alpha=0.85)
        )

    def _clear_dynamic_elements(self):
        """清理所有动态元素"""
        for label in self.dynamic_elements["congestion_labels"].values():
            if label in self.ax.texts:
                self.ax.texts.remove(label)
        for cmd in self.dynamic_elements["control_cmds"].values():
            if cmd in self.ax.texts:
                self.ax.texts.remove(cmd)
        self.dynamic_elements["congestion_labels"].clear()
        self.dynamic_elements["control_cmds"].clear()

    def _update_lane_color(self, congestion_data):
        """更新车道颜色"""
        for _, row in congestion_data.iterrows():
            lane_key = f"{row['路口ID']}-{row['车道ID']}"
            if lane_key in self.lane_patches:
                patch = self.lane_patches[lane_key]["patch"]
                congestion_level = row["拥堵等级"]
                style = self.vis_config["congestion_style"].get(
                    congestion_level,
                    self.vis_config["congestion_style"]["无数据"]
                )
                patch.set_facecolor(style["color"])
                patch.set_edgecolor(style["border_color"])

    def _update_congestion_labels(self, congestion_data):
        """更新拥堵标签"""
        for _, row in congestion_data.iterrows():
            lane_key = f"{row['路口ID']}-{row['车道ID']}"
            if lane_key in self.lane_patches:
                lane_info = self.lane_patches[lane_key]["lane_info"]
                label_pos = lane_info["label_pos"]
                congestion_level = row["拥堵等级"]
                heavy_prob = row["重度拥堵概率(%)"]

                style = self.vis_config["congestion_style"].get(
                    congestion_level,
                    self.vis_config["congestion_style"]["无数据"]
                )

                label_text = f"{style['label']}\\n{heavy_prob:.1f}%"
                label = self.ax.text(
                    label_pos[0], label_pos[1], label_text,
                    ha="center", va="bottom",
                    fontsize=self.vis_config["layers"]["congestion_label"]["fontsize"],
                    color=style["prob_color"],
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9)
                )
                self.dynamic_elements["congestion_labels"][lane_key] = label

    def _update_traffic_lights(self, control_cmds):
        """更新交通信号灯状态"""
        for _, row in control_cmds.iterrows():
            lane_key = f"{row['路口ID']}-{row['车道ID']}"
            if lane_key in self.traffic_lights:
                light_dict = self.traffic_lights[lane_key]
                command = row.get("绿灯调整指令", "保持")

                if "延长" in command:
                    light_state = "绿灯"
                elif "缩短" in command:
                    light_state = "红灯"
                else:
                    light_state = light_dict.get("current_state", "绿灯")

                # 更新灯光颜色
                if light_state == "绿灯":
                    light_dict["green"].set_color(self.vis_config["traffic_light_color"]["绿灯"])
                    light_dict["red"].set_color(self.vis_config["traffic_light_color"]["off"])
                else:
                    light_dict["green"].set_color(self.vis_config["traffic_light_color"]["off"])
                    light_dict["red"].set_color(self.vis_config["traffic_light_color"]["红灯"])

                light_dict["current_state"] = light_state

    def _update_control_commands(self, control_cmds):
        """更新控制命令显示"""
        for _, row in control_cmds.iterrows():
            lane_key = f"{row['路口ID']}-{row['车道ID']}"
            if lane_key in self.lane_patches:
                lane_info = self.lane_patches[lane_key]["lane_info"]
                label_pos = lane_info["label_pos"]

                cmd_text = f"控制: {row['绿灯调整指令']}"
                cmd_label = self.ax.text(
                    label_pos[0], label_pos[1] - 0.001,
                    cmd_text,
                    ha="center", va="top",
                    fontsize=self.vis_config["layers"]["control_cmd"]["fontsize"],
                    color=self.vis_config["layers"]["control_cmd"]["text_color"],
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor=self.vis_config["layers"]["control_cmd"]["bg_color"],
                        alpha=self.vis_config["layers"]["control_cmd"]["alpha"]
                    )
                )
                self.dynamic_elements["control_cmds"][lane_key] = cmd_label

    def _update_status_panel(self, congestion_data, current_time):
        """更新状态面板"""
        if congestion_data.empty:
            status_text = "暂无交通数据"
        else:
            total_lanes = len(congestion_data)
            heavy_count = len(congestion_data[congestion_data["拥堵等级"] == "重度拥堵"])
            moderate_count = len(congestion_data[congestion_data["拥堵等级"] == "中度拥堵"])

            status_text = (
                f"更新时间: {current_time.strftime('%H:%M:%S') if current_time else '--'}\\n"
                f"监控车道: {total_lanes}条\\n"
                f"🚦 重度拥堵: {heavy_count}条\\n"
                f"🚦 中度拥堵: {moderate_count}条\\n"
                f"🖱️ 提示: 鼠标悬停查看详情"
            )

        self.status_panel.set_text(status_text)

    def update_map(self, congestion_data, control_cmds=None, current_time=None):
        """更新地图显示"""
        # 保存当前数据用于悬停显示
        self.current_congestion_data = congestion_data.copy()

        # 清理动态元素
        self._clear_dynamic_elements()

        # 更新所有可视化元素
        self._update_lane_color(congestion_data)
        self._update_congestion_labels(congestion_data)

        if control_cmds is not None and not control_cmds.empty:
            self._update_traffic_lights(control_cmds)
            self._update_control_commands(control_cmds)

        self._update_status_panel(congestion_data, current_time)

        # 刷新显示
        self.fig.canvas.draw_idle()


# ----------------------模拟数据生成（用于测试）----------------------
def generate_sample_data():
    """生成模拟测试数据"""
    congestion_data = []
    control_cmds = []

    for intersection_id, cfg in CONTROLLED_LANES.items():
        for lane_id in cfg["lanes"].keys():
            # 随机生成拥堵数据
            congestion_levels = ["轻度拥堵", "中度拥堵", "重度拥堵"]
            level = np.random.choice(congestion_levels, p=[0.6, 0.3, 0.1])
            heavy_prob = np.random.uniform(0, 100)

            congestion_data.append({
                "路口ID": intersection_id,
                "车道ID": lane_id,
                "拥堵等级": level,
                "重度拥堵概率(%)": heavy_prob,
                "转向类型": cfg["lanes"][lane_id]["转向"]
            })

            # 生成控制命令
            commands = ["延长绿灯15秒", "缩短绿灯10秒", "保持当前状态"]
            command = np.random.choice(commands, p=[0.4, 0.3, 0.3])

            control_cmds.append({
                "路口ID": intersection_id,
                "车道ID": lane_id,
                "绿灯调整指令": command
            })

    return pd.DataFrame(congestion_data), pd.DataFrame(control_cmds)


# ----------------------数据更新线程----------------------
class DataUpdateThread(threading.Thread):
    def __init__(self, data_manager, update_interval=3):
        super().__init__(daemon=True)
        self.data_manager = data_manager
        self.update_interval = update_interval
        self.is_running = True

    def run(self):
        while self.is_running:
            congestion_data, control_cmds = generate_sample_data()
            current_time = datetime.now()
            self.data_manager.update_data(congestion_data, control_cmds, current_time)
            time.sleep(self.update_interval)

    def stop(self):
        self.is_running = False


# ----------------------主测试函数----------------------
def main():
    """主测试函数"""
    print("🚦 启动车道级交通管控可视化系统...")
    print("🖱️ 提示: 将鼠标悬停在车道或信号灯上查看详细信息")

    # 创建数据管理器
    data_manager = TrafficDataManager()

    # 创建可视化器
    visualizer = TrafficGeoVisualizer(CONTROLLED_LANES, VIS_CONFIG, data_manager)

    # 创建数据更新线程
    data_thread = DataUpdateThread(data_manager, update_interval=3)
    data_thread.start()

    try:
        print("测试完成! 系统运行正常。")
        print("现在您可以:")
        print("1. 将鼠标悬停在车道上查看详细信息")
        print("2. 将鼠标悬停在信号灯上查看状态")
        print("3. 观察实时更新的拥堵状态和控制指令")

        # 显示图形（在主线程中）
        plt.show()

    except KeyboardInterrupt:
        print("\\n正在关闭系统...")
        data_thread.stop()
        plt.close('all')


if __name__ == "__main__":
    main()