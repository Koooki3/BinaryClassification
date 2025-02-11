import tkinter as tk
from tkinter import ttk, messagebox
import torch
import numpy as np
from src.models.classifier import RainfallPredictor
from src.data.dataset import DataProcessor

class RainfallPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("降水量预测系统")
        
        # 设置窗口最小尺寸
        self.root.minsize(400, 500)
        
        # 获取屏幕尺寸
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # 设置初始窗口大小为屏幕的40%
        window_width = int(screen_width * 0.4)
        window_height = int(screen_height * 0.5)
        
        # 居中窗口
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # 允许窗口调整大小
        self.root.resizable(True, True)
        
        # 设置主题样式和颜色
        self.setup_styles()
        
        # 配置根窗口的网格权重以实现居中
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # 创建主框架并使其可扩展和居中
        self.main_frame = ttk.Frame(self.root, padding="20", style='Main.TFrame')
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        self.create_widgets()
        self.create_shortcuts()
        self.load_model()

    def setup_styles(self):
        # 定义颜色方案
        colors = {
            'bg': '#F0F2F5',           # 浅蓝灰背景
            'frame_bg': '#FFFFFF',     # 白色框架背景
            'button': '#5B7083',       # 蓝灰按钮
            'button_active': '#405A73', # 深蓝灰激活状态
            'text': '#2F3542',         # 深灰文本
            'label_bg': '#E8EBF0'      # 浅灰标签背景
        }
        
        # 配置ttk样式
        style = ttk.Style()
        style.theme_use('clam')
        
        # 配置框架样式
        style.configure('Main.TFrame', background=colors['bg'])
        
        # 配置标签样式
        style.configure('Title.TLabel',
                       font=('微软雅黑', 16, 'bold'),
                       foreground=colors['text'],
                       background=colors['bg'],
                       padding=(0, 10))
                       
        style.configure('Status.TLabel',
                       font=('微软雅黑', 9),
                       foreground=colors['text'],
                       background=colors['bg'])
        
        # 配置输入框样式
        style.configure('Custom.TEntry',
                       fieldbackground=colors['frame_bg'],
                       borderwidth=1)
        
        # 配置按钮样式
        style.configure('Custom.TButton',
                       font=('微软雅黑', 10),
                       background=colors['button'],
                       foreground='white',
                       padding=(20, 10))
        
        style.map('Custom.TButton',
                 background=[('active', colors['button_active'])])
        
        # 配置标签框样式
        style.configure('Custom.TLabelframe',
                       background=colors['bg'],
                       padding=10)
        
        style.configure('Custom.TLabelframe.Label',
                       font=('微软雅黑', 10),
                       background=colors['bg'],
                       foreground=colors['text'])

    def create_widgets(self):
        # 创建居中的滚动框架
        container = ttk.Frame(self.main_frame)
        container.grid(row=0, column=0, sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        canvas = tk.Canvas(container, bg='#F0F2F5', highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        # 确保内容居中
        self.scrollable_frame.grid_rowconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(0, weight=1)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="n")
        canvas.configure(yscrollcommand=scrollbar.set)

        # 内容容器，用于居中显示所有组件
        content_frame = ttk.Frame(self.scrollable_frame)
        content_frame.grid(row=0, column=0, padx=20, sticky="n")

        # 标题
        title_label = ttk.Label(content_frame, text="广州降水量预测系统",
                              style='Title.TLabel')
        title_label.pack(pady=10)

        # 输入框和标签
        self.entries = {}
        labels_info = {
            "平均温度": {"default": "20", "range": "(5-35°C)"},
            "最低温度": {"default": "15", "range": "(0-30°C)"},
            "最高温度": {"default": "25", "range": "(10-40°C)"},
            "风向": {"default": "180", "range": "(0-360度)"},
            "风速": {"default": "10", "range": "(0-50 km/h)"},
            "气压": {"default": "1013", "range": "(990-1020 hPa)"}
        }

        for label, info in labels_info.items():
            frame = ttk.LabelFrame(content_frame, text=f"{label} {info['range']}", 
                                 style='Custom.TLabelframe')
            frame.pack(fill='x', pady=5)
            
            entry = ttk.Entry(frame, style='Custom.TEntry', justify='center')  # 输入框文字居中
            entry.insert(0, info['default'])
            entry.pack(fill='x', padx=5, pady=5)
            
            self.entries[label] = entry

        # 按钮框架居中
        button_frame = ttk.Frame(content_frame)
        button_frame.pack(pady=20)

        self.predict_button = ttk.Button(button_frame, text="预测 (Enter)",
                                       style='Custom.TButton',
                                       command=self.predict)
        self.predict_button.pack(side=tk.LEFT, padx=10)

        self.reset_button = ttk.Button(button_frame, text="重置 (Esc)",
                                     style='Custom.TButton',
                                     command=self.reset)
        self.reset_button.pack(side=tk.LEFT, padx=10)

        # 状态标签居中
        self.status_label = ttk.Label(content_frame, text="准备就绪",
                                    style='Status.TLabel')
        self.status_label.pack(pady=10)

        # 配置画布和滚动条
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        # 配置容器的网格权重
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

    def _bound_to_mousewheel(self, event, canvas):
        canvas.bind_all("<MouseWheel>", lambda e: self._on_mousewheel(e, canvas))

    def _unbound_to_mousewheel(self, event, canvas):
        canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event, canvas):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def create_shortcuts(self):
        # 添加键盘快捷键
        self.root.bind('<Return>', lambda e: self.predict())
        self.root.bind('<Escape>', lambda e: self.reset())
        
        # 添加Tab键循环焦点
        for entry in self.entries.values():
            entry.bind('<Return>', lambda e: self.focus_next_entry(e.widget))

    def focus_next_entry(self, current):
        # 循环切换输入框焦点
        entries = list(self.entries.values())
        try:
            next_idx = (entries.index(current) + 1) % len(entries)
            entries[next_idx].focus_set()
        except ValueError:
            entries[0].focus_set()

    def validate_input(self, entry):
        # 输入验证
        try:
            value = float(entry.get())
            ranges = {
                "平均温度": (5, 35),
                "最低温度": (0, 30),
                "最高温度": (10, 40),
                "风向": (0, 360),
                "风速": (0, 50),
                "气压": (990, 1020)
            }
            
            for label, (min_val, max_val) in ranges.items():
                if entry == self.entries[label] and not min_val <= value <= max_val:
                    messagebox.showwarning("输入错误", 
                                         f"{label}的值应在{min_val}到{max_val}之间")
                    entry.focus_set()
                    return False
            return True
        except ValueError:
            messagebox.showwarning("输入错误", "请输入有效的数值")
            entry.focus_set()
            return False

    def load_model(self):
        try:
            self.status_label.config(text="正在加载模型...")
            model_path = "models/rainfall_model_latest.pth"
            
            # 使用异步加载来避免界面卡顿
            self.root.after(100, self._load_model_async, model_path)
        except Exception as e:
            self.status_label.config(text=f"模型加载失败: {str(e)}")
            messagebox.showerror("错误", f"模型加载失败: {str(e)}")

    def _load_model_async(self, model_path):
        try:
            checkpoint = torch.load(model_path)
            self.config = checkpoint['config']
            self.model = RainfallPredictor(self.config['model'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.scaler_X = checkpoint['scaler_X']
            self.scaler_y = checkpoint['scaler_y']
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.status_label.config(text="模型加载完成，可以开始预测")
        except Exception as e:
            self.status_label.config(text=f"模型加载失败: {str(e)}")
            messagebox.showerror("错误", f"模型加载失败: {str(e)}")

    def predict(self):
        try:
            features = [float(entry.get()) for entry in self.entries.values()]
            features = np.array([features])
            features = self.scaler_X.transform(features)
            features_tensor = torch.FloatTensor(features).to(self.device)

            with torch.no_grad():
                output = self.model(features_tensor)
                prediction = output.item()
                prediction = self.scaler_y.inverse_transform([[prediction]])[0][0]
                if self.config['data']['scaler']['target_transform'] == 'log1p':
                    prediction = np.expm1(prediction)

            # 根据降水量判断天气状况
            if prediction < 0.1:
                weather = "晴朗"
                suggestion = "适合户外活动"
            elif prediction < 10:
                weather = "小雨"
                suggestion = "外出请携带雨伞"
            elif prediction < 25:
                weather = "中雨"
                suggestion = "尽量避免户外活动"
            elif prediction < 50:
                weather = "大雨"
                suggestion = "注意防范积水和交通安全"
            else:
                weather = "暴雨"
                suggestion = "密切关注天气预警，做好防汛准备"

            self.show_prediction_result(prediction, weather, suggestion)
        except Exception as e:
            messagebox.showerror("错误", f"预测失败: {str(e)}")

    def show_prediction_result(self, prediction, weather, suggestion):
        result_window = tk.Toplevel(self.root)
        result_window.title("预测结果")
        
        # 设置对话框位置和大小
        window_width = 400
        window_height = 250
        x = self.root.winfo_x() + (self.root.winfo_width() - window_width) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - window_height) // 2
        result_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # 配置对话框网格以实现居中
        result_window.grid_rowconfigure(0, weight=1)
        result_window.grid_columnconfigure(0, weight=1)
        
        # 主框架
        frame = ttk.Frame(result_window, padding="20")
        frame.grid(row=0, column=0, sticky="nsew")
        
        # 内容框架
        content_frame = ttk.Frame(frame)
        content_frame.pack(expand=True)
        
        # 显示结果（所有标签居中对齐）
        ttk.Label(content_frame, text=f"预测降水量：{prediction:.2f} mm",
                 style='Custom.TLabelframe.Label',
                 anchor="center").pack(pady=10)
        ttk.Label(content_frame, text=f"天气状况：{weather}",
                 style='Custom.TLabelframe.Label',
                 anchor="center").pack(pady=10)
        ttk.Label(content_frame, text=f"建议：{suggestion}",
                 style='Custom.TLabelframe.Label',
                 anchor="center").pack(pady=10)
        
        # 关闭按钮居中
        ttk.Button(content_frame, text="确定", 
                  style='Custom.TButton',
                  command=result_window.destroy).pack(pady=20)
        
        # 使对话框模态
        result_window.transient(self.root)
        result_window.grab_set()
        self.root.wait_window(result_window)

    def reset(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.status_label.config(text="准备就绪")

if __name__ == "__main__":
    root = tk.Tk()
    app = RainfallPredictionApp(root)
    root.mainloop()
