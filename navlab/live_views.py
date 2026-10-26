"""MuJoCo + Tk/Pillow live dashboard, following newmujoco's read-only UI pattern."""
import math
import time
from functools import lru_cache
from pathlib import Path
from . import runtime
import mujoco
import numpy as np

STAGES = {"RESET":"初始状态","FLIGHT":"飞行 / 穿越","PROBE":"Probe 主动接触",
          "PUSH":"正式推门 / 补推","PUSH_BRAKE":"提前制动 / 门惯性续开",
          "BACKOFF":"制动 / 脱离 / 姿态恢复","DONE":"场景完成"}
PHASES = {"INITIAL":"初始等权","PROBE":"Probe 中：同一时刻的观测一致性更新",
          "PROBE_COMPLETE":"Probe 结束：最终观测一致性权重",
          "STATE_CONDITIONED":"Probe 后：使用剩余路线标签；执行期间保留权重"}


@lru_cache(maxsize=12)
def font(size=13):
    from PIL import ImageFont
    try:
        return ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",size)
    except OSError:
        return ImageFont.load_default()


def hypothesis_image(theta,weights,labels,truth,cfg,size=(620,350)):
    from PIL import Image,ImageDraw
    w,h = size
    image = Image.new("RGB",size,"#ffffff")
    draw = ImageDraw.Draw(image)
    left,right,top,bottom = 56,w-28,30,h-42
    draw.line([(left,top),(left,bottom),(right,bottom)],fill="#94a3b8",width=1)
    for fraction in np.linspace(0,1,5):
        x,y = left+(right-left)*fraction,bottom-(bottom-top)*fraction
        draw.line([(x,top),(x,bottom)],fill="#edf2f7")
        draw.line([(left,y),(right,y)],fill="#edf2f7")
        iv = cfg.inertia_min+fraction*(cfg.inertia_max-cfg.inertia_min)
        fv = cfg.friction_min+fraction*(cfg.friction_max-cfg.friction_min)
        draw.text((x,bottom+7),f"{iv:.2f}",fill="#475569",font=font(11),anchor="mt")
        draw.text((left-8,y),f"{fv:.2f}",fill="#475569",font=font(11),anchor="rm")
    draw.text((left,6),"τ_f (N m)",fill="#334155",font=font(12))
    draw.text(((left+right)/2,h-18),"I：绕铰链的转动惯量 (kg m²)",fill="#334155",font=font(12),anchor="mm")
    xy = np.empty_like(theta)
    xy[:,0] = left+(theta[:,0]-cfg.inertia_min)/(cfg.inertia_max-cfg.inertia_min)*(right-left)
    xy[:,1] = bottom-(theta[:,1]-cfg.friction_min)/(cfg.friction_max-cfg.friction_min)*(bottom-top)
    maximum = max(float(np.max(weights)),1e-15)
    for i in np.argsort(weights):
        ratio = float(weights[i])/maximum
        radius = 1.4+4.*math.sqrt(ratio)
        raw = np.array([5,150,105]) if labels[i] else np.array([220,38,38])
        color = tuple((255+(.08+.9*ratio)*(raw-255)).astype(int))
        x,y = xy[i]
        draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=color)
    tx = left+(truth[0]-cfg.inertia_min)/(cfg.inertia_max-cfg.inertia_min)*(right-left)
    ty = bottom-(truth[1]-cfg.friction_min)/(cfg.friction_max-cfg.friction_min)*(bottom-top)
    draw.line([(tx-7,ty),(tx+7,ty)],fill="#0f172a",width=2)
    draw.line([(tx,ty-7),(tx,ty+7)],fill="#0f172a",width=2)
    return image


def curve_image(timeline,size=(620,180),decision=False):
    from PIL import Image,ImageDraw
    image = Image.new("RGB",size,"#ffffff")
    draw = ImageDraw.Draw(image)
    w,h = size
    left,right,top,bottom = 48,w-25,30,h-28
    draw.line([(left,top),(left,bottom),(right,bottom)],fill="#94a3b8")
    if not timeline:
        return image
    rows = np.asarray([[r[0],r[4],r[5],r[9],r[10]] for r in timeline],dtype=float)
    t0,t1 = rows[0,0],max(rows[-1,0],rows[0,0]+.1)
    x = left+(rows[:,0]-t0)/(t1-t0)*(right-left)
    if decision:
        series = [(rows[:,3],1.,"#059669","p_D"),(rows[:,4],1.,"#7c3aed","D")]
        draw.text((left,7),"动作支持度 / 决策分歧（0–1）",fill="#334155",font=font(12))
    else:
        force = max(float(rows[:,1].max()),5.)
        series = [(rows[:,1],force,"#dc2626",f"接触力 / {force:.1f} N"),
                  (np.degrees(rows[:,2]),90.,"#2563eb","门角 / 90°")]
        draw.text((left,7),"实时接触力与门角（各按标注上限缩放）",fill="#334155",font=font(12))
    legend_y = top
    for values,maximum,color,label in series:
        y = bottom-np.clip(values/maximum,0.,1.)*(bottom-top)
        points = list(zip(x,y))
        if len(points)>1:
            draw.line(points,fill=color,width=2)
        draw.text((right,legend_y),label,fill=color,font=font(11),anchor="ra")
        legend_y += 17
    draw.text((left,bottom+7),f"{t0:.1f}",fill="#475569",font=font(11))
    draw.text((right,bottom+7),f"{t1:.1f} s",fill="#475569",font=font(11),anchor="ra")
    return image


class LiveViews:
    def __init__(self,speed=1.,native_viewer=True,hidden=False):
        import tkinter as tk
        from tkinter import ttk
        from PIL import ImageTk
        self.tk,self.ttk,self.ImageTk = tk,ttk,ImageTk
        self.root = tk.Tk()
        self.root.title("mujoco_1024kinds：实时推门画面与 1024 候选权重")
        self.root.geometry(f"{min(1320,self.root.winfo_screenwidth()-60)}x{min(900,self.root.winfo_screenheight()-90)}+20+20")
        self.root.minsize(940,650)
        if hidden:
            self.root.withdraw()
        self.running,self.paused,self.next_requested = True,False,False
        self.pause_requested = False
        self.native_viewer,self.hidden = native_viewer,hidden
        self.world=self.belief=self.viewer=self.renderer=None
        self.viewer_model=self.viewer_data=None
        self.folder=self.case=None
        self.last_sim,self.last_wall,self.last_draw = None,time.perf_counter(),-1.
        self.photos,self.images = {},{}
        self.frames_rendered = 0
        self.captured_contacts = set()
        self.completed = []
        self.status = tk.StringVar(value="等待场景初始化")
        self.metrics = tk.StringVar(value="G / p_D / D")
        self.sensor = tk.StringVar(value="实时接触力 / 门角")
        self.phase = tk.StringVar(value="初始等权")
        self.speed = tk.StringVar(value=str(speed))
        toolbar = ttk.Frame(self.root,padding=8); toolbar.pack(fill="x")
        ttk.Label(toolbar,textvariable=self.status).pack(side="left",fill="x",expand=True)
        self.next_button = ttk.Button(toolbar,text="下一场景",command=self.request_next,state="disabled")
        self.next_button.pack(side="right",padx=4)
        self.pause_button = ttk.Button(toolbar,text="暂停",command=self.toggle_pause)
        self.pause_button.pack(side="right",padx=4)
        ttk.Button(toolbar,text="保存当前画面",command=self.save_snapshot).pack(side="right",padx=4)
        speed_box = ttk.Combobox(toolbar,textvariable=self.speed,values=("0.25","0.5","1","2","4"),width=5,state="readonly")
        speed_box.pack(side="right",padx=4)
        ttk.Label(toolbar,text="速度 ×").pack(side="right")
        ttk.Label(self.root,textvariable=self.metrics,font=("Microsoft YaHei",12,"bold")).pack(anchor="w",padx=12)
        panels = ttk.Panedwindow(self.root,orient="horizontal"); panels.pack(fill="both",expand=True,padx=8,pady=8)
        left,right = ttk.Frame(panels),ttk.Frame(panels)
        panels.add(left,weight=1); panels.add(right,weight=1)
        ttk.Label(left,text="MuJoCo 实际渲染：接近、Probe、正式推门、脱离和绕行").pack(anchor="w")
        self.camera = tk.Label(left,bg="#18202a"); self.camera.pack(fill="both",expand=True)
        ttk.Label(left,textvariable=self.sensor).pack(anchor="w",pady=4)
        self.force_curve = tk.Label(left); self.force_curve.pack(fill="x")
        ttk.Label(right,textvariable=self.phase).pack(anchor="w")
        self.scatter = tk.Label(right,bg="white"); self.scatter.pack(fill="both",expand=True)
        ttk.Label(right,text="绿色支持 Door；红色支持 Bypass；黑十字仅表示评估真值。\n点大小/深浅按本帧最大权重归一化；具体权重见下表。").pack(anchor="w")
        tabs = ttk.Notebook(right); tabs.pack(fill="x")
        curve_tab,table_tab = ttk.Frame(tabs),ttk.Frame(tabs)
        tabs.add(curve_tab,text="p_D / D 曲线"); tabs.add(table_tab,text="高权重候选")
        self.decision_curve = tk.Label(curve_tab); self.decision_curve.pack(fill="x")
        columns = ("id","I","tau","weight","action")
        self.top = ttk.Treeview(table_tab,columns=columns,show="headings",height=4)
        for key,label,width in zip(columns,("候选","I","τ_f","权重","支持动作"),(60,85,85,105,80)):
            self.top.heading(key,text=label); self.top.column(key,width=width,anchor="center")
        self.top.pack(fill="x")
        self.summary = tk.StringVar(value="空格暂停 / 继续；场景结束后点击下一场景。关闭面板继续后台实验。")
        ttk.Label(self.root,textvariable=self.summary).pack(anchor="w",padx=12,pady=(0,8))
        self.root.protocol("WM_DELETE_WINDOW",self.close)
        self.root.bind("<space>",lambda _:self.toggle_pause())

    def begin_case(self,name,world,belief,truth,folder):
        self.release_scene()
        if not self.running:
            return
        self.case,self.world,self.belief,self.truth,self.folder = name,world,belief,np.asarray(truth),Path(folder)
        self.next_requested,self.paused = False,False
        self.captured_contacts = set()
        self.next_button.configure(state="disabled",text="下一场景")
        self.pause_button.configure(text="暂停")
        self.renderer = mujoco.Renderer(world.model,height=360,width=600)
        self.camera_spec = mujoco.MjvCamera()
        self.camera_spec.azimuth = 90.
        self.scene_option = mujoco.MjvOption(); self.scene_option.sitegroup[:] = 0
        if self.native_viewer:
            from mujoco import viewer
            import copy
            # viewer.sync() may run mj_forward / apply UI state. Mirror into its own
            # model/data so displaying a frame cannot change the research rollout.
            self.viewer_model = copy.copy(world.model)
            self.viewer_data = mujoco.MjData(self.viewer_model)
            self.copy_native_state()
            self.viewer = viewer.launch_passive(self.viewer_model,self.viewer_data,key_callback=self.native_key)
            with self.viewer.lock():
                self.viewer.cam.lookat[:] = [0.,1.7,1.]
                self.viewer.cam.distance,self.viewer.cam.azimuth,self.viewer.cam.elevation = 12.,90.,-75.
        self.last_sim,self.last_wall,self.last_draw = world.data.time,time.perf_counter(),-1.
        self.update(world,force=True)
        if self.running and not self.hidden:
            self.root.lift()

    def copy_native_state(self):
        source,target = self.world.data,self.viewer_data
        target.time = float(source.time)
        for name in ("qpos","qvel","act","ctrl","qfrc_applied","xfrc_applied","qacc_warmstart"):
            getattr(target,name)[:] = getattr(source,name)
        mujoco.mj_forward(self.viewer_model,target)

    def sync_native(self):
        if self.viewer and self.viewer.is_running():
            self.copy_native_state()
            self.viewer.sync()

    def put_image(self,key,widget,image):
        from PIL import Image
        self.images[key] = image
        width,height = widget.winfo_width(),widget.winfo_height()
        displayed = image
        if width>20 and height>20:
            scale = min(1.,width/image.width,height/image.height)
            if scale<.99:
                displayed = image.resize((max(1,int(image.width*scale)),max(1,int(image.height*scale))),
                                         Image.Resampling.LANCZOS)
        photo = self.ImageTk.PhotoImage(displayed,master=self.root)
        self.photos[key] = photo
        widget.configure(image=photo)

    def render_scene(self):
        from PIL import Image
        world = self.world
        pos = world.observed()[0]
        if np.linalg.norm(pos[:2]-np.array([.05,1.45])) < 1.65:
            self.camera_spec.lookat[:] = [.05,1.45,1.15]
            self.camera_spec.distance,self.camera_spec.elevation = 3.6,-58.
        else:
            self.camera_spec.lookat[:] = pos
            self.camera_spec.distance,self.camera_spec.elevation = 3.3,-70.
        self.renderer.update_scene(world.data,camera=self.camera_spec,scene_option=self.scene_option)
        image = Image.fromarray(self.renderer.render().copy())
        self.frames_rendered += 1
        self.put_image("camera",self.camera,image)

    def update(self,world,force=False):
        if not self.running:
            return
        belief = self.belief
        self.status.set(f"Case {self.case}  |  t={world.data.time:.2f}s  |  {STAGES.get(world.stage,world.stage)}  |  动作：{belief.action}")
        self.metrics.set(f"G={belief.G:.3f} m    p_D={belief.p:.6f}    D={belief.D:.6f}    最大权重={belief.weights.max():.4f}")
        self.sensor.set(f"接触力={world.force:.3f} N   门角={math.degrees(world.data.qpos[world.qa]):.2f}°   Probe 累计峰值={belief.observation[0]:.3f} N")
        self.phase.set(PHASES.get(belief.phase,belief.phase))
        if force or world.data.time-self.last_draw >= .099:
            self.last_draw = world.data.time
            self.render_scene()
            self.put_image("scatter",self.scatter,hypothesis_image(belief.library["theta"],belief.weights,
                belief.labels,self.truth,world.cfg))
            self.put_image("force",self.force_curve,curve_image(belief.timeline))
            self.put_image("decision",self.decision_curve,curve_image(belief.timeline,decision=True))
            for item in self.top.get_children():
                self.top.delete(item)
            for i in np.argsort(-belief.weights)[:4]:
                self.top.insert("","end",values=(int(i),f"{belief.library['theta'][i,0]:.5f}",
                    f"{belief.library['theta'][i,1]:.5f}",f"{belief.weights[i]:.6f}","Door" if belief.labels[i] else "Bypass"))
        self.pump()

    def on_world(self,world):
        if not self.running:
            return
        self.update(world)
        if world.stage in {"PROBE","PUSH"} and world.force>.05 and world.stage not in self.captured_contacts:
            self.captured_contacts.add(world.stage)
            self.update(world,force=True)
            self.save_snapshot(stem=world.stage.lower()+"_first_contact")
        self.sync_native()
        while self.paused and self.running:
            self.pump()
            self.sync_native()
            time.sleep(.015)
            self.last_wall = time.perf_counter()
        now = time.perf_counter()
        if self.running:
            speed = max(.01,float(self.speed.get()))
            dt = max(0.,min(.1,float(world.data.time)-float(self.last_sim)))
            delay = dt/speed-(now-self.last_wall)
            if delay>0:
                time.sleep(delay)
        self.last_sim,self.last_wall = float(world.data.time),time.perf_counter()

    def pump(self):
        if not self.running:
            return
        try:
            if self.pause_requested:
                self.pause_requested = False
                self.toggle_pause()
            self.root.update_idletasks(); self.root.update()
        except self.tk.TclError:
            self.running = False

    def native_key(self,key):
        # MuJoCo's key callback runs on its UI thread; only the main thread touches Tk.
        if key==32:
            self.pause_requested = not self.pause_requested

    def toggle_pause(self):
        if self.running:
            self.paused = not self.paused
            self.pause_button.configure(text="继续" if self.paused else "暂停")

    def request_next(self):
        self.next_requested = True
        self.paused = False

    def waiting(self,text):
        if self.running:
            self.status.set(f"Case {self.case} | {text}")
            self.pump()
            self.sync_native()

    def finish_case(self,result):
        if not self.running:
            return
        self.paused = False
        self.update(self.world,force=True)
        self.status.set(f"Case {self.case} 完成：{result['final_action']}；到 M：{result['mission_success']}")
        self.completed.append(f"{self.case}: {result['prior']['action']}→{result['final_action']}")
        self.summary.set("已完成："+"； ".join(self.completed))
        self.save_snapshot(stem="final_dashboard")

    def wait_for_next(self,is_last=False,auto_advance=False,close_on_done=False):
        if not self.running:
            return
        if is_last and close_on_done:
            return
        self.next_requested = False
        self.next_button.configure(state="normal",text="完成 / 关闭" if is_last else "下一场景")
        self.status.set(self.status.get()+"  |  "+("可查看最终结果，点击完成关闭" if is_last else "点击下一场景继续"))
        deadline = time.perf_counter()+1. if auto_advance and not is_last else None
        while self.running and not self.next_requested:
            self.pump()
            self.sync_native()
            if deadline is not None and time.perf_counter()>=deadline:
                break
            time.sleep(.02)

    def save_snapshot(self,stem=None):
        from PIL import Image,ImageDraw
        if not self.running or not self.images or self.folder is None:
            return
        folder = self.folder/"live_views"; folder.mkdir(parents=True,exist_ok=True)
        stem = stem or f"live_{self.world.data.time:.2f}"
        image = Image.new("RGB",(1280,790),"#f8fafc")
        draw = ImageDraw.Draw(image)
        draw.text((15,12),self.status.get(),font=font(17),fill="#0f172a")
        draw.text((15,45),self.metrics.get(),font=font(15),fill="#0f172a")
        draw.text((15,75),self.phase.get(),font=font(13),fill="#334155")
        image.paste(self.images["camera"],(15,112))
        image.paste(self.images["scatter"],(650,105))
        image.paste(self.images["force"],(15,510))
        image.paste(self.images["decision"],(650,510))
        draw.text((15,720),self.sensor.get(),font=font(13),fill="#334155")
        draw.text((15,750),"实际 MuJoCo 渲染 + 实时权重；绿色 Door，红色 Bypass；黑十字仅用于评估。",font=font(13),fill="#334155")
        image.save(folder/(stem+".png"))
        return folder/(stem+".png")

    def release_scene(self):
        if self.viewer:
            self.viewer.close(); self.viewer = None
        self.viewer_model=self.viewer_data=None
        if self.renderer:
            self.renderer.close(); self.renderer = None

    def close(self):
        if self.running:
            self.running = False
            self.release_scene()
            self.root.destroy()
