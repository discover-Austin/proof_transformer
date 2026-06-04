from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

from .api import DesktopCommanderAPIServer
from .executor import DesktopExecutor
from .history import HistoryStore
from .paths import get_app_home
from .planner import TaskPlanner


class DesktopCommanderApp:
    def __init__(self, root: tk.Tk, *, executor: DesktopExecutor, planner: TaskPlanner, history: HistoryStore) -> None:
        self.root = root
        self.executor = executor
        self.planner = planner
        self.history = history
        self.api_server: DesktopCommanderAPIServer | None = None
        self.api_status = tk.StringVar(value="API stopped")
        self.status_var = tk.StringVar(value=f"Ready. Data home: {get_app_home()}")
        self.task_var = tk.StringVar(value="open powershell and search the web for desktop automation tips")
        self.result_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.running = False

        self._configure_root()
        self._build_layout()
        self._load_capabilities()
        self._refresh_history()
        self._refresh_presets()
        self._restore_draft_or_default()
        self._poll_queue()

    def _configure_root(self) -> None:
        self.root.title("Desktop Commander")
        self.root.geometry("1380x860")
        self.root.minsize(1100, 720)
        self.root.configure(bg="#f6efe3")
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10))
        style.configure("Banner.TLabel", background="#f6efe3", foreground="#153243", font=("Segoe UI Semibold", 18))
        style.configure("Subtle.TLabel", background="#f6efe3", foreground="#5a6872")
        style.configure("Accent.TButton", padding=(12, 7), background="#1f7a8c", foreground="#ffffff")
        style.map("Accent.TButton", background=[("active", "#165f6d")])
        style.configure("Panel.TFrame", background="#fffaf2")

    def _build_layout(self) -> None:
        shell = ttk.Frame(self.root, padding=16, style="Panel.TFrame")
        shell.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(shell, style="Panel.TFrame")
        header.pack(fill=tk.X)
        ttk.Label(header, text="Desktop Commander", style="Banner.TLabel").pack(anchor=tk.W)
        ttk.Label(
            header,
            text="Standalone desktop task runner with planner, history, presets, and a local API.",
            style="Subtle.TLabel",
        ).pack(anchor=tk.W, pady=(2, 10))

        task_row = ttk.Frame(shell, style="Panel.TFrame")
        task_row.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(task_row, text="Task").pack(side=tk.LEFT, padx=(0, 8))
        entry = ttk.Entry(task_row, textvariable=self.task_var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        entry.bind("<Return>", lambda _event: self.draft_plan())
        ttk.Button(task_row, text="Draft Plan", command=self.draft_plan, style="Accent.TButton").pack(side=tk.LEFT, padx=(8, 6))
        self.run_button = ttk.Button(task_row, text="Run Plan", command=self.run_plan, style="Accent.TButton")
        self.run_button.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(task_row, text="Save Preset", command=self.save_preset).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(task_row, text="Load Example", command=self.load_example).pack(side=tk.LEFT)

        quick_row = ttk.Frame(shell, style="Panel.TFrame")
        quick_row.pack(fill=tk.X, pady=(0, 10))
        quick_specs = [
            ("Search Web", "search the web for Windows automation tools"),
            ("Open PowerShell", "open powershell"),
            ("Fetch URL", "fetch https://example.com"),
            ("Open Folder", "open folder ."),
            ("Read README", "read README.md"),
            ("Open Data", "open folder runtime_data"),
        ]
        for label, task in quick_specs:
            ttk.Button(quick_row, text=label, command=lambda value=task: self.apply_quick_task(value)).pack(side=tk.LEFT, padx=(0, 6))

        paned = ttk.Panedwindow(shell, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned, style="Panel.TFrame", padding=(0, 0, 12, 0))
        right = ttk.Frame(paned, style="Panel.TFrame")
        paned.add(left, weight=5)
        paned.add(right, weight=4)

        ttk.Label(left, text="Plan JSON").pack(anchor=tk.W)
        self.plan_text = tk.Text(left, wrap=tk.WORD, height=20, bg="#fffdf8", fg="#132a36", insertbackground="#132a36")
        self.plan_text.pack(fill=tk.BOTH, expand=True, pady=(4, 8))

        result_header = ttk.Frame(left, style="Panel.TFrame")
        result_header.pack(fill=tk.X)
        ttk.Label(result_header, text="Latest Result").pack(side=tk.LEFT)
        ttk.Button(result_header, text="Open Data Folder", command=self.open_data_folder).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(result_header, text="Clear Result", command=self.clear_result).pack(side=tk.RIGHT)
        self.result_text = tk.Text(left, wrap=tk.WORD, height=14, bg="#14213d", fg="#f1faee", insertbackground="#f1faee")
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        notebook = ttk.Notebook(right)
        notebook.pack(fill=tk.BOTH, expand=True)

        history_tab = ttk.Frame(notebook, padding=10)
        presets_tab = ttk.Frame(notebook, padding=10)
        status_tab = ttk.Frame(notebook, padding=10)
        notebook.add(history_tab, text="History")
        notebook.add(presets_tab, text="Presets")
        notebook.add(status_tab, text="Status")

        self.history_tree = ttk.Treeview(history_tab, columns=("time", "status", "task"), show="headings", height=16)
        self.history_tree.heading("time", text="Time")
        self.history_tree.heading("status", text="Status")
        self.history_tree.heading("task", text="Task")
        self.history_tree.column("time", width=170, anchor=tk.W)
        self.history_tree.column("status", width=90, anchor=tk.W)
        self.history_tree.column("task", width=430, anchor=tk.W)
        self.history_tree.pack(fill=tk.BOTH, expand=True)

        history_buttons = ttk.Frame(history_tab)
        history_buttons.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(history_buttons, text="Refresh", command=self._refresh_history).pack(side=tk.LEFT)
        ttk.Button(history_buttons, text="Load Selected", command=self.load_selected_history).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(history_buttons, text="Rerun Selected", command=self.rerun_selected_history).pack(side=tk.LEFT, padx=(6, 0))

        self.presets_list = tk.Listbox(presets_tab, height=18, bg="#fffdf8", fg="#132a36")
        self.presets_list.pack(fill=tk.BOTH, expand=True)
        preset_buttons = ttk.Frame(presets_tab)
        preset_buttons.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(preset_buttons, text="Refresh", command=self._refresh_presets).pack(side=tk.LEFT)
        ttk.Button(preset_buttons, text="Load Preset", command=self.load_selected_preset).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(status_tab, text="Capabilities").pack(anchor=tk.W)
        self.capabilities_text = tk.Text(status_tab, wrap=tk.WORD, height=16, bg="#fffdf8", fg="#132a36")
        self.capabilities_text.pack(fill=tk.BOTH, expand=True, pady=(4, 8))

        api_row = ttk.Frame(status_tab)
        api_row.pack(fill=tk.X)
        ttk.Label(api_row, textvariable=self.api_status).pack(side=tk.LEFT)
        ttk.Button(api_row, text="Start API", command=self.start_api_server).pack(side=tk.RIGHT)
        ttk.Button(api_row, text="Stop API", command=self.stop_api_server).pack(side=tk.RIGHT, padx=(0, 6))

        footer = ttk.Frame(shell, style="Panel.TFrame")
        footer.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(footer, textvariable=self.status_var, style="Subtle.TLabel").pack(side=tk.LEFT)

    def _load_capabilities(self) -> None:
        self.capabilities_text.delete("1.0", tk.END)
        self.capabilities_text.insert(tk.END, json.dumps(self.executor.capabilities(), indent=2))

    def apply_quick_task(self, task: str) -> None:
        self.task_var.set(task)
        self.draft_plan()

    def load_example(self) -> None:
        self.task_var.set("open powershell and type Get-Date and search the web for Python tkinter examples")
        self.draft_plan()

    def draft_plan(self) -> None:
        try:
            payload = self.planner.draft_plan(self.task_var.get())
        except Exception as exc:
            messagebox.showerror("Draft failed", str(exc))
            return
        self.plan_text.delete("1.0", tk.END)
        self.plan_text.insert(tk.END, json.dumps(payload, indent=2))
        self.history.save_draft(task_text=self.task_var.get(), plan=payload)
        self.status_var.set(f"Drafted plan: {payload['summary']}")

    def run_plan(self) -> None:
        if self.running:
            return
        try:
            payload = json.loads(self.plan_text.get("1.0", tk.END).strip())
            if not isinstance(payload, dict) or not isinstance(payload.get("steps"), list):
                raise ValueError("Plan editor must contain a JSON object with a steps list.")
        except Exception as exc:
            messagebox.showerror("Invalid plan", str(exc))
            return
        self.history.save_draft(task_text=self.task_var.get(), plan=payload)
        self.running = True
        self.run_button.state(["disabled"])
        self.status_var.set("Running plan...")
        threading.Thread(target=self._execute_in_background, args=(payload,), daemon=True).start()

    def _execute_in_background(self, payload: dict) -> None:
        try:
            result = self.executor.execute_plan(
                payload,
                continue_on_error=bool(payload.get("continue_on_error", False)),
            )
            record = self.history.append_run(task_text=payload.get("task_text"), plan=payload, result=result)
            self.result_queue.put(("completed", {"record": record, "result": result}))
        except Exception as exc:
            self.result_queue.put(("error", str(exc)))

    def _poll_queue(self) -> None:
        try:
            while True:
                event, payload = self.result_queue.get_nowait()
                if event == "completed":
                    result = payload["result"]
                    self.result_text.delete("1.0", tk.END)
                    self.result_text.insert(tk.END, json.dumps(result, indent=2))
                    self.status_var.set(f"Plan finished with status={result['status']}")
                    self._refresh_history()
                elif event == "error":
                    messagebox.showerror("Execution failed", str(payload))
                    self.status_var.set("Plan failed before execution completed.")
                self.running = False
                self.run_button.state(["!disabled"])
        except queue.Empty:
            pass
        self.root.after(150, self._poll_queue)

    def _refresh_history(self) -> None:
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        self.history_records = self.history.list_runs(limit=50)
        for record in self.history_records:
            status = record["result"].get("status", "unknown")
            task = record.get("task_text") or record["plan"].get("summary") or "<no task text>"
            self.history_tree.insert("", tk.END, iid=record["id"], values=(record["timestamp"], status, task))

    def _refresh_presets(self) -> None:
        self.preset_records = self.history.list_presets()
        self.presets_list.delete(0, tk.END)
        for preset in self.preset_records:
            self.presets_list.insert(tk.END, preset["name"])

    def load_selected_history(self) -> None:
        selected = self.history_tree.selection()
        if not selected:
            return
        record = next((item for item in self.history_records if item["id"] == selected[0]), None)
        if record is None:
            return
        self.plan_text.delete("1.0", tk.END)
        self.plan_text.insert(tk.END, json.dumps(record["plan"], indent=2))
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, json.dumps(record["result"], indent=2))
        if record.get("task_text"):
            self.task_var.set(record["task_text"])
        self.status_var.set(f"Loaded history item {record['id']}")

    def rerun_selected_history(self) -> None:
        self.load_selected_history()
        self.run_plan()

    def save_preset(self) -> None:
        try:
            payload = json.loads(self.plan_text.get("1.0", tk.END).strip())
            if not isinstance(payload, dict) or not isinstance(payload.get("steps"), list):
                raise ValueError("Plan editor must contain a JSON object with a steps list.")
        except Exception as exc:
            messagebox.showerror("Invalid plan", str(exc))
            return
        name = simpledialog.askstring("Save preset", "Preset name:", initialvalue=self.task_var.get()[:50] or "New preset")
        if not name:
            return
        try:
            self.history.save_preset(name=name, task_text=self.task_var.get(), plan=payload)
        except Exception as exc:
            messagebox.showerror("Preset error", str(exc))
            return
        self._refresh_presets()
        self.status_var.set(f"Saved preset '{name}'.")

    def load_selected_preset(self) -> None:
        selected = self.presets_list.curselection()
        if not selected:
            return
        preset = self.preset_records[selected[0]]
        self.task_var.set(preset.get("task_text") or preset["name"])
        self.plan_text.delete("1.0", tk.END)
        self.plan_text.insert(tk.END, json.dumps(preset["plan"], indent=2))
        self.history.save_draft(task_text=self.task_var.get(), plan=preset["plan"])
        self.status_var.set(f"Loaded preset '{preset['name']}'.")

    def clear_result(self) -> None:
        self.result_text.delete("1.0", tk.END)
        self.status_var.set("Cleared latest result view.")

    def open_data_folder(self) -> None:
        try:
            self.executor.execute_step({"action": "open_path", "path": str(get_app_home())})
            self.status_var.set(f"Opened data folder: {get_app_home()}")
        except Exception as exc:
            messagebox.showerror("Open folder failed", str(exc))

    def start_api_server(self) -> None:
        if self.api_server is not None:
            self.api_status.set(f"API already running on {self.api_server.address[0]}:{self.api_server.address[1]}")
            return
        self.api_server = DesktopCommanderAPIServer(executor=self.executor, planner=self.planner, history=self.history)
        self.api_server.start_in_thread()
        host, port = self.api_server.address
        self.api_status.set(f"API running on http://{host}:{port}")

    def stop_api_server(self) -> None:
        if self.api_server is None:
            self.api_status.set("API stopped")
            return
        self.api_server.shutdown()
        self.api_server = None
        self.api_status.set("API stopped")

    def _restore_draft_or_default(self) -> None:
        draft = self.history.load_draft()
        if draft and isinstance(draft.get("plan"), dict):
            self.task_var.set(draft.get("task_text") or self.task_var.get())
            self.plan_text.delete("1.0", tk.END)
            self.plan_text.insert(tk.END, json.dumps(draft["plan"], indent=2))
            self.status_var.set("Restored previous draft.")
            return
        self.draft_plan()


def launch_gui(*, executor: DesktopExecutor | None = None, planner: TaskPlanner | None = None, history: HistoryStore | None = None) -> None:
    root = tk.Tk()
    app = DesktopCommanderApp(
        root,
        executor=executor or DesktopExecutor(),
        planner=planner or TaskPlanner(),
        history=history or HistoryStore(),
    )
    root.protocol("WM_DELETE_WINDOW", lambda: _close_app(root, app))
    root.mainloop()


def _close_app(root: tk.Tk, app: DesktopCommanderApp) -> None:
    app.stop_api_server()
    root.destroy()
