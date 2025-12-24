import customtkinter as ctk
import ollama
import threading
import json
import os
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import time
import io
import wave
import pygame
from piper.voice import PiperVoice
from PIL import Image
from tkinter import filedialog, font, messagebox

# --- CONFIGURATION ---
PIPER_DIR = "piper"
CONFIG_FILE = "app_config.json"
SAVE_FILE = "characters.json"

# Base de données pour la recherche globale (modèles du cloud Ollama)
OLLAMA_LIBRARY = [
    "llama3", "llama3:70b", "llama3:instruct", "mistral", "mixtral", "mixtral:8x7b",
    "phi3", "phi3:mini", "gemma", "gemma:7b", "gemma:2b", "codellama", "codellama:7b-python",
    "dolphin-llama3", "dolphin-mistral", "command-r", "command-r-plus", "llava", "llava:13b",
    "neural-chat", "qwen", "qwen:7b", "qwen:14b", "tinyllama", "orca-mini", "deepseek-coder",
    "deepseek-coder-v2", "starcoder2", "stable-code", "medllama2", "llama2-uncensored"
]

THEMES_CONFIG = {
    "Cherry":    {"main": "#132440", "accent": "#FDB5CE", "sidebar": "#16476A"},
    "Basic":     {"main": "#2b2b2b", "accent": "#3b8ed0", "sidebar": "#212121"},
    "Dark Blue": {"main": "#1a1a1a", "accent": "#1f538d", "sidebar": "#161617"},
    "Cyberpunk": {"main": "#1a001a", "accent": "#ff00ff", "sidebar": "#120012"},
    "Forest":    {"main": "#1b2e1c", "accent": "#2e7d32", "sidebar": "#141f14"}
}

class UltimateOllamaApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        pygame.mixer.init()
        ctk.set_appearance_mode("dark")
        self.title("Ollama OS - Master Full Edition")
        self.geometry("1200x850")

        # Chargement des données
        self.app_config = self.load_app_config()
        self.characters = self.load_chars()
        
        # Variables d'état
        self.recognizer = sr.Recognizer()
        self.selected_mic_index = None
        self.is_listening = False
        self.is_calling = False
        self.is_testing_mic = False
        self.voice = None
        self.checks = {}
        
        self.load_selected_voice(self.app_config.get("voice_model"))
        self.update_font_objects()

        # Fond d'écran
        self.bg_canvas = ctk.CTkLabel(self, text="")
        self.bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)

        # Barre latérale (Sidebar)
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0)
        self.sidebar_title = ctk.CTkLabel(self.sidebar, text="OLLAMA OS", font=(self.app_config.get("font_family", "Arial"), 30, "bold"))
        self.sidebar_title.pack(pady=40)
        
        self.menu_buttons = []
        for text, page_id in [("💬 Chat", "chat"), ("🎭 Roleplay", "rp"), ("📥 Modèles", "catalog"), ("⚙️ Paramètres", "settings")]:
            btn = ctk.CTkButton(self.sidebar, text=text, command=lambda p=page_id: self.show_page(p), height=50)
            btn.pack(pady=10, padx=20)
            self.menu_buttons.append(btn)

        # Zone de survol pour afficher la sidebar
        self.hover_zone = ctk.CTkFrame(self, width=20, fg_color="transparent")
        self.hover_zone.place(x=0, y=0, relheight=1)
        self.hover_zone.bind("<Enter>", lambda e: self.toggle_sidebar(True))

        # Conteneur principal
        self.main_container = ctk.CTkFrame(self, corner_radius=15)
        self.main_container.pack(side="right", expand=True, fill="both", padx=(40, 20), pady=20)
        
        self.pages = {}
        self.create_chat_page()
        self.create_rp_page()
        self.create_catalog_page()
        self.create_settings_page()
        
        self.apply_theme(self.app_config.get("theme", "Cherry"))
        if self.app_config.get("bg_path"): self.load_background(self.app_config["bg_path"])
        
        self.bind("<Configure>", self.resizer)
        
        # On affiche la page par défaut
        self.show_page("chat")
        self.after(500, self.refresh_mics)

    # --- NAVIGATION ---
    def show_page(self, n):
        for p in self.pages.values(): 
            p.pack_forget()
        self.pages[n].pack(fill="both", expand=True)

    def toggle_sidebar(self, show):
        if show: 
            self.sidebar.place(x=0, y=0, relheight=1)
            self.sidebar.lift()
            self.check_sidebar_leave()
        else: 
            self.sidebar.place_forget()

    def check_sidebar_leave(self):
        try:
            if (self.winfo_pointerx() - self.winfo_rootx()) > 240: 
                self.toggle_sidebar(False)
            else: 
                self.after(100, self.check_sidebar_leave)
        except: 
            pass

    # --- AUDIO LOGIC ---
    def listen_and_recognize(self):
        if pygame.mixer.music.get_busy(): return None
        fs, duration = 16000, 5
        try:
            sd.stop()
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, device=self.selected_mic_index, dtype='int16')
            sd.wait()
            if pygame.mixer.music.get_busy(): return None
            audio_data = sr.AudioData(recording.tobytes(), fs, 2)
            return self.recognizer.recognize_google(audio_data, language="fr-FR")
        except: 
            return None

    def speak_with_piper(self, text):
        if not text or not self.voice: return
        try:
            clean_text = text.replace('*', '').replace('_', '').replace('#', '').strip()
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.voice.config.sample_rate)
                self.voice.synthesize_wav(clean_text, wav_file)
            buffer.seek(0)
            pygame.mixer.music.load(buffer, "wav")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy(): time.sleep(0.1)
        except: 
            pass

    # --- PAGE CATALOGUE (RECHERCHE GLOBALE) ---
    def create_catalog_page(self):
        p = ctk.CTkFrame(self.main_container, fg_color="transparent")
        ctk.CTkLabel(p, text="BIBLIOTHÈQUE DE MODÈLES", font=("Arial", 24, "bold")).pack(pady=10)
        
        self.model_entry = ctk.CTkEntry(p, placeholder_text="Chercher (installés) ou découvrir (ex: llama3, mistral)...", width=600)
        self.model_entry.pack(pady=10)
        self.model_entry.bind("<KeyRelease>", lambda e: self.refresh_catalog())
        
        self.download_frame = ctk.CTkFrame(p, fg_color="transparent")
        self.download_label = ctk.CTkLabel(self.download_frame, text="Prêt")
        self.download_label.pack()
        self.download_bar = ctk.CTkProgressBar(self.download_frame, width=500)
        self.download_bar.set(0)
        self.download_bar.pack(pady=5)
        
        self.catalog_list = ctk.CTkScrollableFrame(p)
        self.catalog_list.pack(fill="both", expand=True, padx=20, pady=10)
        self.pages["catalog"] = p
        self.refresh_catalog()

    def refresh_catalog(self):
        for w in self.catalog_list.winfo_children(): w.destroy()
        query = self.model_entry.get().lower()
        
        installed_names = []
        try:
            # 1. Modèles installés (Locaux)
            for m in ollama.list().models:
                installed_names.append(m.model)
                if not query or query in m.model.lower():
                    f = ctk.CTkFrame(self.catalog_list, fg_color="#2c3e50")
                    f.pack(fill="x", pady=2, padx=5)
                    ctk.CTkLabel(f, text=f"📦 {m.model} (Installé)").pack(side="left", padx=10)
                    ctk.CTkButton(f, text="Supprimer", width=80, fg_color="#c0392b", command=lambda x=m.model: self.delete_model(x)).pack(side="right", padx=10)
        except: 
            pass

        # 2. Modèles Distants (Recherche Intelligente dans OLLAMA_LIBRARY)
        if query:
            for lib_m in OLLAMA_LIBRARY:
                if query in lib_m and lib_m not in installed_names and (lib_m + ":latest") not in installed_names:
                    f = ctk.CTkFrame(self.catalog_list, fg_color="transparent")
                    f.pack(fill="x", pady=2, padx=5)
                    ctk.CTkLabel(f, text=f"🌐 {lib_m} (Disponible)").pack(side="left", padx=10)
                    ctk.CTkButton(f, text="Installer", width=80, fg_color="#27ae60", command=lambda x=lib_m: self.pull_model(x)).pack(side="right", padx=10)

    def pull_model(self, name):
        self.download_frame.pack(pady=10)
        threading.Thread(target=self.pull_task, args=(name,), daemon=True).start()

    def pull_task(self, name):
        try:
            for part in ollama.pull(name, stream=True):
                if 'completed' in part and 'total' in part:
                    p = part['completed'] / part['total']
                    self.after(0, lambda x=p, n=name: (self.download_bar.set(x), self.download_label.configure(text=f"Téléchargement {n} : {int(x*100)}%")))
            self.after(0, lambda: (self.download_frame.pack_forget(), self.refresh_catalog(), messagebox.showinfo("Ollama", f"Modèle {name} installé !")))
        except: 
            self.after(0, lambda: self.download_frame.pack_forget())

    def delete_model(self, name):
        try:
            ollama.delete(name)
            self.refresh_catalog()
        except: 
            pass

    # --- PAGES CHAT & ROLEPLAY ---
    def create_chat_page(self):
        p = ctk.CTkFrame(self.main_container, fg_color="transparent")
        side = ctk.CTkFrame(p, width=200); side.pack(side="left", fill="y", padx=10, pady=10)
        self.chat_model_menu = ctk.CTkOptionMenu(side, values=self.get_models())
        self.chat_model_menu.pack(pady=10)
        self.btn_call_chat = ctk.CTkButton(side, text="📞 APPEL", fg_color="#27ae60", command=self.toggle_call_chat)
        self.btn_call_chat.pack(pady=10, fill="x")
        self.chat_display = ctk.CTkTextbox(p, state="disabled", font=self.global_font)
        self.chat_display.pack(side="top", fill="both", expand=True, padx=10, pady=10)
        in_f = ctk.CTkFrame(p, fg_color="transparent"); in_f.pack(side="bottom", fill="x", padx=10, pady=10)
        self.chat_input = ctk.CTkEntry(in_f, height=45)
        self.chat_input.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.chat_input.bind("<Return>", lambda e: self.run_chat())
        self.btn_v_chat = ctk.CTkButton(in_f, text="🎤", width=45, height=45, command=lambda: self.toggle_voice_continuous(self.chat_input, self.btn_v_chat))
        self.btn_v_chat.pack(side="right")
        self.pages["chat"] = p

    def create_rp_page(self):
        p = ctk.CTkFrame(self.main_container, fg_color="transparent")
        side = ctk.CTkFrame(p, width=320); side.pack(side="left", fill="y", padx=10, pady=10)
        self.rp_model_menu = ctk.CTkOptionMenu(side, values=self.get_models())
        self.rp_model_menu.pack(pady=5)
        self.rp_name_in = ctk.CTkEntry(side, placeholder_text="Nom...")
        self.rp_name_in.pack(fill="x", padx=10, pady=5)
        self.rp_bio_in = ctk.CTkTextbox(side, height=80)
        self.rp_bio_in.pack(fill="x", padx=10, pady=5)
        btn_f = ctk.CTkFrame(side, fg_color="transparent"); btn_f.pack(fill="x", padx=10)
        ctk.CTkButton(btn_f, text="Ajouter", width=80, command=self.add_char).pack(side="left", padx=2)
        ctk.CTkButton(btn_f, text="Suppr", width=80, fg_color="#c0392b", command=self.delete_char).pack(side="right", padx=2)
        self.btn_call_rp = ctk.CTkButton(side, text="📞 APPEL RP", fg_color="#27ae60", command=self.toggle_call_rp)
        self.btn_call_rp.pack(pady=10, fill="x")
        self.rp_list = ctk.CTkScrollableFrame(side, label_text="Personnages")
        self.rp_list.pack(fill="both", expand=True, padx=10, pady=5)
        self.rp_display = ctk.CTkTextbox(p, state="disabled", font=self.global_font)
        self.rp_display.pack(side="top", fill="both", expand=True, padx=10, pady=10)
        in_f = ctk.CTkFrame(p, fg_color="transparent"); in_f.pack(side="bottom", fill="x", padx=10, pady=10)
        self.rp_input = ctk.CTkEntry(in_f, height=45)
        self.rp_input.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.rp_input.bind("<Return>", lambda e: self.run_rp())
        self.btn_v_rp = ctk.CTkButton(in_f, text="🎤", width=45, height=45, command=lambda: self.toggle_voice_continuous(self.rp_input, self.btn_v_rp))
        self.btn_v_rp.pack(side="right")
        self.pages["rp"] = p
        self.update_rp_list()

    # --- APPELS ET FLUX ---
    def toggle_call_chat(self):
        self.is_calling = not self.is_calling
        self.btn_call_chat.configure(fg_color="red" if self.is_calling else "#27ae60", text="🔴" if self.is_calling else "📞 APPEL")
        if self.is_calling: threading.Thread(target=self.call_loop, args=(self.chat_display, self.chat_model_menu.get(), "Assistant"), daemon=True).start()

    def toggle_call_rp(self):
        self.is_calling = not self.is_calling
        self.btn_call_rp.configure(fg_color="red" if self.is_calling else "#27ae60", text="🔴" if self.is_calling else "📞 APPEL RP")
        if self.is_calling:
            active = [n for n, c in self.checks.items() if c.get()]
            threading.Thread(target=self.call_loop_rp, args=(active,), daemon=True).start()

    def call_loop(self, disp, mod, name):
        while self.is_calling:
            if not pygame.mixer.music.get_busy():
                text = self.listen_and_recognize()
                if text:
                    self.after(0, lambda t=text: self.append_msg(disp, "Moi", t))
                    self.call_stream_sync(disp, mod, text, name)
            time.sleep(0.3)

    def call_loop_rp(self, active):
        while self.is_calling:
            if not pygame.mixer.music.get_busy():
                text = self.listen_and_recognize()
                if text:
                    self.after(0, lambda t=text: self.append_msg(self.rp_display, "Moi", t))
                    for n in active:
                        p = f"Tu es {n}. Bio: {self.characters[n]}. Réponds à: {text}"
                        self.call_stream_sync(self.rp_display, self.rp_model_menu.get(), p, n)
            time.sleep(0.3)

    def call_stream_sync(self, disp, mod, p, lbl):
        full_res = ""
        self.after(0, lambda: disp.configure(state="normal") or disp.insert("end", f"● {lbl}: "))
        try:
            for chunk in ollama.chat(model=mod, messages=[{'role':'user','content':p}], stream=True):
                if not self.is_calling: break
                c = chunk['message']['content']
                full_res += c
                self.after(0, lambda x=c: disp.insert("end", x) or disp.see("end"))
            self.after(0, lambda: disp.insert("end", "\n\n") or disp.configure(state="disabled"))
            if full_res and self.is_calling: self.speak_with_piper(full_res)
        except: pass

    def run_chat(self):
        m = self.chat_input.get()
        if m: 
            self.chat_input.delete(0, "end")
            self.append_msg(self.chat_display, "Moi", m)
            threading.Thread(target=lambda: self.call_stream_text_only(self.chat_display, self.chat_model_menu.get(), m, "Assistant"), daemon=True).start()

    def run_rp(self):
        m = self.rp_input.get()
        active = [n for n, c in self.checks.items() if c.get()]
        if m and active:
            self.rp_input.delete(0, "end")
            self.append_msg(self.rp_display, "Moi", m)
            for n in active:
                p = f"Tu es {n}. Bio: {self.characters[n]}. Réponds à: {m}"
                threading.Thread(target=lambda pr=p, name=n: self.call_stream_text_only(self.rp_display, self.rp_model_menu.get(), pr, name), daemon=True).start()

    def call_stream_text_only(self, disp, mod, p, lbl):
        self.after(0, lambda: disp.configure(state="normal") or disp.insert("end", f"● {lbl}: "))
        try:
            for chunk in ollama.chat(model=mod, messages=[{'role':'user','content':p}], stream=True):
                c = chunk['message']['content']
                self.after(0, lambda x=c: disp.insert("end", x) or disp.see("end"))
        except: 
            pass
        self.after(0, lambda: disp.insert("end", "\n\n") or disp.configure(state="disabled"))

    # --- PARAMÈTRES (TEST MIC + FONTS) ---
    def create_settings_page(self):
        p = ctk.CTkFrame(self.main_container, fg_color="transparent")
        ctk.CTkLabel(p, text="PARAMÈTRES SYSTÈME", font=("Arial", 24, "bold")).pack(pady=20)
        
        ctk.CTkLabel(p, text="CHOIX DE LA POLICE", font=self.global_font).pack()
        self.font_menu = ctk.CTkOptionMenu(p, values=sorted(font.families()), command=self.set_font)
        self.font_menu.pack(pady=5)
        self.font_menu.set(self.app_config.get("font_family", "Arial"))

        ctk.CTkLabel(p, text="CONFIGURATION AUDIO", font=self.global_font).pack(pady=10)
        self.mic_menu = ctk.CTkOptionMenu(p, values=["Chargement..."], command=self.set_mic)
        self.mic_menu.pack()
        self.btn_test_mic = ctk.CTkButton(p, text="Démarrer Test Micro", command=self.toggle_mic_test)
        self.btn_test_mic.pack(pady=10)
        self.mic_bar = ctk.CTkProgressBar(p, width=300)
        self.mic_bar.set(0)
        self.mic_bar.pack(pady=5)
        
        ctk.CTkLabel(p, text="APPARENCE", font=self.global_font).pack(pady=10)
        t_f = ctk.CTkFrame(p, fg_color="transparent")
        t_f.pack()
        for t in THEMES_CONFIG.keys(): 
            ctk.CTkButton(t_f, text=t, width=80, command=lambda x=t: self.apply_theme(x)).pack(side="left", padx=2)
        
        ctk.CTkButton(p, text="🖼️ Changer Fond d'écran", command=self.import_bg).pack(pady=20)
        self.pages["settings"] = p

    def toggle_mic_test(self):
        self.is_testing_mic = not self.is_testing_mic
        self.btn_test_mic.configure(text="Arrêter Test" if self.is_testing_mic else "Démarrer Test")
        if self.is_testing_mic: threading.Thread(target=self.run_test, daemon=True).start()

    def run_test(self):
        def cb(indata, frames, time, status):
            if self.is_testing_mic:
                vol = np.linalg.norm(indata) * 20
                self.mic_bar.set(min(vol/100, 1.0))
        try:
            with sd.InputStream(device=self.selected_mic_index, callback=cb, channels=1):
                while self.is_testing_mic: sd.sleep(100)
        except: 
            self.is_testing_mic = False
        finally: 
            self.mic_bar.set(0)

    def toggle_voice_continuous(self, entry, btn):
        self.is_listening = not self.is_listening
        btn.configure(fg_color="red" if self.is_listening else "#3b8ed0")
        if self.is_listening: threading.Thread(target=self.voice_task, args=(entry,), daemon=True).start()

    def voice_task(self, entry):
        while self.is_listening:
            text = self.listen_and_recognize()
            if text: self.after(0, lambda t=text: entry.insert("end", f" {t}"))

    # --- HELPERS SYSTÈME ---
    def set_font(self, f): 
        self.app_config["font_family"] = f
        self.save_app_config()
        self.update_font_objects()

    def import_bg(self):
        p = filedialog.askopenfilename()
        if p: 
            self.app_config["bg_path"] = p
            self.save_app_config()
            self.load_background(p)

    def load_background(self, p):
        try:
            img = Image.open(p)
            ctk_img = ctk.CTkImage(img, img, size=(self.winfo_width(), self.winfo_height()))
            self.bg_canvas.configure(image=ctk_img)
        except: 
            pass

    def resizer(self, e):
        if self.app_config.get("bg_path"): self.load_background(self.app_config["bg_path"])

    def load_selected_voice(self, v):
        if v and v != "Aucune voix":
            try: 
                self.voice = PiperVoice.load(os.path.join(PIPER_DIR, v))
                self.app_config["voice_model"] = v
                self.save_app_config()
            except: 
                pass

    def refresh_mics(self):
        try:
            devs = sd.query_devices()
            mics = [f"{i}: {d['name']}" for i, d in enumerate(devs) if d['max_input_channels'] > 0]
            if mics: 
                self.mic_menu.configure(values=mics)
                self.mic_menu.set(mics[0])
                self.selected_mic_index = int(mics[0].split(":")[0])
        except: 
            pass

    def set_mic(self, val): 
        self.selected_mic_index = int(val.split(":")[0])

    def apply_theme(self, t):
        self.app_config["theme"] = t
        c = THEMES_CONFIG[t]
        self.configure(fg_color=c["main"])
        self.sidebar.configure(fg_color=c["sidebar"])
        for b in self.menu_buttons: b.configure(fg_color=c["accent"])

    def update_rp_list(self):
        for w in self.rp_list.winfo_children(): w.destroy()
        self.checks = {}
        for n in self.characters:
            f = ctk.CTkFrame(self.rp_list, fg_color="transparent")
            f.pack(fill="x")
            self.checks[n] = ctk.CTkCheckBox(f, text=n)
            self.checks[n].pack(side="left")

    def add_char(self):
        n = self.rp_name_in.get()
        if n: 
            self.characters[n] = self.rp_bio_in.get("1.0", "end-1c")
            self.save_chars()
            self.update_rp_list()

    def delete_char(self):
        n = self.rp_name_in.get()
        if n in self.characters: 
            del self.characters[n]
            self.save_chars()
            self.update_rp_list()

    def get_models(self):
        try: 
            return [m.model for m in ollama.list().models]
        except: 
            return ["llama3"]

    def update_font_objects(self): 
        self.global_font = (self.app_config.get("font_family", "Arial"), 16, "bold")

    def load_app_config(self): 
        return json.load(open(CONFIG_FILE, "r")) if os.path.exists(CONFIG_FILE) else {"theme": "Cherry"}

    def save_app_config(self): 
        json.dump(self.app_config, open(CONFIG_FILE, "w"))

    def load_chars(self): 
        return json.load(open(SAVE_FILE, "r")) if os.path.exists(SAVE_FILE) else {"Assistant": "Aide-moi."}

    def save_chars(self): 
        json.dump(self.characters, open(SAVE_FILE, "w"))

    def append_msg(self, d, s, t):
        d.configure(state="normal")
        d.insert("end", f"● {s}\n{t}\n\n")
        d.configure(state="disabled")
        d.see("end")

if __name__ == "__main__":
    app = UltimateOllamaApp()
    app.mainloop()