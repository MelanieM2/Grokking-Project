"""
BASELINE EXPERIMENT: MLP mit Feature-Scaling (x/P) und AdamW Optimizer
--------------------------------------------------
Dieses Skript repräsentiert die erste Phase meines Projekts. 
Beobachtung: Durch die Skalierung lernt das Modell die Arithmetik sehr effizient (100% Train-Acc), 
zeigt aber kaum das klassische "Grokking"-Verhalten, da die kontinuierlichen Eingaben 
dem Netzwerk eine zu starke Heuristik bieten. 

Dies führte zur Entscheidung, auf diskrete Embeddings umzusteigen, um den 
Übergang von Memorization zu Generalization sauberer zu isolieren.

Dazu verwenden wir hier AdamW statt SGD.

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import datetime
import glob
import pandas as pd

# Überprüfung der Hardware-Ressourcen
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# -----------------------------------------------------------------------------
# Hyperparameter & Experiment-Konfiguration
# -----------------------------------------------------------------------------
P = 97 
EPOCHS_CONTINUE = 10000     
BATCH_SIZE = 128    

# ÄNDERUNG: AdamW benötigt oft eine kleinere Lernrate als SGD (Startwert 0.001 empfohlen).
LR = 0.001             

# ÄNDERUNG: Einführung von Weight Decay für AdamW.
# Dies ersetzt die L2-Regularisierung direkt im Optimierungsschritt.
WEIGHT_DECAY = 0.01  

SAVE_EVERY = 100            
KEEP_LAST = 3               
train_frac = 0.7            
HIDDEN_SIZE = 64            
NUM_LAYERS = 2              

# Architektur-Label angepasst
MODEL_ARCH = "MLP_AdamW" # default für GRokking
# oder 
# MODEL_ARCH = "MLP_SGD"      #<--nur zum Ziel der Illustration, zwinkt kein Grokking.
                        # Falls dies ausgewählt wird, nicht vergessen den RESUME_RUN anzupassen
                        # Falls dies ausgewahlt ist WEIGHT_DECAY irrelevant




# -----------------------------------------------------------------------------
# Experiment-Management & Pfad-Konfiguration
# -----------------------------------------------------------------------------
RESUME_RUN = None  
#RESUME_RUN = "./runs/grok_P97_MLP_Baseline_SGD_20260213_140536" #<--Falls der Ordner schon existiert, dann hier
                                                                  #dessen Pfad eingeben und RESUME_RUN=None oben auskommentieren

if RESUME_RUN and os.path.exists(RESUME_RUN):
    RUN_DIR = RESUME_RUN
    print(f"🔄 Setze Training fort: {RUN_DIR}")
else:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = f"./runs/grok_P{P}_Baseline_{MODEL_ARCH}_{timestamp}"
    os.makedirs(RUN_DIR, exist_ok=True) 
    print(f"📂 Erstelle NEUEN Ordner: {RUN_DIR}")

CKPT_DIR = os.path.join(RUN_DIR, "checkpoints")
CSV_PATH = os.path.join(RUN_DIR, "history.csv")       
INDICES_PATH = os.path.join(RUN_DIR, "train_indices.npy") 
os.makedirs(CKPT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Datensatz & Skalierung
# -----------------------------------------------------------------------------
xs, ys = [], []
for a in range(P):
    for b in range(P):
        xs.append([a, b])
        ys.append((a + b) % P)

# --- WICHTIGER HINWEIS ZUR DATENVORBEREITUNG ---
# Die Skalierung der Eingabewerte (xs) auf das Intervall [0, 1] dient der 
# numerischen Stabilität und verhindert explodierende Gradienten. 
# Die Zielwerte (ys) verbleiben als Integers (0 bis P-1), da das Problem 
# als Klassifizierungsaufgabe mit P diskreten Klassen definiert ist.
# Die Skalierung auf [0, 1] via (P-1) dient der numerischen Stabilität.
# Sie gibt dem MLP eine kontinuierliche Heuristik (numerische Nachbarschaft).
xs = np.array(xs, dtype=np.float32) / (P - 1)
ys = np.array(ys, dtype=np.int32)


# Um Grokking valide zu messen, muss der Split über Resumes hinweg identisch bleiben.
if os.path.exists(INDICES_PATH):
    print("📂 Lade gespeicherte Train-Indizes für konsistenten Split...")
    train_indices = np.load(INDICES_PATH)

    # Ermittlung der Test-Indizes (Komplementärmenge der Trainings-Daten)
    all_indices = np.arange(len(xs))
    mask = np.ones(len(xs), dtype=bool)
    mask[train_indices] = False
    test_indices = all_indices[mask]
else:
    # Zufällige Auswahl der Trainingsdaten beim ersten Start des Experiments
    print("🎲 Erstelle neue Train-Indizes und speichere sie...")
    perm = np.random.permutation(len(xs))
    train_size = int(train_frac * len(xs))
    train_indices = perm[:train_size]
    test_indices = perm[train_size:]

    # Persistierung der Indizes für absolute Reproduzierbarkeit
    np.save(INDICES_PATH, train_indices)

# Finale Zuweisung der Daten-Splits
x_train, y_train = xs[train_indices], ys[train_indices]
x_test, y_test   = xs[test_indices], ys[test_indices]

print(f"✅ Datenvorbereitung abgeschlossen: Train={len(x_train)}, Test={len(x_test)}")

# -----------------------------------------------------------------------------
# Modell-Definition (MLP entweder mit AdamW & Weight Decay oder SGD)
# -----------------------------------------------------------------------------

#Hier können wir wählen zwischen den Optimizer AdamW oder SGD. 
# Aber SGD ist nur zur Illustration: Weil Weight-Decay bei der Weight-Norm und L2-Regularisierung 
# entscheiden für Grokking ist, SGD wird nur verwenden um zu zeigen wie ohne Weight-Decay bei der 
# Weight-Norm, bringt die L2-Regulasierung kein Grokking bei einem sehr kleinen Trainingsatz
#

def build_simple_mlp_scaling_AdamW(P, HIDDEN_SIZE, WEIGHT_DECAY):
    model = tf.keras.Sequential([
        # ÄNDERUNG: kernel_regularizer entfernt. 
        # AdamW wendet Weight Decay direkt auf die Gewichte an, 
        # was bei adaptiven Optimizern mathematisch korrekter ist als L2.
        tf.keras.layers.Dense(HIDDEN_SIZE, activation='tanh', input_shape=(2,)),
        tf.keras.layers.Dense(HIDDEN_SIZE, activation='tanh'),
        tf.keras.layers.Dense(P)
    ])
    
    # AdamW Optimizer.
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )
    return model


def build_simple_mlp_scaling_SGD(P, HIDDEN_SIZE):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(HIDDEN_SIZE, activation='tanh', 
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                              input_shape=(2,)),
        tf.keras.layers.Dense(HIDDEN_SIZE, activation='tanh', 
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dense(P)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=LR, momentum=0.9, nesterov=True),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )
    return model

def get_model(arch_name):
    """Optimizerauswahl zwischen AdamW oder SGD"""
    match arch_name:
        case "MLP_AdamW": 
            return build_simple_mlp_scaling_AdamW(P, HIDDEN_SIZE, WEIGHT_DECAY)
        case "MLP_SGD":
            return build_simple_mlp_scaling_SGD(P, HIDDEN_SIZE)
        case _:
            # Sicherheitsnetz für Tippfehler in der Konfiguration
            raise ValueError(f"Unbekannter Modell-Typ: {arch_name}")

# Erstmalige Modell-Erstellung zur Strukturprüfung
model = get_model(MODEL_ARCH) 
model.summary()


# -----------------------------------------------------------------------------
# Checkpoint-Management: Training fortsetzen oder neu starten
# -----------------------------------------------------------------------------
# Suche nach vorhandenen .keras Dateien im Checkpoint-Verzeichnis
# Sortierung erfolgt numerisch nach der Epochenzahl im Dateinamen

checkpoints = sorted(glob.glob(os.path.join(CKPT_DIR, "*.keras")),
                     key=lambda x: int(os.path.basename(x).replace('_FINAL', '').split('_')[-1].replace('.keras', '')))

if checkpoints:
    latest = checkpoints[-1]
    print(f"🔄 Bestehender Checkpoint gefunden: {latest}")
    
    # Laden des exakten Modellzustands inklusive Optimizer-Status (zB, AdamW-Momente bei AdamW)
    model = tf.keras.models.load_model(latest) 
    
    # Extraktion der letzten Epoche aus dem Dateinamen
    clean_name = os.path.basename(latest).replace('_FINAL', '')
    initial_epoch = int(clean_name.split("_")[-1].replace(".keras", ""))
else:
    print("⚠️ Keine Checkpoints gefunden. Initialisiere neues Modell für Epoche 0.")
    initial_epoch = 0
    # Das Modell wurde oben bereits durch model

print(f"▶️ Status: Training startet/geht weiter ab Epoche {initial_epoch}")



# -----------------------------------------------------------------------------
#  Klassen: Custom Callbacks für Logging & Analyse
# -----------------------------------------------------------------------------

class CSVLoggerAppend(tf.keras.callbacks.Callback):
    """
    Spezialisiertes Logging: Speichert Metriken und die L2-Gewichtsnorm in eine CSV.
    Unterstützt das nahtlose Fortsetzen (Resume) von Experimenten.
    """
    def __init__(self, initial_epoch=0, csv_path="history.csv"):
        super().__init__()
        self.initial_epoch = initial_epoch
        self.csv_path = csv_path

    def on_train_begin(self, logs=None):
        """
        Bereinigt die CSV-Datei beim Start, um Inkonsistenzen nach einem 
        Neustart des Trainings zu vermeiden. Behält nur Daten bis zur aktuellen Epoche.
        """
        if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
            try:
                df = pd.read_csv(self.csv_path)
                # Filtert alle Zeilen, die zeitlich nach dem Resume-Punkt liegen
                df_cleaned = df[df['epoch'] <= self.initial_epoch]
                df_cleaned.to_csv(self.csv_path, index=False)
                print(f"🧹 CSV bereinigt: Datenstand bis Epoche {self.initial_epoch} fixiert.")
            except Exception as e:
                print(f"⚠️ Warnung beim Bereinigen der CSV: {e}")

    def on_epoch_end(self, epoch, logs=None):
        """
        Berechnet nach jeder Epoche die Gewichtsnorm und schreibt alle Metriken in die CSV.
        """
        logs = logs or {}  
        real_epoch = epoch + 1  # Keras zählt intern ab 0, wir loggen ab 1
        
        # Flexibler Abruf der Accuracy (unterstützt verschiedene Keras-Namenskonventionen)
        # next() durchläuft die Liste und gibt den ersten gefundenen Key zurück (oder None)
        # Das ersetzt eine klassische for-Schleife mit break und wirkt sehr "pythonic".
        train_acc = next((logs.get(k) for k in ["accuracy", "sparse_categorical_accuracy", "acc"] if k in logs), None)
        val_acc = next((logs.get(k) for k in ["val_accuracy", "val_sparse_categorical_accuracy", "val_acc"] if k in logs), None)
        
        train_loss = logs.get("loss")
        val_loss = logs.get("val_loss")

        # -------- Berechnung der L2-Gewichtsnorm ||W||₂ --------
        # Ein kritischer Indikator für Grokking: Strukturreorganisation zeigt sich oft in der Norm-Dynamik.
        w_norm = tf.sqrt(tf.add_n([tf.reduce_sum(tf.square(w)) for w in self.model.trainable_weights])).numpy()

        # Daten persistent in die CSV-Datei schreiben
        header_exists = os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not header_exists or os.path.getsize(self.csv_path) == 0:
                writer.writerow(["epoch", "train_acc", "test_acc", "train_loss", "test_loss", "W_norm"])
            
            writer.writerow([real_epoch, train_acc, val_acc, train_loss, val_loss, w_norm])



# -----------------------------------------------------------------------------
# Klassen: Fortschrittssicherung & Checkpointing
# -----------------------------------------------------------------------------

class KeepLastCheckpoints(tf.keras.callbacks.Callback):
    """
    Verwaltet regelmäßige Checkpoints und hält den Speicherplatz sauber,
    indem nur die aktuellsten Modellzustände aufbewahrt werden.
    """
    def __init__(self, initial_epoch, ckpt_dir, run_dir, config_data, save_every=1, keep_last=5):
        super().__init__()
        self.initial_epoch = initial_epoch
        self.ckpt_dir = ckpt_dir
        self.run_dir = run_dir 
        self.config_data = config_data 
        self.save_every = save_every
        self.keep_last = keep_last
        # Pfad für die Live-Dokumentation der Parameter
        self.log_fname = os.path.join(self.run_dir, "training_settings.txt")

    def _write_current_status(self, epoch):
        """Aktualisiert die Log-Datei live mit aktuellen Optimizer-Parametern."""
        with open(self.log_fname, "w") as f:
            f.write("=== EXPERIMENT CONFIGURATION (LIVE UPDATE) ===\n")
            f.write(f"Zuletzt aktualisiert bei Epoche: {epoch}\n\n")

            # 1. Basis-Parameter schreiben
            for key, value in self.config_data.items():
                f.write(f"{key}: {value}\n")

            # 2. Optimizer-Details robust erfassen
            if hasattr(self.model, 'optimizer'):
                opt = self.model.optimizer
                opt_config = opt.get_config()
                
                # Namens-Logik: Priorität auf lesbares "AdamW" oder "SGD" für den Plotting-Skript
                raw_name = getattr(opt, 'name', opt_config.get('name', opt.__class__.__name__))
                if MODEL_ARCH == "MLP_AdamW":
                    opt_name = str(raw_name).replace('adamw', 'AdamW').capitalize()
                else: 
                    opt_name = str(raw_name).replace('sgd', 'SGD').capitalize()

                f.write("\n=== OPTIMIZER DETAILS (CURRENT) ===\n")
                # Wichtig: "Optimizer Name" nutzen, damit der Plotter den Key findet
                f.write(f"Optimizer Name: {opt_name}\n")
                
                # Alle weiteren Config-Parameter (LR, Decay, etc.)
                for key, value in opt_config.items():
                    if key != 'name':  # Überspringt das kleine 'adamw', oder 'sgd' in training_settings.txt
                        f.write(f"{key}: {value}\n")

    def on_epoch_end(self, epoch, logs=None):
        real_epoch = epoch + 1 

        if real_epoch % self.save_every == 0:
            fname = os.path.join(self.ckpt_dir, f"model_epoch_{real_epoch}.keras")  
            self.model.save(fname)
            self._write_current_status(real_epoch)

            print(f"\n💾 Checkpoint gespeichert: Epoche {real_epoch}")

            # Bereinigung: Sortiert Checkpoints numerisch nach Epochenzahl
            ckpts = sorted(glob.glob(os.path.join(self.ckpt_dir, "*.keras")),
                           key=lambda x: int(os.path.basename(x).replace('_FINAL', '').split('_')[-1].replace('.keras', '')))

            # Behält nur die letzten 'keep_last' Dateien, um Speicherplatz zu sparen
            while len(ckpts) > self.keep_last:
                old = ckpts.pop(0)
                # Sicherheitscheck: Lösche niemals finalisierte Modelle
                if "_FINAL" not in old:
                    os.remove(old)
                    print(f"🗑 Alte Epoche gelöscht: {old}")

class SaveFinalCheckpoint(tf.keras.callbacks.Callback):
    """Sichert den finalen Modellzustand und dokumentiert die End-Konfiguration."""
    def __init__(self, save_dir, config_data): 
        super().__init__()   
        self.save_dir = save_dir
        self.config_data = config_data
        self.log_fname = os.path.join(self.save_dir, "training_settings.txt")

    def _write_config(self, include_optimizer=False):
        """Hilfsmethode: Schreibt Parameter & Optimizer-Details sauber in die training_settings.txt"""
        with open(self.log_fname, "w") as f:
            f.write("=== EXPERIMENT CONFIGURATION ===\n")
            for key, value in self.config_data.items():
                f.write(f"{key}: {value}\n")

            if include_optimizer and hasattr(self.model, 'optimizer'):
                opt = self.model.optimizer
                opt_config = opt.get_config()
                
                # Wir nutzen exakt den Key, den der Plot-Skript erwartet: "Optimizer Name"
                # Und wir stellen sicher, dass AdamW, oder SGD schön formatiert ist.
                raw_name = getattr(opt, 'name', opt_config.get('name', opt.__class__.__name__))
                if MODEL_ARCH == "MLP_AdamW":
                    opt_name = str(raw_name).replace('adamw', 'AdamW').capitalize()
                else: 
                    opt_name = str(raw_name).replace('sgd', 'SGD').capitalize()


                f.write("\n=== OPTIMIZER DETAILS ===\n")
                f.write(f"Optimizer: {opt_name}\n")  # <--- EXAKTER KEY FÜR DEN PLOT
                
                # Alle weiteren technischen Details des Optimizers (LR, Weight Decay etc.)
                for key, value in opt_config.items():
                    # Wir überspringen 'name', damit es nicht doppelt (und klein) erscheint
                    if key != 'name':
                        f.write(f"{key}: {value}\n")

    def on_train_begin(self, logs=None):
        # Dokumentiert die Start-Parameter unmittelbar vor Trainingsbeginn
        self._write_config(include_optimizer=False)
        print(f"📝 Experiment-Konfiguration initial gespeichert: {self.log_fname}")       

    def on_train_end(self, logs=None):
        # Sichert das Modell nach dem regulären Ende aller Epochen
        final_epoch = self.params['epochs']  
        model_fname = os.path.join(self.save_dir, f"model_epoch_{final_epoch}_FINAL.keras")
        self.model.save(model_fname)
        self._write_config(include_optimizer=True)
        print(f"💾 Training beendet. Finale Daten gespeichert in: {self.save_dir}")







# -----------------------------------------------------------------------------
# Laufzeit-Konfiguration & Experiment-Parameter
# -----------------------------------------------------------------------------

# Berechnung der Ziel-Epoche für die Fortsetzungs-Logik
TOTAL_TARGET_EPOCHS = initial_epoch + EPOCHS_CONTINUE

# Zentrale Konfigurations-Daten für die Dokumentation (training_settings.txt)
# Dies ermöglicht eine vollständige Rekonstruktion des Experiments.

experiment_params = {
    "P": P,                                  # Modulo-Basis
    "INITIAL_EPOCHS": initial_epoch,          # Startpunkt (0 bei Neustart)
    "CONTINUE_BY": EPOCHS_CONTINUE,           # Geplante Zusatz-Epochen
    "TOTAL_TARGET_EPOCHS": TOTAL_TARGET_EPOCHS,
    "MODEL_TYPE": MODEL_ARCH.capitalize(),    # Formatiert (hier in diesem Skript gibt es nur MLP miz AdamW oder SGD)
    "BATCH_SIZE": BATCH_SIZE, 
    "LR": LR,                                # Lernrate
    "WEIGHT_DECAY": WEIGHT_DECAY,            # Entscheidend für die Regularisierung (Grokking)
    "SAVE_EVERY": SAVE_EVERY, 
    "KEEP_LAST": KEEP_LAST,                  # Anzahl rotierender Checkpoints
    "train_frac": train_frac,                # Anteil der genutzten Trainingsdaten
    "HIDDEN_SIZE": HIDDEN_SIZE,              # Breite des Modells
    "NUM_LAYERS": NUM_LAYERS                 # Tiefe des Modells
}

print(f"📈 Experiment-Setup: Ziel ist Epoche {TOTAL_TARGET_EPOCHS}")


# -----------------------------------------------------------------------------
# Callback-Integration
# -----------------------------------------------------------------------------

# Zusammenführung der spezialisierten Logik-Komponenten
callbacks = [
    # Custom Logging für Metriken und Gewichtsnorm
    CSVLoggerAppend(initial_epoch=initial_epoch, csv_path=CSV_PATH),
    
    # Automatisches Checkpointing & Live-Parameter-Update
    KeepLastCheckpoints(
        initial_epoch=initial_epoch, 
        ckpt_dir=CKPT_DIR, 
        run_dir=RUN_DIR,              
        config_data=experiment_params, 
        save_every=SAVE_EVERY, 
        keep_last=KEEP_LAST
    ),
    
    # Finale Sicherung des Modells und der Metadaten bei regulärem Ende
    SaveFinalCheckpoint(save_dir=RUN_DIR, config_data=experiment_params)
]

# -----------------------------------------------------------------------------
# Training: Die Suche nach "Grokking"
# -----------------------------------------------------------------------------

# Start (oder Fortsetzung) des Trainingsprozesses
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=BATCH_SIZE,
    epochs=TOTAL_TARGET_EPOCHS,
    initial_epoch=initial_epoch,
    
    # Verbose-Level: 2 ist ideal für automatisierte Logs (reduziert Output-Rauschen),
    # während unsere Custom-Callbacks detaillierte Infos in CSV/TXT schreiben.
    verbose=2,  
    
    callbacks=callbacks
)

print(f"🎉 Experiment erfolgreich abgeschlossen. Daten befinden sich in: {RUN_DIR}")



# -----------------------------------------------------------------------------
# Post-Processing: Datenvalidierung & Bereinigung
# -----------------------------------------------------------------------------

def load_clean_history(csv_path):
    """
    Lädt die Trainings-Historie und stellt die Datenintegrität sicher.
    Filtert unvollständige Zeilen (z.B. nach einem Systemabsturz während des Schreibvorgangs).
    """
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["epoch", "train_acc", "test_acc", "train_loss", "test_loss", "w_norm"])
    
    cleaned_rows = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
            for row in reader:
                # Validierung: Nur Zeilen mit korrekter Spaltenanzahl übernehmen
                if len(row) == len(headers):
                    cleaned_rows.append(row)
        except StopIteration:
            return pd.DataFrame()

    df = pd.DataFrame(cleaned_rows, columns=headers)
    
    # Konvertierung aller Spalten in numerische Formate (für spätere Plots)
    for c in headers:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    return df

# Finales Laden der Daten für die anschließende Analyse/Visualisierung
history_df = load_clean_history(CSV_PATH)

print(f"✅ Training und Datenvalidierung abgeschlossen. Datensätze: {len(history_df)}")