import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import datetime
import glob
import pandas as pd

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# ------------------------------------------------------------
# 🔧 Einstellungen
# ------------------------------------------------------------
P = 97
EPOCHS_CONTINUE = 10000     # wie lange wir noch *weiter* trainieren es sei den wir beginnen von anfang an
BATCH_SIZE = 128             #max 128
LR = 0.01               #Initial bis Epoche 80K war LR 0.05. Ab Epoche 80K habe ich die LR auf 0.03 untergestiegen
WEIGHT_DECAY = 0.1    # kommt für AdamW optimzer, nicht für SGD
SAVE_EVERY = 100
KEEP_LAST = 3
train_frac = 0.7

HIDDEN_SIZE = 64
NUM_LAYERS = 2

# ------------------------------------------------------------
# 📁 EXISTIERENDER Run-Ordner (NICHT ändern!)
# ------------------------------------------------------------
# ------------------------------------------------------------
# 📂 Run-Verzeichnis wählen
# ------------------------------------------------------------
# FALL A: Wir wollen einen BESTEHENDEN Run fortsetzen:
# Geben wir hier den Ordnernamen manuell ein, wenn wir weitermachen wollen:

#RESUME_RUN = None  # Auf None lassen für einen ganz NEUEN Run

RESUME_RUN = "./runs/grok_P97_20260129_131959" 

if RESUME_RUN and os.path.exists(RESUME_RUN):
    RUN_DIR = RESUME_RUN
    print(f"🔄 Setze Training in existierendem Ordner fort: {RUN_DIR}")
else:
    # FALL B: Ganz neuer Run mit Zeitstempel
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    #Bauen wir den Ordner RUN_DIR
    RUN_DIR = f"./runs/grok_P{P}_{timestamp}"

    #Erstellen wir den Ordner RUN_DIR
    #dies garantiert dass, RUND_DIR wirklich (mit der richtigen Namen in Speicherraum) existiert:
    os.makedirs(RUN_DIR, exist_ok=True) 
    print(f"📂 Erstelle NEUEN Run-Ordner: {RUN_DIR}")

#Bauen wir den Unterordner CKPT_DIR
CKPT_DIR = os.path.join(RUN_DIR, "checkpoints")

#und definieren wir die Dateien in CKPT_DIR, die log-Info des Experiments beibehalten
CSV_PATH = os.path.join(RUN_DIR, "history.csv")
INDICES_PATH = os.path.join(RUN_DIR, "train_indices.npy")

# Erstellen wir zuerst den Unterordner (ganz wichtig!)
os.makedirs(CKPT_DIR, exist_ok=True)
print(f"✅ Definitiv erstellt: {os.path.abspath(RUN_DIR)}")

# ------------------------------------------------------------
# 🧮 Datensatz (a + b) mod P
# ------------------------------------------------------------
xs, ys = [], []
for a in range(P):
    for b in range(P):
        xs.append([a, b])
        ys.append((a + b) % P)

xs = np.array(xs, dtype=np.float32) / (P - 1)
ys = np.array(ys, dtype=np.int32)

# ------------------------------------------------------------
# 🔀 Shuffle + Split (Konsistente Indizes)
# ------------------------------------------------------------
if os.path.exists(INDICES_PATH):
    print("📂 Lade gespeicherte Train-Indizes für konsistenten Split...")
    train_indices = np.load(INDICES_PATH)
    
    # Erstelle Test-Indizes aus allen, die NICHT in Train sind
    all_indices = np.arange(len(xs))
    # Wir filtern die Indizes, die nicht in train_indices enthalten sind
    mask = np.ones(len(xs), dtype=bool)
    mask[train_indices] = False
    test_indices = all_indices[mask]
else:
    print("🎲 Erstelle neue Train-Indizes und speichere sie...")
    perm = np.random.permutation(len(xs))
    train_size = int(train_frac * len(xs))
    train_indices = perm[:train_size]
    test_indices = perm[train_size:]
    
    # Speichern, damit beim nächsten Resume alles gleich bleibt
    np.save(INDICES_PATH, train_indices)

# Daten basierend auf den konsistenten Indizes zuweisen
x_train, y_train = xs[train_indices], ys[train_indices]
x_test, y_test   = xs[test_indices], ys[test_indices]

print(f"✅ Split fertig: Train={len(x_train)}, Test={len(x_test)}")


# ------------------------------------------------------------
# 🛠 Modell-Definition
# ------------------------------------------------------------
# ... (dein Code bis zur Modell-Definition Funktion) ...

def build_model():
    model = tf.keras.Sequential([
        # First Hidden Layer 
        tf.keras.layers.Dense(HIDDEN_SIZE, activation='tanh', 
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        
        # Second Hidden Layer
        tf.keras.layers.Dense(HIDDEN_SIZE, activation='tanh', 
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        
        # Output Layer (P classes)
        tf.keras.layers.Dense(P)
    ])

    model.compile(
            optimizer=tf.keras.optimizers.SGD(
            learning_rate=LR,
            momentum=0.9,
            nesterov=True
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    
    return model

# ------------------------------------------------------------
# 📦 Resume: Checkpoint wiederherstellen
# ------------------------------------------------------------
# WICHTIG: Hier entscheiden wir, ob wir neu bauen ODER laden
checkpoints = sorted(glob.glob(os.path.join(CKPT_DIR, "*.keras")),
                     key=lambda x: int(os.path.basename(x).split('_')[-1].replace('.keras', '').replace('_FINAL','')))

if checkpoints:
    latest = checkpoints[-1]
    print(f"🔄 Lade letzten Checkpoint: {latest}")
    model = tf.keras.models.load_model(latest) # Lade existierendes Modell
    filename = os.path.basename(latest)
    initial_epoch = int(filename.split("_")[-1].replace(".keras","").replace("FINAL",""))
else:
    print("⚠️ Keine Checkpoints gefunden, erstelle neues Modell (Epoch 0).")
    initial_epoch = 0
    model = build_model() # Erstelle neues Modell nur bei Bedarf
 

print(f"▶️ Training startet weiter ab Epoch {initial_epoch}")

# ------------------------------------------------------------
#  ✏️ Klassen: Callbacks
# ------------------------------------------------------------

class CSVLoggerAppend(tf.keras.callbacks.Callback):
    def __init__(self, initial_epoch=0, csv_path="history.csv"):
        super().__init__()
        self.initial_epoch = initial_epoch
        self.csv_path = csv_path

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Keras setzt 'epoch' automatisch auf den richtigen Wert (z.B. 1000),
        # wenn initial_epoch im model.fit übergeben wurde.
        # Wir müssen nur noch +1 rechnen, da Keras bei 0 startet.
        real_epoch = epoch + 1 
        
        # Metriken flexibel abrufen
        train_acc = logs.get("sparse_categorical_accuracy") or logs.get("accuracy") or logs.get("acc")
        val_acc = logs.get("val_sparse_categorical_accuracy") or logs.get("val_accuracy") or logs.get("val_acc")
        train_loss = logs.get("loss")
        val_loss = logs.get("val_loss")

        # -------- Gewichtsnorm ||W||₂ --------
        w_norm = tf.sqrt(
            tf.add_n([tf.reduce_sum(tf.square(w)) for w in self.model.trainable_weights])
        ).numpy()

        # -------- 

        header_exists = os.path.exists(self.csv_path)
        
        # Sicherstellen, dass Datei mit Newline endet vor dem Anhängen
        if header_exists and os.path.getsize(self.csv_path) > 0:
            with open(self.csv_path, "rb+") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) not in (b"\n", b"\r"):
                    f.write(b"\n")

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not header_exists or os.path.getsize(self.csv_path) == 0:
                writer.writerow(["epoch", "train_acc", "test_acc", "train_loss", "test_loss", "W_norm"])
            writer.writerow([real_epoch, train_acc, val_acc, train_loss, val_loss, w_norm])

class KeepLastCheckpoints(tf.keras.callbacks.Callback):
    def __init__(self, initial_epoch, ckpt_dir, save_every=1, keep_last=5):
        super().__init__()
        self.initial_epoch = initial_epoch
        self.ckpt_dir = ckpt_dir
        self.save_every = save_every
        self.keep_last = keep_last

    def on_epoch_end(self, epoch, logs=None):
        #real_epoch = self.initial_epoch + epoch + 1
        logs = logs or {}
        # Keras setzt 'epoch' automatisch auf den richtigen Wert (z.B. 1000),
        # wenn initial_epoch im model.fit übergeben wurde.
        # Wir müssen nur noch +1 rechnen, da Keras bei 0 startet.
        real_epoch = epoch + 1 

        if real_epoch % self.save_every == 0:
            fname = os.path.join(self.ckpt_dir, f"model_epoch_{real_epoch}.keras")  
            self.model.save(fname)
            print(f"\n💾 Checkpoint gespeichert: {fname}")

            # Numerisch korrekt sortiert aufräumen
            ckpts = sorted(glob.glob(os.path.join(self.ckpt_dir, "*.keras")),
                           key=lambda x: int(os.path.basename(x).split('_')[-1].replace('.keras', '').replace('_FINAL','')))

            while len(ckpts) > self.keep_last:
                old = ckpts.pop(0)
                # Sicherheitscheck: Lösche niemals Dateien, die "FINAL" im Namen haben
                if "_FINAL" not in old:
                    os.remove(old)
                    print(f"🗑 Alte Epoche gelöscht: {old}")

class SaveFinalCheckpoint(tf.keras.callbacks.Callback):
    # Nutzt die von Keras berechnete End-Epoche
    def __init__(self, save_dir, config_data): # Umbenannt von ckpt_dir 
                                  # wo die technische Details über das Experiment
                                  # gespeichert werden.
        super().__init__()   
        self.save_dir = save_dir
        self.config_data = config_data
        self.log_fname = os.path.join(self.save_dir, "training_settings.txt")

    def _write_config(self, include_optimizer = False):
        """Hilfsmethode zum Schreiben der Textdatei""" 
        #der Text oben bescrheibt die Funktionalitaet dieser Methode
        with open(self.log_fname, "w") as f:
            f.write("=== EXPERIMENT CONFIGURATION ===\n")
            for key, value in self.config_data.items():
                f.write(f"{key}: {value}\n")

            if include_optimizer and hasattr(self.model, 'optimizer'):
                opt_config = self.model.optimizer.get_config()
                opt_name = self.model.optimizer.__class__.__name__
                f.write("\n=== OPTIMIZER DETAILS (FINAL) ===\n")
                f.write(f"Optimizer Name: {opt_name}\n")
                for key, value in opt_config.items():
                    f.write(f"{key}: {value}\n")

    def on_train_begin(self, logs=None):
        # Wird sofort beim Aufruf von model.fit() ausgeführt
        self._write_config(include_optimizer=False)
        print(f"📝 Experiment-Konfiguration initial gespeichert: {self.log_fname}")       

    def on_train_end(self, logs=None):
         # Wird nur ausgeführt, wenn das Training regulär beendet wird
        final_epoch = self.params['epochs']  

        # 1. Modell speichern in RUN_DIR
        model_fname = os.path.join(self.save_dir, f"model_epoch_{final_epoch}_FINAL.keras")
        self.model.save(model_fname)

        # 2. Log-Datei mit Optimizer-Details aktualisieren
        self._write_config(include_optimizer=True)
        
        print(f"💾 Training beendet. Finale Daten gespeichert in: {self.save_dir}")


# ------------------------------------------------------------
# Epochen-Logik vorbereiten
# ------------------------------------------------------------
# Wir berechnen das Ziel, bevor wir es im Dictionary oder fit() verwenden
TOTAL_TARGET_EPOCHS = initial_epoch + EPOCHS_CONTINUE

# ------------------------------------------------------------
#  Parameter für die Log-Datei
# ------------------------------------------------------------
experiment_params = {
    "P": P, 
    "INITIAL_EPOCHS": initial_epoch,
    "CONTINUE_BY": EPOCHS_CONTINUE,
    "TOTAL_TARGET_EPOCHS": TOTAL_TARGET_EPOCHS, # Jetzt ist die Variable definiert!
    "BATCH_SIZE": BATCH_SIZE, 
    "LR": LR,
    "WEIGHT_DECAY": WEIGHT_DECAY, 
    "SAVE_EVERY": SAVE_EVERY, 
    "KEEP_LAST": KEEP_LAST,
    "train_frac": train_frac,
    "HIDDEN_SIZE": HIDDEN_SIZE,
    "NUM_LAYERS": NUM_LAYERS

}

# ------------------------------------------------------------
# 🏁 Callbacks
# ------------------------------------------------------------

callbacks = [
    CSVLoggerAppend(initial_epoch=initial_epoch, csv_path=CSV_PATH),
    KeepLastCheckpoints(initial_epoch=initial_epoch, ckpt_dir=CKPT_DIR, 
                        save_every=SAVE_EVERY, keep_last=KEEP_LAST),
    SaveFinalCheckpoint(save_dir=RUN_DIR, config_data=experiment_params)
]

# Starte model.fit(...) wie gewohnt



# ------------------------------------------------------------
# 🚀 Weitertrainieren
# ------------------------------------------------------------
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=BATCH_SIZE,
    epochs=TOTAL_TARGET_EPOCHS,
    initial_epoch=initial_epoch,
    verbose=1,
    callbacks=callbacks
)
# ------------------------------------------------------------
# 📈 Plot: alles sauber
# ------------------------------------------------------------
def load_clean_history(csv_path):
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["epoch","train_acc","test_acc","train_loss","test_loss", "w_norm"])
    
    cleaned_rows = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            if len(row) == 5:
                cleaned_rows.append(row)
    df = pd.DataFrame(cleaned_rows, columns=headers)
    for c in headers:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

history_df = load_clean_history(CSV_PATH)



print("Neues Training durchgeführt")
