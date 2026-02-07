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
EPOCHS_CONTINUE = 1500     # wie lange wir noch *weiter* trainieren es sei den wir beginnen von anfang an
BATCH_SIZE = 64              
LR = 0.001               #0.05 für SGD, aber = 0.001 für Transformer & AdamW (Viel stabiler )
LEARNING_INTERVAL = 500 #Für die Scheduler
DECAY_RATE = 0.5          #Für die Scheduler             
WEIGHT_DECAY = 0.8    # kommt für AdamW optimzer, nicht für SGD, 
SAVE_EVERY = 100
KEEP_LAST = 3
train_frac = 0.18      ## 35 % der Daten nutzen (bewährter Wert aus der Forschung) P=29

HIDDEN_SIZE = 64        # ich glaube mit 64 ist das hybird Modell zu mächtig...
NUM_LAYERS = 2

#OPTIONS: "transformer", "mlp", "hybrid"
MODEL_ARCH = "hybrid" 

# ------------------------------------------------------------
# 📁 EXISTIERENDER Run-Ordner (NICHT ändern!)
# ------------------------------------------------------------
# ------------------------------------------------------------
# 📂 Run-Verzeichnis wählen
# ------------------------------------------------------------
# FALL A: Wir wollen einen BESTEHENDEN Run fortsetzen:
# Geben wir hier den Ordnernamen manuell ein, wenn wir weitermachen wollen:

RESUME_RUN = None  # Auf None lassen für einen ganz NEUEN Run

#RESUME_RUN = "./runs/grok_P29_20260127_140846" 

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
# Da wir jetzt Embeddings nutzen, dürfen wir die Daten nicht mehr skalieren. 
# Das modell braucht die reinen Ganzzahlen. Diese 2 Zeilen gelten nicht mehr 
#xs = np.array(xs, dtype=np.float32) / (P - 1)
#ys = np.array(ys, dtype=np.int32)

xs = np.array(xs, dtype=np.int32) # Reine Ganzzahlen 0, 1, 2...
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

# Berechnung der Schritten (Steps) pro Epoche
steps_per_epoch = len(x_train) // BATCH_SIZE

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=LR,
    decay_steps = LEARNING_INTERVAL * steps_per_epoch, # zB Alle 5000 Epochen...
    decay_rate = DECAY_RATE,              # ...halbiere die Lernrate wenn DECAY_RATE = 0.5
                                         
                                         # neue_LR = LR * decay_rate^(step/decay_steps)
                                         # hier step ist die Laufvariable: 
                                         # der globale Trainingsschritt 
                                         # (ein Zähler, der bei jedem verarbeiteten Batch um 1 steigt).

                                         # Wenn die Laufvariable genau den Wert von decay_steps
                                         # erreicht (learning_interval * steps_per_epoch) ist 
                                         # der Exponent genau 1 und die neue Lernrate genau das halbe
                                         # LR*0.5^1 wenn DECAY_RATE=0.5 ist.

    staircase = True  # <--- Das ist die entscheidende Änderung:
                       # Der Wert im Exponenten bleibt so lange 
                       # bei einer ganzen Zahl (z. B. 0), bis 
                       # die vollen decay_steps (zB 5000 Epochen) 
                       # erreicht sind. Erst in diesem Moment 
                       # springt der Exponent auf 1, 
                       # und die Lernrate halbiert sich schlagartig.
)


# ------------------------------------------------------------
# Modell-Definition
# ------------------------------------------------------------
# 

def build_mlp_with_embeddings(P, HIDDEN_SIZE, WEIGHT_DECAY):
    model = tf.keras.Sequential([
        # 1. Embedding: Macht aus [a, b] -> [Vektor_a, Vektor_b]
        # Input_dim=P (29 Zahlen), output_dim=HIDDEN_SIZE (64)

        tf.keras.layers.Embedding(input_dim=P, output_dim=HIDDEN_SIZE, input_length=2),
        
        # 2. Flatten: Macht aus den zwei Vektoren eine lange Liste (128 Werte)
        tf.keras.layers.Flatten(),

        #unsere Schichten wie gewohnt bei Dense MLP
        # First Hidden Layer 
        tf.keras.layers.Dense(HIDDEN_SIZE, activation='tanh'), # hier haben wir die Regularisierung der Kernel aufgegeben  
                                                               # kernel_regularizer=tf.keras.regularizers.l2(1e-4)

        # Second Hidden Layer
        tf.keras.layers.Dense(HIDDEN_SIZE, activation='tanh'), # hier haben wir die Regularisierung der Kernel aufgegeben  
                                                               # kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        
        # Output Layer (P Klassen)
        tf.keras.layers.Dense(P)
    ])

    #Probieren wir wieder mit AdamW...am ersten hat das nicht ohne Embedding functioniert, 
    #vielleicht doch mit Embedding. Mit SGD hat das wirklich nicht funktioniert
    model.compile(
            optimizer = tf.keras.optimizers.AdamW(
            #learning_rate =LR, #<--ohne lr_schedule
            learning_rate = lr_schedule,  #<--mit lr_schedule 
            weight_decay = WEIGHT_DECAY  # <--Der "Grokking-Motor"...angeblich...
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    
    return model



#. Der Tiny Transformer (Keras Functional API)
# Er nutzt Attention, was wie ein "Suchscheinwerfer" funktioniert, 
# um die Beziehung zwischen Zahl \(a\) und Zahl \(b\) direkt zu verknüpfen.
#  Er "grokkt" oft schneller als ein MLP.
def build_tiny_transformer_model(P, HIDDEN_SIZE, WEIGHT_DECAY):
    inputs = tf.keras.layers.Input(shape=(2,))
    
    # Embedding (wie beim MLP)
    x = tf.keras.layers.Embedding(input_dim=P, output_dim=HIDDEN_SIZE)(inputs)
    
    # Minimaler Transformer Block
    attn_out = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = tf.keras.layers.Add()([x, attn_out])
    x = tf.keras.layers.LayerNormalization()(x)
    
    # Feed Forward Teil
    ffn = tf.keras.layers.Dense(HIDDEN_SIZE, activation='relu')(x)
    x = tf.keras.layers.Add()([x, ffn])
    x = tf.keras.layers.LayerNormalization()(x)
    
    # Reduktion auf einen Vektor und Output
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(P)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]##############################
    )
    return model


#HYBRIDES MODELL aufbauen:
# Input [a, b] → Embedding → Transformer-Layer → MLP-Layer → Output. 
# Sinnhaftigkeit: Das wäre ein sehr starkes Modell. 
# Der Transformer berechnet die Interaktion zwischen 
# zwei Zahlen a und b, und das MLP "interpretiert" das Ergebnis.Gefahr: 
# Da das Problem (mod 29) mathematisch sehr "klein" ist, 
# könnte ein kombiniertes Modell so mächtig sein, 
# dass es die Lösung sofort findet oder einfach auswendig lernt, 
# ohne dass wir die schöne "Grokking-Kurve" (den plötzlichen Sprung nach langer Zeit) sehen.

def build_hybrid_model(P, hidden_size, weight_decay):
    """
    Kombiniert Transformer-Attention mit einem MLP-Kopf.
    Optimiert für Grokking-Experimente auf CPU/GPU.
    """
    inputs = tf.keras.layers.Input(shape=(2,), dtype=tf.int32)
    
    # 1. EMBEDDING LAYER
    # Erzeugt Vektoren für die Zahlen a und b
    x = tf.keras.layers.Embedding(input_dim=P, output_dim=hidden_size, name="embedding")(inputs)
    
    # 2. TRANSFORMER BLOCK (Interaktion)
    # Berechnet, wie a und b mathematisch zusammenhängen
    attn_out = tf.keras.layers.MultiHeadAttention(
        num_heads=4, 
        key_dim=hidden_size // 4,
        name="attention"
    )(x, x)
    
    # Skip Connection & Layer Norm
    x = tf.keras.layers.Add()([x, attn_out])
    x = tf.keras.layers.LayerNormalization(name="norm_1")(x)
    
    # 3. MLP TEIL (Verarbeitung)
    # Wir flachen die zwei Vektoren zu einem langen Vektor ab
    x = tf.keras.layers.Flatten(name="flatten")(x)
    
    # Deine bewährte MLP-Struktur (tanh ist gut für Grokking-Analysen)
    x = tf.keras.layers.Dense(hidden_size, activation='tanh', name="mlp_dense_1")(x)
    x = tf.keras.layers.Dense(hidden_size, activation='tanh', name="mlp_dense_2")(x)
    
    # 4. OUTPUT LAYER
    # Logits für P Klassen (0 bis P-1)
    outputs = tf.keras.layers.Dense(P, name="output_logits")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Hybrid_Grok_Model")
    
    # OPTIMIZER (AdamW ist der Motor für Grokking)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule, 
            weight_decay=weight_decay
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # WICHTIG: Name auf "accuracy" fixieren für den CSV-Logger
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )
    
    return model
# Warum dieses Hybrid-Modell im Vergleich besonders ist: 
# 1.- Explizite Interaktion: Während ein reines MLP die Embeddings von a und b 
# oft nur nebeneinanderlegt (Flatten), zwingt die MultiHeadAttention das Modell dazu, 
# Beziehungen zwischen den beiden Zahlen zu berechnen, bevor sie ins MLP gehen. 
# Das führt oft zu einem "saubereren" Grokking.

# 2.-Flatten vs. Pooling: In diesem Hybrid-Modell nutzen wir Flatten(). 
# Das bedeutet, das Modell kann einen Unterschied machen, 
# ob eine Zahl an Position a oder Position b steht. Da a+b kommutativ ist, 
# wird das Modell lernen, beide Positionen gleich zu behandeln 
# (ein zusätzlicher interessanter Lernschritt für das Grokking).

# 3.-Metrik-Namen: Ich habe name="accuracy" hinzugefügt, 
# damit dein CSVLoggerAppend nicht wieder NaN oder None in die Datei schreibt, 
# weil er den Schlüssel nicht findet.


# ------------------------------------------------------------
# Modell Auswahl:
#-------------------------------------------------------------

# Options: "transformer", "mlp", "hybrid"
MODEL_ARCH = "hybrid" 


match MODEL_ARCH:
    case "transformer":
        model = build_tiny_transformer_model(P, HIDDEN_SIZE, WEIGHT_DECAY)
    case "mlp":
        model = build_mlp_with_embeddings(P, HIDDEN_SIZE, WEIGHT_DECAY)
    case "hybrid":
        model = build_hybrid_model(P, HIDDEN_SIZE, WEIGHT_DECAY) 
    case _:
        raise ValueError(f"Unknown model type: {MODEL_ARCH}") 
        #  acts as a safety net. If you accidentally type MODEL_ARCH = "tranzformer", 
        #  the code will throw an error immediately rather than defaulting to the wrong model.

model.summary()



# ------------------------------------------------------------
# 📦 Resume: Checkpoint wiederherstellen
# ------------------------------------------------------------
# WICHTIG: Hier entscheiden wir, ob wir neu bauen ODER laden
checkpoints = sorted(glob.glob(os.path.join(CKPT_DIR, "*.keras")),
                     key=lambda x: int(os.path.basename(x).replace('_FINAL', '').split('_')[-1].replace('.keras', '')))

if checkpoints:
    latest = checkpoints[-1]
    print(f"🔄 Lade letzten Checkpoint: {latest}")
    model = tf.keras.models.load_model(latest) # Lade existierendes Modell

    # Wir entfernen erst '_FINAL', dann bleibt am Ende immer die Zahl + .keras
    clean_name = os.path.basename(latest).replace('_FINAL', '')
    initial_epoch = int(clean_name.split("_")[-1].replace(".keras",""))
else:
    print("⚠️ Keine Checkpoints gefunden, erstelle neues Modell (Epoch 0).")
    initial_epoch = 0

    # Hier nutzen wir die Entscheidung von oben:
    match MODEL_ARCH:
        case "transformer":
            model = build_tiny_transformer_model(P, HIDDEN_SIZE, WEIGHT_DECAY)
        case "mlp":
            model = build_mlp_with_embeddings(P, HIDDEN_SIZE, WEIGHT_DECAY)
        case "hybrid":
            model = build_hybrid_model(P, HIDDEN_SIZE, WEIGHT_DECAY)
     

print(f"▶️ Training startet/geht weiter ab Epoch {initial_epoch}")



    


# ------------------------------------------------------------
#  ✏️ Klassen: Callbacks
# ------------------------------------------------------------

class CSVLoggerAppend(tf.keras.callbacks.Callback):
    def __init__(self, initial_epoch=0, csv_path="history.csv"):
        super().__init__()
        self.initial_epoch = initial_epoch
        self.csv_path = csv_path

    # Um die extra Zeilen in .csv zu löschen nach ungewollter Unterbrechung
    def on_train_begin(self, logs=None):
        """
        Wird beim Start von model.fit() aufgerufen. 
        Löscht alle Zeilen in der CSV, die hinter der initial_epoch liegen.
        """
        if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
            try:
                # Wir laden die CSV
                df = pd.read_csv(self.csv_path)
                
                # Wir behalten nur die Zeilen, deren Epoche <= initial_epoch ist
                # Annahme: Die Spalte in .csv heißt "epoch"!
                df_cleaned = df[df['epoch'] <= self.initial_epoch]
                
                # Überschreiben der Datei mit den bereinigten Daten
                df_cleaned.to_csv(self.csv_path, index=False)
                
                print(f"🧹 CSV bereinigt: Nur Daten bis Epoche {self.initial_epoch} wurden behalten.")
            except Exception as e:
                print(f"⚠️ Warnung beim Bereinigen der CSV: {e}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}  
        # NUR ZUM TESTEN: Zeigt dir alle verfügbaren Namen in der Konsole
        #print(f"DEBUG: Verfügbare Logs: {logs.keys()}") 

        # or if logs is None:
        #        logs = {}
        # Keras setzt 'epoch' automatisch auf den richtigen Wert (z.B. 1000),
        # wenn initial_epoch im model.fit übergeben wurde.
        # Wir müssen nur noch +1 rechnen, da Keras bei 0 startet.
        real_epoch = epoch + 1 
        
        train_acc = None
        for key in ["accuracy", "sparse_categorical_accuracy", "acc"]:
            if key in logs:
                train_acc = logs[key]
                break

        val_acc = None
        for key in ["val_accuracy", "val_sparse_categorical_accuracy", "val_acc"]:
            if key in logs:
                val_acc = logs[key]
                break

        train_loss = logs.get("loss")
        val_loss = logs.get("val_loss")

        # Gewichtsnorm Berechnung bleibt gleich...
        w_norm = tf.sqrt(tf.add_n([tf.reduce_sum(tf.square(w)) for w in self.model.trainable_weights])).numpy()

        # Schreiben in die CSV
        header_exists = os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not header_exists or os.path.getsize(self.csv_path) == 0:
                writer.writerow(["epoch", "train_acc", "test_acc", "train_loss", "test_loss", "W_norm"])
            # Hier werden nun train_acc (0.0) und val_acc (0.0) korrekt geschrieben
            writer.writerow([real_epoch, train_acc, val_acc, train_loss, val_loss, w_norm])

        #----------------------------unten---debuggen
        # # Metriken flexibel abrufen
        # train_acc = logs.get("sparse_categorical_accuracy") or logs.get("accuracy") or logs.get("acc")
        # val_acc = logs.get("val_sparse_categorical_accuracy") or logs.get("val_accuracy") or logs.get("val_acc")
        # train_loss = logs.get("loss")
        # val_loss = logs.get("val_loss")

        # # -------- Gewichtsnorm ||W||₂ --------
        # w_norm = tf.sqrt(
        #     tf.add_n([tf.reduce_sum(tf.square(w)) for w in self.model.trainable_weights])
        # ).numpy()

        # # -------- 

        # header_exists = os.path.exists(self.csv_path)
        
        # # Sicherstellen, dass Datei mit Newline endet vor dem Anhängen
        # if header_exists and os.path.getsize(self.csv_path) > 0:
        #     with open(self.csv_path, "rb+") as f:
        #         f.seek(-1, os.SEEK_END)
        #         if f.read(1) not in (b"\n", b"\r"):
        #             f.write(b"\n")

        # with open(self.csv_path, "a", newline="") as f:
        #     writer = csv.writer(f)
        #     if not header_exists or os.path.getsize(self.csv_path) == 0:
        #         writer.writerow(["epoch", "train_acc", "test_acc", "train_loss", "test_loss", "W_norm"])
        #     writer.writerow([real_epoch, train_acc, val_acc, train_loss, val_loss, w_norm])



class KeepLastCheckpoints(tf.keras.callbacks.Callback):
    def __init__(self, initial_epoch, ckpt_dir, run_dir, config_data, save_every=1, keep_last=5):
        super().__init__()
        self.initial_epoch = initial_epoch
        self.ckpt_dir = ckpt_dir
        self.run_dir = run_dir #<--neu
        self.config_data = config_data #<--
        self.save_every = save_every
        self.keep_last = keep_last
        # Pfad zur Textdatei im Hauptordner
        self.log_fname = os.path.join(self.run_dir, "training_settings.txt")

    def _write_current_status(self, epoch):
        """Aktualisiert die .txt Datei sofort mit Optimizer-Daten"""
        with open(self.log_fname, "w") as f:
            f.write("=== EXPERIMENT CONFIGURATION (LIVE UPDATE) ===\n")
            f.write(f"Zuletzt aktualisiert bei Epoche: {epoch}\n\n")
            for key, value in self.config_data.items():
                f.write(f"{key}: {value}\n")

            if hasattr(self.model, 'optimizer'):
                opt_config = self.model.optimizer.get_config()
                f.write("\n=== OPTIMIZER DETAILS (CURRENT) ===\n")
                f.write(f"Optimizer: {self.model.optimizer.__class__.__name__}\n")
                for key, value in opt_config.items():
                    f.write(f"{key}: {value}\n")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Nicht unbedingt nötig (aber schadet nicht), weil dies in der Klasse nicht benutz wrid. 
        # Wenn wir uns später entscheiden, z. B. den Dateinamen des Checkpoints 
        # vom aktuellen Loss abhängig zu machen (z. B. model_val_loss_0.04.keras), 
        # bräuchtest wir die logs wieder.

        # Keras setzt 'epoch' automatisch auf den richtigen Wert (z.B. 1000),
        # wenn initial_epoch im model.fit übergeben wurde.
        # Wir müssen nur noch +1 rechnen, da Keras bei 0 startet.
        real_epoch = epoch + 1 

        if real_epoch % self.save_every == 0:
            fname = os.path.join(self.ckpt_dir, f"model_epoch_{real_epoch}.keras")  
            self.model.save(fname)

            # --- NEU: Sofortige Aktualisierung der .txt ---
            self._write_current_status(real_epoch)

            print(f"\n💾 Checkpoint & Setting Info gespeichert: Epoche {real_epoch} in {fname}")

            # Numerisch korrekt sortiert aufräumen
            ckpts = sorted(glob.glob(os.path.join(self.ckpt_dir, "*.keras")),
                           key=lambda x: int(os.path.basename(x).replace('_FINAL', '').split('_')[-1].replace('.keras', '')))

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
    "MODEL_TYPE": MODEL_ARCH.capitalize(), # Automatically becomes "Transformer", "Mlp", or "Hybrid",
    "BATCH_SIZE": BATCH_SIZE, 
    "LR": LR,
    "LEARNING_INTERVAL" : LEARNING_INTERVAL, #Für die Scheduler
    "DECAY_RATE" : DECAY_RATE,
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
    KeepLastCheckpoints(initial_epoch=initial_epoch, 
                        ckpt_dir=CKPT_DIR, 
                        run_dir=RUN_DIR,              # NEU
                        config_data=experiment_params, # NEU
                        save_every=SAVE_EVERY, 
                        keep_last=KEEP_LAST),
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
    verbose=2,  #1, wenn man kurz testen möchte, ob die geschwindigkeit pro batch stimmt.
                #2, für ein langes training
                #0, wenn man parallel einen eigenen Logger 
                #   (wie hier CSVLoggerAppend) nutzt, der uns
                #   sowieso eigene Nachrichten in die Konsole schreibt.
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