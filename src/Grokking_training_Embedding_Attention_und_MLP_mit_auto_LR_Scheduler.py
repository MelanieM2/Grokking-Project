"""
Dynamisches Lernraten-Management: Einführung eines Schedulers.
(Erweiterung der mit der Klasse MultiStepGrokking)
-----------------------------------------------------------------------------
Skript ist eine Erweiterung des Grokking_training_Embedding_Attention_und_MLP, 
bei der die `Learning Rate` des Modells um einen automatischen
(als Aufgabe einer spezialisierten Klasse) **Lernraten-Scheduler** ergänzt wurde. 
Dieser dient speziell der Überwindung von Rauschen ab einem bestimmten Wert der Test-Accuracy.

Dieser kann eingeschaltet oder abgeschaltet werden. 
Wenn der Scheduler nicht eingeschaltet wird, wird die Klasse MultiStepGrokkingScheduler 
nicht aufgerufen und dann wird die Skriptsfunktionalität identisch wie jene 
von Grokking_training_Embedding_Attention_und_MLP.
-----------------------------------------------------------------------------
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import datetime
import glob
import pandas as pd

# Überprüfung der Hardware-Ressourcen (Wichtig für das Training neuronaler Netze)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# -----------------------------------------------------------------------------
# Hyperparameter & Experiment-Konfiguration
# -----------------------------------------------------------------------------

# Modulo-Basis für die Arithmetik (Primzahl P)
P = 97 

# Trainings-Dauer und Checkpoint-Intervall
EPOCHS_CONTINUE = 10000     # Anzahl der Epochen für die Fortsetzung oder das Haupttraining
BATCH_SIZE = 128              

# Optimizer-Einstellungen
# Hinweis: AdamW wird oft bevorzugt, da das Grokking-Phänomen stark von der 
# Interaktion zwischen Weight Decay und Lernrate abhängt.
LR = 0.001                  # Lernrate (0.001 ist stabil für AdamW/Transformer-Architekturen)
WEIGHT_DECAY = 0.9          # Entscheidend für Grokking: Zwingt das Modell zu simplen (generalisierbaren) Lösungen

# Speicher-Management für Modelle
SAVE_EVERY = 500            # Speichert alle X Epochen einen Checkpoint
KEEP_LAST = 3               # Behält nur die neuesten Checkpoints, um Speicherplatz zu sparen

# Daten-Splitting
# Ein geringer Anteil an Trainingsdaten (hier 18%) provoziert das Grokking-Verhalten:
# Das Modell muss erst auswendig lernen (Memorization), bevor es die Regel (Generalization) findet.
train_frac = 0.18      

# Modell-Architektur-Parameter
HIDDEN_SIZE = 64            # Breite der Layer (Dimension der internen Repräsentation)
NUM_LAYERS = 2              # Tiefe des Modells



# Auswahl der Architektur: "attention", "mlp", "hybrid"
# "attention": Attention

# "hybrid": Attention + mlp 
#           (eigentlich bezeichnet "transformer" auch diese Architektur in der Literatur)

# "mlp": klassisch Multilayer Perceptron mit Embeddings (ohne Skalierung*).

## *HINWEIS: 
## Das reine "mlp" mit Embeddings (ohne Skalierung) ist experimentell:
## Aufgrund des Fehlens eines Aufmerksamkeitsmechanismus (Attention) benötigt 
## diese Architektur oft eine deutlich höhere Lernrate oder spezifische 
## Initialisierung für das Embedding, um aus der initialen Stagnation (Random Guess) auszubrechen.
##
## Am Anfang des Projekts habe ich mit einem MLP mit Skalierung (ohne Embeddings) Grokking mit
## SGD und anderen Optimizers und Learning Rates untersucht und mein Ergebnis:
## Schnelles Lernen, aber kein echtes Grokking. 
## Der entsprechende Code ist in dem Skript "Grokking_training_baseline_MLP.py" getrennt beigelegt.  

MODEL_ARCH = "hybrid"


# Automatisierter Scheduler für Grokking: Senkt die LR bei Stagnation ab einem Threshold.
# Nutzt die bestehende history.csv zur Synchronisation bei Fortsetzung (Resume).




# --- SCHALTER FÜR DEN SCHEDULER ---
USE_GROKKING_SCHEDULER = True  # Auf True setzen, um den Scheduler aktivieren. 
                                # Dieser wird aktiv, später in Training wenn Stagnierung der Test-Accuracy 
                                # über einen Threshold-Wert.
                                # Danach steuert er die Sinkung der Learning_rate
                                # automatisch unter bestimmten Bedingungen (siehe Dokumentation der
                                # Klasse MultiStepGrokkingScheduler unten) aber nie niedriger als einen
                                # minimalen Wert.

#USE_GROKKING_SCHEDULER = False  # Auf False setzen, um den Scheduler komplett zu deaktivieren.

# Definierte Werte für den Scheduler 
SCHEDULER_THRESHOLD = 0.97
SCHEDULER_DECAY = 0.8
SCHEDULER_PATIENCE = 100 # Sicherer Startwert: Wie viele Epochen warten bis sich die LR ändert
MIN_LR = 1e-6  # Sicherheitsuntergrenze für die Lernrate


#Wenn der Scheduler eingeschaltet ist (USE_GROKKING_SCHEDULER = True): 
# Ab welcher Präzision gilt das Modell als "fertig"?
SCHEDULER_STOP_THRESHOLD = 0.9999  # 1 - epsilon
BREITE_FENSTER = 10 # Das ist die Zahl der Epochen, 
                    # über die die Präzision der Testgenauigkeit 
                    # über dem SCHEDULER_STOP_THRESHOLD bleibt. 
                    # Default = 10.



                    

# -----------------------------------------------------------------------------
# Experiment-Management & Pfad-Konfiguration
# -----------------------------------------------------------------------------

# Option zum Fortsetzen eines abgebrochenen Trainings (Resume-Funktion)
RESUME_RUN = None  # Pfad zu einem existierenden Ordner eingeben, um dort weiterzumachen

#RESUME_RUN = "./runs/grok_P97_hybrid_mit_LR_Scheduler_20260215_162523" #<--Falls der Ordner schon existiert, dann hier
                                        #dessen Pfad eingeben und RESUME_RUN=None oben auskommentieren

# Suffix basierend auf dem Schalter festlegen
sched_suffix = "mit_LR_Scheduler" if USE_GROKKING_SCHEDULER else "ohne_LR_Scheduler"

if RESUME_RUN and os.path.exists(RESUME_RUN):
    RUN_DIR = RESUME_RUN
    print(f"🔄 Setze Training in existierendem Ordner fort: {RUN_DIR}")
else:
    # Erstellung eines eindeutigen Verzeichnisses pro Experiment-Lauf mittels Zeitstempel
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Wir hängen das Suffix (WithSched/NoSched) an den Ordnernamen an
    RUN_DIR = f"./runs/grok_P{P}_{MODEL_ARCH}_{sched_suffix}_{timestamp}"

    os.makedirs(RUN_DIR, exist_ok=True) 
    print(f"📂 Erstelle NEUEN Run-Ordner: {RUN_DIR}")

# Definition der Pfade für Checkpoints und Metriken
CKPT_DIR = os.path.join(RUN_DIR, "checkpoints")
CSV_PATH = os.path.join(RUN_DIR, "history.csv")       # Speichert Loss/Accuracy pro Epoche
INDICES_PATH = os.path.join(RUN_DIR, "train_indices.npy") # Fixiert den Daten-Split

os.makedirs(CKPT_DIR, exist_ok=True)
print(f"✅ Setup abgeschlossen in: {os.path.abspath(RUN_DIR)}")

# -----------------------------------------------------------------------------
# Datengenerierung: Modulare Addition (a + b) mod P
# -----------------------------------------------------------------------------

xs, ys = [], []
for a in range(P):
    for b in range(P):
        xs.append([a, b])
        ys.append((a + b) % P)

# WICHTIG: Für Embedding-Layer nutzen wir reine Ganzzahlen (Integers).
# Eine Skalierung auf [0,1] ist hier kontraproduktiv, da das Modell 
# diskrete Repräsentationen für jede Zahl lernen soll.
xs = np.array(xs, dtype=np.int32) 
ys = np.array(ys, dtype=np.int32)

# -----------------------------------------------------------------------------
# Konsistenter Shuffle & Train-Test-Split
# -----------------------------------------------------------------------------

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
# Modell-Architekturen: Vergleich von MLP, Attention & Hybrid
# -----------------------------------------------------------------------------

def build_mlp_with_embeddings(P, HIDDEN_SIZE, WEIGHT_DECAY):
    """
    Klassisches Multi-Layer Perceptron mit Embedding-Layer.
       
    RESEARCH BEMERKUNG: 
    Ohne die explizite Feature-Interaktion eines Transformers (Attention) fällt es diesem Modell schwer,
    Korrelationen zwischen den diskreten Embeddings von 'a' und 'b' allein über den Flatten-Layer zu lernen. 
    Dies führt oft zu einem "Frozen Model" (beide Train/Test-Accuracies bleiben 
    über das gesamte Training niedrig unter der minimalen Random-Schwelle), 
    sofern keine Skalierung (x/P) oder erhöhte Lernrate genutzt wird.
    """
    model = tf.keras.Sequential([
        # Wandelt diskrete Zahlen in dichte Vektoren um (Lernen der Zahlen-Repräsentation)
        tf.keras.layers.Embedding(input_dim=P, output_dim=HIDDEN_SIZE, input_length=2),
        tf.keras.layers.Flatten(),
        # Tipp: Eine stärkere Initialisierung könnte das 'Frozen Model' Problem beheben:
        # embeddings_initializer="he_normal"
        
        # Hidden Layers: 'tanh' wird oft in theoretischen Grokking-Studien bevorzugt.
        # 'tanh' bietet eine glattere Gradientenlandschaft für theoretische Analysen
        tf.keras.layers.Dense(HIDDEN_SIZE, activation='tanh'), 
        tf.keras.layers.Dense(HIDDEN_SIZE, activation='tanh'), 
        
        # Output: P Klassen (Logits für SparseCategoricalCrossentropy)
        tf.keras.layers.Dense(P)
    ])

    # AdamW (Adam mit Weight Decay) ist essenziell, um Grokking zu induzieren
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )
    return model

def build_tiny_transformer_model(P, HIDDEN_SIZE, WEIGHT_DECAY):
    """Minimaler Transformer zur Nutzung von Self-Attention zwischen Operanden."""
    inputs = tf.keras.layers.Input(shape=(2,))
    x = tf.keras.layers.Embedding(input_dim=P, output_dim=HIDDEN_SIZE)(inputs)
    
    # Attention-Mechanismus berechnet die Relation zwischen Operand a und b
    attn_out = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = tf.keras.layers.Add()([x, attn_out]) # Residual Connection
    x = tf.keras.layers.LayerNormalization()(x)
    
    # Feed-Forward-Netzwerk innerhalb des Transformers
    ffn = tf.keras.layers.Dense(HIDDEN_SIZE, activation='relu')(x)
    x = tf.keras.layers.Add()([x, ffn])
    x = tf.keras.layers.LayerNormalization()(x)
    
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(P)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )
    return model

def build_hybrid_model(P, hidden_size, weight_decay):
    """
    Hybrid-Ansatz: Attention für Feature-Interaktion, 
    gefolgt von MLP-Kopf für die Klassifizierung.
    """
    inputs = tf.keras.layers.Input(shape=(2,), dtype=tf.int32)
    x = tf.keras.layers.Embedding(input_dim=P, output_dim=hidden_size, name="embedding")(inputs)
    
    # 1. Interaktions-Phase: Attention erzwingt das Erlernen mathematischer Beziehungen
    attn_out = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=hidden_size//4)(x, x)
    x = tf.keras.layers.Add()([x, attn_out])
    x = tf.keras.layers.LayerNormalization()(x)
    
    # 2. Verarbeitungs-Phase: Flatten ermöglicht das Lernen positionsabhängiger Features
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hidden_size, activation='tanh')(x)
    x = tf.keras.layers.Dense(hidden_size, activation='tanh')(x)
    
    outputs = tf.keras.layers.Dense(P, name="output_logits")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Hybrid_Grok_Model")
    
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=LR, weight_decay=weight_decay),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )
    return model


# -----------------------------------------------------------------------------
# Modell-Instanziierung & Architektur-Auswahl
# -----------------------------------------------------------------------------

def get_model(arch_name):
    """Zentrale Factory-Funktion zur Modellauswahl."""
    match arch_name:
        case "attention": 
            return build_tiny_transformer_model(P, HIDDEN_SIZE, WEIGHT_DECAY)
        case "mlp": #<-- so wie gerade steht (ohne Skalierung, mit Embedding), 
                    #reproduziert mlp allein nicht nur kein Grokking
                    #sondern auch kein Standard Training, ohne eine andere Auswahl 
                    #der LR oder "Tricks" beim Embedding
            return build_mlp_with_embeddings(P, HIDDEN_SIZE, WEIGHT_DECAY)  
        case "hybrid":
            return build_hybrid_model(P, HIDDEN_SIZE, WEIGHT_DECAY)
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
    
    # Laden des exakten Modellzustands inklusive Optimizer-Status (AdamW-Momente)
    model = tf.keras.models.load_model(latest) 

    ## --- MANUELLE LR-KONTROLLE BEIM RESUME ---
    #target_resume_lr = 1e-4  # Hier kannst man den Wunschwert setzen nach manuller Forsetzung
                              # eines vorherigen unterbrochenen Runs mit variablen LR
    #current_ckpt_lr = model.optimizer.learning_rate.numpy()

    #if current_ckpt_lr < target_resume_lr:
    #    print(f"🔄 Manuelle Korrektur: Hebe LR von {current_ckpt_lr:.2e} auf {target_resume_lr:.2e} an.")
    #    model.optimizer.learning_rate.assign(target_resume_lr)
    #else:
    #    print(f"✅ Behalte aktuelle LR bei: {current_ckpt_lr:.2e}")



    
    # Extraktion der letzten Epoche aus dem Dateinamen
    clean_name = os.path.basename(latest).replace('_FINAL', '')
    initial_epoch = int(clean_name.split("_")[-1].replace(".keras", ""))
else:
    print("⚠️ Keine Checkpoints gefunden. Initialisiere neues Modell für Epoche 0.")
    initial_epoch = 0
    # Das Modell wurde oben bereits durch model = get_model(...) initialisiert

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
                
                # Namens-Logik: Priorität auf lesbares "AdamW" für den Plotting-Skript
                raw_name = getattr(opt, 'name', opt_config.get('name', opt.__class__.__name__))
                opt_name = str(raw_name).replace('adamw', 'AdamW').capitalize()

                f.write("\n=== OPTIMIZER DETAILS (CURRENT) ===\n")
                # Wichtig: "Optimizer Name" nutzen, damit der Plotter den Key findet
                f.write(f"Optimizer Name: {opt_name}\n")
                
                # Alle weiteren Config-Parameter (LR, Decay, etc.)
                for key, value in opt_config.items():
                    if key != 'name':  # Überspringt das kleine 'adamw'
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
                # Und wir stellen sicher, dass AdamW schön formatiert ist.
                raw_name = getattr(opt, 'name', opt_config.get('name', 'AdamW'))
                opt_name = str(raw_name).replace('adamw', 'AdamW').capitalize()


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
# Klassen: Scheduler
# -----------------------------------------------------------------------------

class MultiStepGrokkingScheduler(tf.keras.callbacks.Callback):
    """
    DYNAMISCHES LERNRATEN-MANAGEMENT (MultiStepGrokking)
    ---------------------------------------------------
    Zweck: Optimierung der Generalisierung (Grokking) durch adaptive LR-Senkung 
    bei Plateaus und Sicherstellung einer stabilen Endphase.

    MECHANISMUS:
    - Trigger: Aktivierung der Stagnations-Prüfung erst ab einem THRESHOLD (z.B. 0.92).
    - Patience: Senkung der LR nach genau X Epochen ohne neuen Bestwert (1 Unit = 1 Epoche).
    - Decay: Multipliziert die LR bei Stagnation mit einem Faktor (z.B. 0.8).

    STABILITÄT & ABSCHALTUNG (NEU):
    - Stop-Threshold: Definiert die Ziel-Präzision (z.B. 0.9999).
    - Moving Window: Prüft die Stabilität über ein definiertes Fenster (breite_fenster).
    - Auto-Off: Deaktiviert den Scheduler permanent, sobald das Ziel stabil erreicht wurde,
      um unnötige LR-Senkungen in der perfekten Finalphase zu verhindern.

    FEATURES & SICHERHEIT:
    - Resume-Sync: Lädt den bisherigen Bestwert automatisch aus der 'history.csv'.
    - Sicherheitsnetz: MIN_LR (z.B. 1e-6) verhindert ein numerisches Einfrieren des Modells.
    - Logging: Dokumentiert den Zeitpunkt der stabilen Generalisierung in der .txt Datei.

    Strategie: Hohe LR für initiale Exploration, niedrige LR für präzises Fine-Tuning 
    und automatische Deaktivierung nach erfolgreicher Stabilisierung.
    ---------------------------------------------------
    """
   
    def __init__(self, threshold=0.90, decay_rate=0.5, stop_threshold = 0.9999, breite_fenster=10,
                 patience=10, min_lr=1e-7, initial_epoch=0, csv_path="history.csv", run_dir="."):
        super().__init__()
        self.threshold = threshold
        self.stop_threshold = stop_threshold
        self.decay_rate = decay_rate
        self.patience = patience
        self.min_lr = min_lr # <--- Speichert das Sicherheitsnetz
        self.initial_epoch = initial_epoch
        self.csv_path = csv_path

        self.run_dir = run_dir
        # Pfad zur Dokumentationsdatei (wie in KeepLastCheckpoints)
        self.log_fname = os.path.join(self.run_dir, "training_settings.txt")
        
        # Internes Gedächtnis
        self.best_acc = 0.0
        self.wait = 0
        self.acc_window = []  # Ein Fenster für die Stabilitäts-Prüfung,
                              # wenn die Konvergence nährst sich von Grokking,
                              # aber noch nicht wirklich da ist...
        self.breite_fenster = breite_fenster
        self.is_finished = False # Schalter: Hat das Modell das Ziel stabil erreicht?

    def on_train_begin(self, logs=None):
        """
        Synchronisiert den Scheduler mit dem bisherigen Verlauf.
        Die LR wird automatisch durch load_model() im Optimizer wiederhergestellt.
        Wir müssen hier nur den Rekordwert (best_acc) laden.
        """
        if self.initial_epoch > 0 and os.path.exists(self.csv_path):
            try:
                df = pd.read_csv(self.csv_path)
                # Wir betrachten nur Daten bis zum Resume-Punkt
                past_data = df[df['epoch'] <= self.initial_epoch]
                if not past_data.empty:
                    # Wir nutzen 'test_acc', da dies dein Spaltenname im CSVLoggerAppend ist
                    self.best_acc = past_data['test_acc'].max()
                    print(f"📈 Scheduler-Sync: Best-Acc von {self.best_acc:.4f} aus CSV geladen.")
            except Exception as e:
                print(f"⚠️ Scheduler-Sync Hinweis: Best-Acc konnte nicht geladen werden ({e}).")

    def on_epoch_end(self, epoch, logs=None):
        if self.is_finished:
            return # Scheduler tut nichts mehr, Modell ist im "Ziel-Modus"
        
        logs = logs or {}
        # Suche nach der Validierungs-Genauigkeit (kompatibel mit den Keys)
        test_acc = next((logs.get(k) for k in ["val_accuracy", "val_sparse_categorical_accuracy", "val_acc"] if k in logs), 0)

        
        # 1. STABILITÄTS-CHECK: Wir sammeln die letzten 10 Werte in dem Liste-Fenster
        self.acc_window.append(test_acc)
        if len(self.acc_window) > self.breite_fenster: # wenn die Breite des Fenster überholt wird.
                                      # Die Bedingung sorgt dafür, dass
                                      # die Liste nie mehr als breite_fenster=10 Einträge hat. 
                                      # Oder in anderen Worter: das ist ein "Moving Window“ 
            self.acc_window.pop(0)  


        # Durchschnitt der letzten breite_fenster = 10 Epochen berechnen
        avg_acc = sum(self.acc_window) / len(self.acc_window)
        
        #--Live-Anzeige für dich in der Konsole ---
        if test_acc >= self.stop_threshold:
            current_count = len(self.acc_window)
            # Zeigt z.B. an: ✨ Ziel erreicht! Stabilitäts-Check: 4/10
            print(f"Ziel erreicht! Stabilitäts-Check: {current_count}/{self.breite_fenster} (Avg: {avg_acc:.4f})")

        # 2. ABSCHALTUNG: Nur wenn der DURCHSCHNITT stabil oben bleibt
        if len(self.acc_window) == self.breite_fenster and avg_acc >= self.stop_threshold:
            self.is_finished = True

            ## --- DIESE ZEILE STOPPT DAS GESAMTE TRAINING ---
            # self.model.stop_training = True # <--Training sofort aufhört, 
                                              # sobald die Stabilität (10 Epochen bei 1.0) erreicht ist,
                                              # hier ist nicht notwendig, weil das immer manuell 
                                              # deaktiviert werden kann.
            real_epoch = epoch + 1
            
            # Dokumentation in .txt
            if os.path.exists(self.log_fname):
                with open(self.log_fname, "a") as f:
                    f.write(f"STABILE GENERALISIERUNG: Ziel bei Epoche {real_epoch} fixiert (Avg Acc: {avg_acc:.4f}).\n")
                    f.write(f" Scheduler wurde deaktiviert. GRokking läuft stabil über die letzte {self.breite_fenster} Epochen aus.\n")
            print(f"\n✨ Modell stabilisiert (Avg: {avg_acc:.4f}). Scheduler deaktiviert.")
            return

        

        # 3. Scheduler senkt LR bei Stagnation
        #    Diese Bedingung greift erst, wenn der Grokking-Threshold erreicht wurde
        if test_acc >= self.threshold:
            # Check auf Verbesserung gegenüber dem bisherigen Maximum.
            # Hier nutzen wir weiterhin die normale test_acc für den Bestwert
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.wait = 0  # Fortschritt! Geduld wird zurückgesetzt
            else:
                self.wait += 1 # Stagnation: Zähler (oder Epoche) erhöhen
                
            # Trigger: Wenn die Geduld (Patience) abgelaufen ist
            if self.wait >= self.patience:
                if hasattr(self.model, 'optimizer'):
                    opt = self.model.optimizer
                    current_lr = opt.learning_rate.numpy()

                    # --- Sicherheitscheck: Nur senken, wenn wir über dem Minimum sind ---
                    if current_lr > self.min_lr:
                        # Hier wird die neue Lernrate gerechnet
                        new_lr = max(current_lr * self.decay_rate, self.min_lr) 

                        # Hier setzen wir die Lernrate im Optimizer ein
                        opt.learning_rate.assign(new_lr)

                        # Geduld wird auch zurückgesetzt
                        self.wait = 0
                        print(f"\n Grokking-Trigger (Epoche {epoch+1}):")
                        print(f"\n📉 Stagnation erkannt. LR gesenkt: {current_lr:.6e} -> {new_lr:.6e}")
                    else:
                        # Fall: MIN_LR schon erreicht
                        self.wait = 0 # Wir setzen wait trotzdem zurück, 
                                      # um nicht bei jeder Epoche diese Meldung als Spam zu haben
                        print(f"\nℹ️ Stagnation bei {test_acc:.4f} erkannt, aber MIN_LR ({self.min_lr:.1e}) bereits erreicht. Keine weitere Senkung.")
                    
                    
                    # Hinweis: Die KeepLastCheckpoints-Klasse schreibt diese neue LR 
                    # beim nächsten Speichern automatisch in die training_settings.txt




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
    "MODEL_TYPE": MODEL_ARCH.capitalize(),    # Formatiert: "Attention", "mlp" oder "Hybrid"
    "BATCH_SIZE": BATCH_SIZE, 
    "LR": LR,                                # Lernrate
    "WEIGHT_DECAY": WEIGHT_DECAY,            # Entscheidend für die Regularisierung (Grokking)
    "SAVE_EVERY": SAVE_EVERY, 
    "KEEP_LAST": KEEP_LAST,                  # Anzahl rotierender Checkpoints
    "train_frac": train_frac,                # Anteil der genutzten Trainingsdaten
    "HIDDEN_SIZE": HIDDEN_SIZE,              # Breite des Modells
    "NUM_LAYERS": NUM_LAYERS,                 # Tiefe des Modells

    # --- VERVOLLSTÄNDIGTE Scheduler-Parameter ---
    "USE_SCHEDULER": USE_GROKKING_SCHEDULER,
    "SCHEDULER_THRESHOLD": SCHEDULER_THRESHOLD,      # Wann geht es los? (z.B. 0.92)
    "SCHEDULER_STOP_THRESHOLD": SCHEDULER_STOP_THRESHOLD, # Wann ist Ziel erreicht? (z.B. 0.9999)
    "SCHEDULER_DECAY_RATE": SCHEDULER_DECAY,
    "SCHEDULER_PATIENCE": SCHEDULER_PATIENCE,
    "MIN_LR": MIN_LR,                                # Sicherheitsnetz
    "BREITE_FENSTER": BREITE_FENSTER                 # Stabilitäts-Check
}

print(f"📈 Experiment-Setup: Ziel ist Epoche {TOTAL_TARGET_EPOCHS}")



# -----------------------------------------------------------------------------
# Callback-Integration
# -----------------------------------------------------------------------------
# Zusammenführung der spezialisierten Callbacks-Komponenten

# Wir starten mit den Standard-Callbacks
callbacks = [
    # Custom Logging für Metriken und Gewichtsnorm
    CSVLoggerAppend(initial_epoch=initial_epoch, csv_path=CSV_PATH),

    # Automatisches Checkpointing & Live-Parameter-Update
    KeepLastCheckpoints(initial_epoch=initial_epoch,
                         ckpt_dir=CKPT_DIR, 
                         run_dir=RUN_DIR, 
                         config_data=experiment_params,
                         save_every=SAVE_EVERY, 
                         keep_last=KEEP_LAST),
    
    # Finale Sicherung des Modells und der Metadaten bei regulärem Ende
    SaveFinalCheckpoint(save_dir=RUN_DIR, config_data=experiment_params)
]


# NUR wenn der Schalter oben auf True steht, fügen wir den Scheduler AM ANFANG hinzu
if USE_GROKKING_SCHEDULER:
    grok_scheduler = MultiStepGrokkingScheduler(
        threshold=SCHEDULER_THRESHOLD,
        stop_threshold=SCHEDULER_STOP_THRESHOLD, 
        breite_fenster=BREITE_FENSTER,
        decay_rate=SCHEDULER_DECAY, 
        patience=SCHEDULER_PATIENCE, 
        min_lr=MIN_LR,
        initial_epoch=initial_epoch, 
        csv_path=CSV_PATH,
        run_dir=RUN_DIR
    )
    # .insert(0, ...) setzt ihn an die erste Stelle der Liste der Callbacks oben
    callbacks.insert(0, grok_scheduler)
    print("🚀 Grokking-Scheduler ist AKTIVIERT.")
else:
    print("ℹ️ Grokking-Scheduler ist DEAKTIVIERT (Konstante Lernrate).")





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