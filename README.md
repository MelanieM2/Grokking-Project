# Grokking Experiment: Visualisierung des Lernens kleiner neuronaler Netze bei modularer Addition

Dieses Projekt untersucht das Phänomen des **"Grokking"** – jenen Moment, in dem ein neuronales Netz von reinem Auswendiglernen (Memorization) zu echter mathematischer Generalisierung übergeht. 

Das Projekt entstand als Abschlussprojekt im Rahmen einer Weiterbildung in *"Data Science und Business Analytics"* am WIFI Voralberg. Mein Ziel war es, meine Interesse für Phasenübergänge (als diese oft in Komplexe Systeme stattfinden) (mit meiner akademischen Background in Differentialgeometrie und Mathematischer Physik) auf der Ebene der modernen KI-Forschung zu untersuchen.


##  Forschungsfokus
- **Lernkurven:** Analyse der Divergenz zwischen Trainings- und Test-Accuracy.
- **Gewichts-Dynamik:** Untersuchung der L2-Norm als Indikator für strukturelle Reorganisation.
- **Interpretability:** Visualisierung des "AHA-Moments" beim Erlernen modularer Arithmetik.


##  Tech Stack
- **Sprache:** Python 3.12.10
- **Frameworks:** TensorFlow, NumPy, Pandas
- **Visualisierung:** Matplotlib (inkl. Animationen & ipympl)
- **Reporting:** PyPDF für automatisierte Analysen

##  Installation & Setup
1. **Repository klonen:**
   ```bash
   git clone https://github.com
   cd Grokking-Code
Um dieses Experiment lokal zu reproduzieren, folgen Sie diesen Schritten:

1. **Repository klonen:**
   ```bash
   git clone https://github.com
   cd Grokking-Code


2. **Virtuelle Umgebung erstellen:**
   ```bash
    python -m venv venv
    # Aktivierung unter Windows:
    .\venv\Scripts\activate
    # Aktivierung unter Mac/Linux:
    source venv/bin/activate 


3. **Abhängigkeiten installieren:**
Die benötigten Pakete sind in der requirements.txt definiert:
    ```bash
    pip install -r requirements.txt

4. **Projektstruktur**

    ```src/:``` Enthält die Kern-Logik und die Trainings-Skripte ```(.py)```. Das Hauptskript ist ```Grokking_training_Embedding_Attention_und_MLP.py```. Das Skript ```Grokking_training_baseline_MLP.py``` ist wurde am Anfang des Projekts verwendet und hat eine unterschiedliche Architekture, siehe**Analysis.md**.

    ```notebooks/```: Jupyter Notebooks für explorative Analyse und Visualisierung.

    ```runs/```: (Lokal) Enthält generierte Logs und Checkpoints (nicht im Repo enthalten).

    ```plots/```: Exportierte Visualisierungen der Grokking-Effekte.

5. **Analyse & Ergebnisse**
Eine detaillierte wissenschaftliche Presaentation der Ergebnisse, der mathematischen Hintergründe, meiner empirischen Beobachtungen, sowie Plots und Code-Snippets finden Sie in der: **Analysis.md**. 
Diese wurde als eine Presaentation für die Abschlusspruefung der Ausbildung *"Data Science und Business Analytics"* am 30.01.2026 am WIFI Voralberg vorbereitet. 