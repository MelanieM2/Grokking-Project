# Grokking Experiment: Visualisierung des Lernens kleiner neuronaler Netze bei modularer Addition

Dieses Projekt untersucht das Phänomen des **"Grokking"** – jenen Moment, in dem ein neuronales Netz von reinem Auswendiglernen (Memorization) zu echter mathematischer Generalisierung übergeht. 

Das Projekt entstand als Abschlussprojekt im Rahmen einer Weiterbildung in „Data Science und Business Analytics“ am WIFI Vorarlberg unter der Leitung von Prof. Jürgen Brauner. Das Projektziel war es, das Phänomen „Grokking” aus empirischer Perspektive zu interpretieren und dabei gleichzeitig Kompetenzen in den Bereichen Deep Learning und Data Science zu erwerben.



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

    -`src/:` Enthält die Kern-Logik und die Trainings-Skripte `(.py)`. 
    
    Das Hauptskript ist `Grokking_training_Embedding_Attention_und_MLP.py`. 
    
    Das Skript `Grokking_training_baseline_MLP.py` wurde am Anfang des Projekts verwendet und hat eine unterschiedliche Architekture, siehe **Analysis.md**. 
    
    Das Skript `Grokking_training_Embedding_Attention_und_MLP_mit_auto_LR_Scheduler.py` ist
    eine Erweiterung des Hauptskripts, bei der die `Learning Rate` des Modells um einen 
    automatischen (als Aufgabe einer spezialisierten Klasse) **Lernraten-Scheduler** ergänzt wurde. Dieser dient speziell der Überwindung von Rauschen ab einem bestimmten Wert der Test-Accuracy mit einer automatischen Absenkung der Learning Rate unter bestimmten Bedingungen.
    



    -`notebooks/`: Jupyter Notebooks für explorative Analyse und Visualisierung.

    -`runs/`: (Lokal) Enthält automatisch generierte Logs, Modell-Checkpoints (Backups), Trainingsdaten sowie `.csv`-Dateien mit den Trainingsmetriken. Diese Verzeichnisse werden zur Laufzeit erstellt und sind nicht im Repository enthalten.


    -`plots/`: Exportierte Visualisierungen der Grokking-Effekte.

   


5. **Analyse & Ergebnisse**
Eine detaillierte Presaentation der Ergebnisse, der theoretischen Hintergründe, meiner empirischen Beobachtungen, sowie Plots und Code-Snippets finden Sie in der: **Analysis.md**. 

---

## Über die Authorin

Ich bin Mathematikerin mit den Schwerpunkten mathematische Physik, Differenzialgeometrie und globale Analyse auf Mannigfaltigkeiten. Derzeit richte ich meine Forschungsinteressen und meine Weiterentwicklung an den Bereichen Deep Learning und Data Science aus.

Melanie Maldonado, PhD

Abschlussprojekt im Rahmen der Weiterbildung  
[Data Science und Business Analytics – WIFI Vorarlberg](https://www.vlbg.wifi.at/Kursbuch/kurs_detail.php?eKey=Eg&eTypNr=1024&eWJ=)  


