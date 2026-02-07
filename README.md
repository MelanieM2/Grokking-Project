# Grokking Experiment: Visualisierung von Generalisierung in Neuronalen Netzen

Dieses Projekt ist mein Abschlussprojekt im Rahmen des Data Science & Business Analytics Kurses. Es untersucht das Phänomen des "Grokking" – den Moment, in dem ein Modell plötzlich von Auswendiglernen auf echte Generalisierung umschaltet.

##  Forschungsfokus
- Untersuchung der Loss-Kurven bei synthetischen Datensätzen.
- Visualisierung der Gewichts-Dynamik während des Lernprozesses.
- Export der Ergebnisse als interaktive Plots und Dokumentationen.

##  Tech Stack
- **Sprache:** Python 3.12.10
- **Frameworks:** TensorFlow, NumPy, Pandas
- **Visualisierung:** Matplotlib (inkl. Animationen & ipympl)
- **Reporting:** PyPDF für automatische Auswertungen

##  Installation & Setup

Um dieses Experiment lokal zu reproduzieren, folgen Sie diesen Schritten:

1. **Repository klonen:**
   ```bash
   git clone https://github.com
   cd Grokking-Code


2. **Virtuelle Umgebung erstellen:**
   ```bash
    python -m venv venv
### Aktivierung unter Windows:
    .\venv\Scripts\activate
### Aktivierung unter Mac/Linux:
source venv/bin/activate


3. **Abhängigkeiten installieren:**
Die benötigten Pakete sind in der requirements.txt definiert:
    ```bash
    pip install -r requirements.txt

4. **Projektstruktur**

    ```src/:``` Enthält die Kern-Logik und Trainings-Skripte ```(.py)```.
    ```notebooks/```: Jupyter Notebooks für explorative Analyse und Visualisierung.
    ```runs/```: (Lokal) Enthält generierte Logs und Checkpoints (nicht im Repo enthalten).
    ```plots/```: Exportierte Visualisierungen der Grokking-Effekte.

5. **Weitere Info über das Projekt selbst kommt bald**