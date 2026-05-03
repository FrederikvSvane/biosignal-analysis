# Direction of exploration and chosen topic

We choose to explore along the Zero-Shot Stress Detection topic from the case description:

>Zero-Shot Stress Detection: Employ One-Class SVMs or anomaly detection methods. Train your model on the ’Resting’ phase data and evaluate if it successfully flags the ’Puzzling/Competing’ phases as anomalies.

So the high level goal is to

i) Train an anomaly detection model on each person's resting periods

ii) Test the model on their "puzzling" periods, and see if it flags them as unusual (stressed).

## Data analysis- and modelling pipeline

1. Preprocess the raw timeseries data into a per-person timeseries csv (```preprocess_data.py```)
2. (optional) Visualize the timeseries data along with the reponse data for each person, on individual "data-cards" (```visualize_data.py```)
3. Extract Features: Roll sliding windows over the timeseries data for each phase (per-person) to compute some per-window features such as

    i) HR mean, variance, min, max

    ii) EDA mean + wave peak count (how many times does it spike)

    iii) TEMP slope

    iv) (optional) a frequency-domain HR feature. So Fourier transform + something. This is what the assignment calls the "Temporal Dynamics" advanced challenge

    v) and maybe more like this

4. Normalizing: Each subject against their own resting baseline, such that the model learns "this person is more stressed/activated than usual" rather than "wow Person A has a much higher heart rate than Person B"

5. Modelling: For each of the 26 persons (leave-one-subject-out scheme, which is what the assignment calls the "Generalization Challenge"): Train an anomaly detector on all the 25 other people from the experiment (only their resting phases), then score the held-out person's puzzle-phase windows under the trained model. Higher score = more "unlike resting".

    The case description requires that we use at least one method from weeks 8-12, and One-Class SVM (week 7) doesn't qualify...
    So we should incoorporate a GMM and some PCA aswell:

    i) A GMM (Gaussian Mixture Model, week 9) as the primary anomaly detector.

    ii) PCA (week 8) on the windowed features before fitting the GMM. This gives a low-dimensional space to fit the mixture in, and produces an interpretable decomposition figure for the report.

    iii) Then compare against the One-Class SVM under the exact same protocol, so we can show we picked the curriculum method deliberately and benchmarked it against a sensible baseline.

6. Evaluation: We want to see how correct and confident the model is in its classifications. A ROC (Receiver Operating Characteristic) curve is exactly the right tool for this. More specifically the AUROC (Area Under ROC), with 0.5 being random change and 1.0 being perfect seperation and confidence. We report it per-person, since leave-one-subject-out shows if the model generalizes across people. We also report it pooled across all people, to get one number to compare the GMM against the SVM.

7. Visualization: Reuse the system-card figure from `visualize_data.py`, adding a fifth panel below the four biosignal panels that overlays the model's anomaly score along the same stitched timeline. Phase shading is already there, so whether the score rises during phase2 is visually obvious. Pick a couple of representative persons (one where it works, one where it doesn't) for the report.

8. Interpretation: Cross-check against the self-report data. Do people whose puzzle windows are flagged as anomalies by the model also report higher frustration, hostility etc (negative emotions)?