# Privacy-Conscious Supervised Face Identification

This is a local educational prototype for identifying consenting enrolled volunteers. It is not a surveillance system and must not be used for public identification, access control, employment, policing, or decisions about people.

## Step 1: Install prerequisites

Install Python 3.11 or newer, VS Code, and the VS Code Python extension. Open this folder in VS Code and open **Terminal > New Terminal**.

## Step 2: Create and activate the virtual environment

From the project root:

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt:

```bat
py -3 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Confirm that VS Code is using `.venv`: press `Ctrl+Shift+P`, choose **Python: Select Interpreter**, and select the interpreter whose path contains `.venv`.

## Step 3: Consent and dataset choices

The preferred dataset is one you collect locally from yourself and adults who explicitly agree. Do not collect people in the background. Use pseudonymous IDs such as `person_01`; keep names and consent documents separate from image files.

For each participant, collect two or more sessions, ideally on different days or under different lighting. The collection script targets 40 images per session. For a useful classroom demonstration, use at least four consenting people, two sessions per person, and approximately 30–60 usable images per person.

The project can also be tested initially on a public research dataset, but check its license and terms before downloading or redistributing it. OpenCV's face-recognition tutorial describes the AT&T/ORL Faces and Yale face databases as examples of classical face-recognition datasets. Those public datasets are useful for learning, but they do not replace consent for collecting your friends' images and may not be appropriate for publication without checking their terms.

## Step 4: Collect your local dataset

Activate the venv, then run:

```bash
python src/collect_images.py
```

Enter a pseudonymous participant ID such as `person_01`, a session such as `session_01`, and type `YES` only after the participant has agreed. Ask the participant to vary expression, small head angle, distance, and ordinary lighting. Press `q` to stop early.

Repeat with at least two sessions per person:

```text
person_01 / session_01
person_01 / session_02
person_02 / session_01
person_02 / session_02
```

The raw files will be placed under `data/raw/<participant_id>/<session_id>/`. They are private and are protected by `.gitignore`.

## Step 5: Detect, crop, and preprocess faces

Run:

```bash
python src/build_and_preprocess.py
```

The script skips images without active consent, unreadable images, or images that do not contain exactly one detectable face. It writes standardized grayscale face crops into `data/processed/<participant>/<session>/` and creates `data/processed_manifest.csv`.

## Step 6: Split by capture session

Run:

```bash
python src/split_dataset.py
```

The split is based on session IDs rather than adjacent video frames. With two sessions, one becomes training data and one becomes test data. With three or more sessions, the script assigns separate training, validation, and test sessions. If you have only one session, the script warns that the evaluation may be optimistic.

## Step 7: Train the supervised model

Run:

```bash
python src/train.py
```

The baseline extracts HOG features from each face crop and trains a linear SVM. The model is written to `models/classifier.joblib`.

## Step 8: Evaluate

Run:

```bash
python src/evaluate.py
```

The evaluation writes metrics to `reports/metrics.json` and a test confusion matrix to `reports/figures/confusion_matrix.png`. Record balanced accuracy, macro precision, macro recall, macro F1, support per participant, and the number of images in each split. Do not claim that this small volunteer dataset represents general performance.

## Step 9: Test one image

Use a new, consented test image that was not used for training:

```bash
python src/infer.py --image path/to/consented_test_image.jpg
```

The program returns the predicted participant and probability. If the probability is below the configured threshold, it returns `unknown`.

## Step 10: Test the webcam

Run:

```bash
python src/infer.py
```

Press `q` to quit. Webcam frames are not saved by this script. Test only with the owner and volunteers who consented. Also test a consenting person who was not enrolled; the expected output is `unknown`, although a small educational model may still make mistakes.

## Project structure

```text
configs/config.yaml       # paths and model settings
data/raw/                 # private original images
data/processed/           # private normalized face crops
consent/                  # private consent log
models/                   # private trained model
reports/                  # metrics and figures
src/collect_images.py     # webcam dataset collection
src/build_and_preprocess.py
src/split_dataset.py
src/train.py
src/evaluate.py
src/infer.py              # image and webcam inference
```

## Troubleshooting

If `cv2.imshow` fails, run the webcam script on a computer with a graphical desktop and camera permission. If no faces are detected, improve lighting, move closer, face the camera, and check that the image contains exactly one face. If training reports fewer than two classes, collect data for at least two participants and ensure that their consent IDs match the directory names. If the model recognizes training images but fails on new sessions, collect more varied sessions and do not weaken the test split.

If you receive a camera-open error, change `camera_index` from `0` to `1` in `configs/config.yaml`. If probability values are too high for incorrect predictions, treat them as model scores rather than proof of identity and calibrate the `unknown_probability_threshold` using validation images only.

## Deleting the dataset

When the demonstration is complete, delete `data/raw/`, `data/processed/`, `consent/`, and `models/`. If a participant withdraws, stop using their data and delete their raw images, processed crops, and any model trained with them. Then retrain from the remaining consented data.
