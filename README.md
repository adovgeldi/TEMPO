# TEMPO: Time-series Engine for Modeling and Parameter Optimization

This repository contains Python code to perform forecasting with Optuna hyperparameter tuning.

## Prerequisites

Before setting up the project, ensure you have Python 3.11 installed on your system.

### Installing Python 3.11

#### On macOS
Using Homebrew:
```bash
brew install python@3.11
```

#### On Windows
1. Download Python 3.11 installer from [python.org](https://www.python.org/downloads/release/python-3110/)
2. Run the installer
3. Make sure to check "Add Python 3.11 to PATH" during installation

## Setting Up the Development Environment

### For macOS

1. Clone the repository
```bash
git clone https://github.com/Inspire11Consulting/rental-demand-forecasting.git
cd rental-demand-forecasting
```

2. Create a virtual environment
```bash
python3.11 -m venv .venv
```

3. Activate the virtual environment
```bash
source .venv/bin/activate
```

4. Install dependencies
```bash
pip install -r requirements.txt
```

### For Windows

1. Clone the repository
```bash
git clone https://github.com/Inspire11Consulting/rental-demand-forecasting.git
cd rental-demand-forecasting
```

2. Create a virtual environment
```bash
python -m venv .venv
```

3. Activate the virtual environment
```bash
.\.venv\Scripts\activate
```

4. Install dependencies
```bash
pip install -r requirements.txt
```

## Project Dependencies

This template includes the following key packages:

- **Data Processing**
  - pandas 2.0.3
  - numpy 1.24.3

- **Machine Learning**
  - scikit-learn 1.3.0
  - xgboost 2.0.3
  - lightgbm 4.1.0

- **Deep Learning**
  - tensorflow 2.15.0
  - keras 2.15.0

- **Time Series Analysis**
  - prophet 1.1.5
  - statsmodels 0.14.1

- **Utilities**
  - joblib 1.3.2
  - typing-extensions 4.6.0
  - python-dateutil 2.8.2

## Troubleshooting

If you encounter issues during installation:

1. Ensure you're using Python 3.11
```bash
python --version
```

2. Make sure pip is up to date
```bash
pip install --upgrade pip
```

3. If Prophet installation fails, try installing it separately:
```bash
pip install prophet==1.1.5
```

4. For TensorFlow issues, ensure you have the required system libraries installed.

## Development

For development work, the project includes pytest for testing:
```bash
pytest
```

## Contributing

1. If the ```dev``` branch already exists locally then switch to the existing ```dev``` branch
```bash
git checkout dev
```
If this command throws an error then you'll need to fetch the remote branch first using
```bash
git fetch origin
```
Check to ensure you can see the ```dev``` branch
```bash
git branch -r
```
Once the metadata is up to date with the remote repo you can create a local ```dev``` branch to track to the remote branch
```bash
git checkout -b dev origin/dev
```

2. Once on ```dev```, create your feature branch off dev 
```bash
git checkout -b initials-amazing-feature
```
Please replace ```initials``` with your initials for easy delineation and ```amazing-feature``` with a short descriptive name for your branch. 

3. After making changes in your new branch, stage and commit the changes with a descriptive message of changes made
```bash
git add .
git commit -m "Your commit message"
```

4. Push to the remote feature branch 
```bash
git push -u origin initials-amazing-feature
```
The ```-u``` flag sets the upstream branch to ```origin/initials-amazing-feature```. This only needs to be set once; after this, ```git pull``` and ```git push``` will automatically know which remote branch to sync with.

5. Open a Pull Request and manually specify your feature branch as the Compare Brand and ```dev``` as the Base Branch.

Note: If you want to align `initials-amazing-feature` with the latest changes in `dev` before commiting changes:

**Fetch Updates from Remote**:
```bash
git fetch origin
```

**Merge or Rebase** `initials-amazing-feature` **with** `dev`:

- **Merge**: Integrates changes from `dev` to `initials-amazing-feature`, preserving both histories

```bash
git checkout initials-amazing-feature
git merge dev
```

- **Rebase**: Reapplies the commits in  `initials-amazing-feature` on top of the latest `dev` commits, creating a cleaner history

```bash
git checkout initials-amazing-feature
git rebase dev
```


After updating, push the changes
```bash
git push
```

If rebasing, you may need to force-push
```bash
git push --force
```
