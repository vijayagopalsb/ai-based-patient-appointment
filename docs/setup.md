[Back to Main README](../README.md)

---

## Project Setup Guide: AI-based Patient Appointment System

This guide provides step-by-step instructions to set up and run the **AI-based Patient Appointment System** on your local machine.

---

### Prerequisites

1. **Python 3.10 or higher** installed on your system.

2. **Git** installed (optional for cloning from GitHub).

### Environment Setup

#### 1. Clone the Repository

```bash
git clone https://github.com/vijayagopalsb/ai-based-patient-appointment.git
cd ai-based-patient-appointment
```
#### 2. Create a Virtual Environment

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate

```

For Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate

```

#### 3. Install Required Packages

```bash
pip install -r requirements.txt

```

### Running the Project

#### 1. Generate Data, Perform EDA, Train Model

From the project root directory:

```bash
python main.py

```

This will:

- Generate synthetic patient data.

- Perform visual exploratory data analysis (EDA).

- Train and evaluate a multi-label classification model.

All visual outputs (charts) will be saved in:

```bash
output_images/eda/

```

### Project Configuration

Configurations like number of records and file paths can be adjusted in:

```bash
src/utils/config.py

```

---
[Back to Main README](../README.md)