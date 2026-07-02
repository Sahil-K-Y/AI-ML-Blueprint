import re
import datetime

start_date = datetime.date(2026, 6, 1)

def get_day_info(day_idx):
    current_date = start_date + datetime.timedelta(days=day_idx - 1)
    date_str = current_date.strftime("%d %b %Y")
    day_name = current_date.strftime("%a")
    return f"Day {day_idx:03d} — {date_str} ({day_name})"

# Read docx extracted text
with open("d:\\repos\\AI-ML-Blueprint\\scratch\\extracted_docx_content.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f]

# We need to extract the days
days = []
current_phase_name = None
current_phase_days = None
current_phase_dates = None
current_phase_status = None

# Let's parse
i = 0
while i < len(lines):
    line = lines[i]
    
    # Detect Phase headers
    if "PHASE " in line and ":" in line:
        current_phase_name = line.strip()
        # Look at next line for Day range, Dates, and Status
        if i + 1 < len(lines) and "Day " in lines[i+1]:
            detail_line = lines[i+1]
            parts = [p.strip() for p in detail_line.split("|")]
            if len(parts) >= 3:
                current_phase_days = parts[0]
                current_phase_dates = parts[1]
                current_phase_status = parts[2]
        i += 2
        continue
        
    # Detect Day numbers
    if line.isdigit() and len(line) == 3:
        day_num = int(line)
        if i + 1 < len(lines):
            date_str = lines[i+1]
            if i + 2 < len(lines):
                topic = lines[i+2]
                if i + 3 < len(lines):
                    status = lines[i+3]
                    days.append({
                        "day": day_num,
                        "date": date_str,
                        "topic": topic,
                        "status": status,
                        "phase_name": current_phase_name,
                        "phase_days": current_phase_days,
                        "phase_dates": current_phase_dates,
                        "phase_status": current_phase_status
                    })
                    i += 4
                    continue
    i += 1

print(f"Parsed {len(days)} days.")

# Generate Phase Overview & Progress Tracker
phases_list = []
seen_phases = set()
for d in days:
    phase_name = d["phase_name"]
    if phase_name not in seen_phases:
        seen_phases.add(phase_name)
        # Parse phase ID: e.g. "PHASE 01: FOUNDATION" -> "01"
        phase_id = phase_name.split(":")[0].replace("PHASE", "").strip()
        phase_title = phase_name.split(":")[1].strip()
        # status clean
        p_status = d["phase_status"].replace("Status:", "").strip()
        if "COMPLETED" in p_status.upper():
            status_symbol = "Completed ✅"
        elif "ACTIVE" in p_status.upper():
            status_symbol = "Active 🚀"
        else:
            status_symbol = "Upcoming ⏳"
            
        phases_list.append({
            "id": phase_id,
            "title": phase_title,
            "days": d["phase_days"],
            "dates": d["phase_dates"],
            "status": status_symbol,
            "raw_name": phase_name
        })

# Format header
header = """# 🏁 Production-Grade AI/ML Engineer Master Roadmap

**Timeline:** 1 June 2026 to 31 December 2026  
**Duration:** 214 Days | **Phases:** 16  
**Execution Formula:** Concept → Practice → Deliverable  
**Primary Stack:** Python, Scikit-Learn, PyTorch, FastAPI, Streamlit, Docker, MLflow, SQL, Linux, Git  

---

## 📅 Phase Overview & Progress Tracker

| Phase | Title | Days | Dates | Status |
| --- | --- | --- | --- | --- |
"""

for p in phases_list:
    header += f"| {p['id']} | {p['title']} | {p['days']} | {p['dates']} | {p['status']} |\n"

header += """
---

═══════════════════════════════════════════════════════════════════
           COMPLETE AIML ROADMAP - SAHIL
           GitHub: Sahil-K-Y | CGPA: 8.1+
           Goal: AIML Engineer by Dec 31, 2026
═══════════════════════════════════════════════════════════════════

YOUR CURRENT STATUS:
✅ Python (Basics, OOPs, File, Exception)
✅ NumPy, Pandas, Matplotlib, Seaborn
✅ EDA, Statistics
✅ Linear Regression, Polynomial, Multiple LR
✅ Ridge, Lasso, Elastic Net
✅ Gradient Descent, MSE, MAE, RMSE
✅ House Price Project
✅ DSA: Array, LL, Stack, Queue, Tree, BST

START FROM: MACHINE LEARNING (Classification)

"""

# Define deliverables mappings for completed days and specific days
completed_deliverables = {
    1: "Phase-0/Day 001 - Env.md",
    2: "Phase-0/Day 002 - Python.md",
    3: "Phase-0/Day 003 - Exception.md",
    4: "Phase-0/Day 004 - Stats.md",
    5: "Phase-0/Day 005 - Probability.md",
    6: "Phase-0/Day 006 - Hypothesis.md",
    7: "Phase-0/Day 007 - Algebra.md",
    8: "Phase-0/Day 008 - Inverse.md",
    9: "Phase-0/Day 009 - Eigen.md",
    10: "Phase-0/Day 010 - DSA.md",
    11: "Phase-1/Day 011 - NumPy.md",
    12: "Phase-1/Day 012 - Pandas.md",
    13: "Phase-1/Day 013 - Aggs.md",
    14: "Phase-1/Day 014 - Matplotlib.md",
    15: "Phase-1/Day 015 - Seaborn.md",
    16: "Phase-1/Day 016 - Plotly.md",
    17: "Phase-1/Day 017 - Univariate.md",
    18: "Phase-1/Day 018 - Multivariate.md",
    19: "Phase-1/Day 019 - Outliers.md",
    20: "Phase-1/Day 020 - LL.md",
    21: "Phase-2/Day 021 - SimpleLR.md",
    22: "Phase-2/Day 022 - MultipleLR.md",
    23: "Phase-2/Day 023 - PolyLR.md",
    24: "Phase-2/Day 024 - GD.md",
    25: "Phase-2/Day 025 - Ridge.md",
    26: "Phase-2/Day 026 - Lasso.md",
    27: "Phase-2/Day 027 - Metrics.md",
    28: "Phase-2/Day 028 - HousePre.md",
    29: "Phase-2/Day 029 - HouseModel.md",
    30: "Phase-3 - Logistic Regression Deep Dive/src/pipeline.py",
    31: "Phase-3 - Logistic Regression Deep Dive/tests/test_pipeline.py"
}

out_content = header + "\n## 📋 Detailed Daily Roadmap Schedule\n\n"

# We split the days by status for display or just show them in order
out_content += "### 🟢 COMPLETED PHASES (Day 001 - Day 030)\n\n"

current_rendered_phase = None

for d in days:
    day_num = d["day"]
    title = d["topic"]
    status_str = d["status"]
    
    # Determine the status and deliverable path
    if day_num < 31:
        status = "COMPLETED"
        deliverable = completed_deliverables.get(day_num, f"Phase-{d['phase_name'].split(':')[0].replace('PHASE', '').strip()}/Day_{day_num:03d}.md")
    elif day_num == 31:
        status = "CURRENT — START HERE TODAY"
        deliverable = completed_deliverables.get(day_num, "Phase-3 - Logistic Regression Deep Dive/tests/test_pipeline.py")
    else:
        status = "SCHEDULED"
        # Determine the phase folder name
        phase_id_str = d["phase_name"].split(":")[0].replace("PHASE", "").strip()
        phase_id = int(phase_id_str)
        if phase_id == 3:
            folder = "Phase-3 - Logistic Regression Deep Dive"
        else:
            folder = f"Phase-{phase_id_str}"
        
        # Clean title for filename
        clean_title = re.sub(r'[^a-zA-Z0-9]', '', title.replace(" ", "_"))
        # Some special cases for file extensions
        if "project" in title.lower() or "pipeline" in title.lower() or "api" in title.lower() or "app" in title.lower() or "code" in title.lower() or "main" in title.lower() or "schemas" in title.lower() or "model" in title.lower():
            if "fastapi" in title.lower() or "api" in title.lower():
                ext = ".py"
            elif "docker" in title.lower():
                ext = "Dockerfile"
            else:
                ext = ".py"
        else:
            ext = ".ipynb"
            
        if ext == "Dockerfile":
            deliverable = f"{folder}/Dockerfile"
        else:
            deliverable = f"{folder}/Day_{day_num:03d}_{clean_title}{ext}"

    # Print separator or phase header transition
    if d["phase_name"] != current_rendered_phase:
        current_rendered_phase = d["phase_name"]
        phase_id_str = current_rendered_phase.split(":")[0].replace("PHASE", "").strip()
        phase_title = current_rendered_phase.split(":")[1].strip()
        
        # If transitioning from completed to active/upcoming
        if day_num == 31:
            out_content += "\n### 🟡 ACTIVE & UPCOMING PHASES (Day 031 - Day 214)\n\n"
            
        out_content += f"\n## 📁 {current_rendered_phase} ({d['phase_days']} | {d['phase_dates']})\n\n"

    info = get_day_info(day_num)
    out_content += f"### 📅 {info} — {title}\n"
    
    if day_num >= 31:
        out_content += f"Study {title} deeply enough to explain the intuition, identify when to use it, and connect it to the previous roadmap topics.\n"
        out_content += f"Learn: Write 3-5 short notes summarizing the theory topic in your own words.\n"
        out_content += f"Code/Practice: Implement and verify coding concepts.\n"
        out_content += f"Build/Submit: `{deliverable}`\n"
        out_content += f"Verify: Open/run the deliverable, check outputs, and make sure it is ready for revision or portfolio use.\n"
    
    out_content += f"**Status:** `{status}`  \n"
    if day_num < 31:
        out_content += f"**Deliverable:** `{deliverable}`  \n\n"
    else:
        out_content += f"**Playlist Videos:** None (use official docs/implementation practice)\n\n"
        out_content += "**Daily Tasks:**\n\n"
        
    out_content += "---\n\n"

# Outcomes and deliverables
out_content += """
# 🎯 Final Outcome Targets

- **Classical ML Projects (4):** Titanic (Classification), Fraud Detection (Ensemble), Customer Segmentation (Clustering), Employee Churn (Pipeline/FastAPI).
- **PyTorch Deep Learning Project (1):** Custom training loop, checkpoints, and baseline comparison.
- **Computer Vision Project (1):** YOLOv8 Custom Object Detection or U-Net Image Segmentation.
- **NLP Project (1):** Preprocessing, classical baseline, and sequence/deep model comparison.
- **LLM/RAG/Agent Project (1):** Embeddings, vector store, retrieval, generation, evaluation notes, and CrewAI/LangGraph deployment.
- **Full-Stack Deployed Capstone (1):** FastAPI backend, Streamlit frontend, Docker packaging, deployment, and README.
- **MLOps Evidence:** MLflow runs, Dockerfile, testing, CI/CD outline, monitoring/drift notes, and deployment plan.
- **Job-Readiness Assets:** Resume, portfolio summaries, project stories, interview notes, mock reflections, and application tracker.

---

This roadmap is FINAL. It will not be rebuilt again — only followed, one day at a time.
"""

with open("d:\\repos\\AI-ML-Blueprint\\final.md", "w", encoding="utf-8") as f:
    f.write(out_content)

with open("d:\\repos\\AI-ML-Blueprint\\final_detailed_roadmap_from_1_june.md", "w", encoding="utf-8") as f:
    f.write(out_content)

print("Roadmaps successfully updated!")
