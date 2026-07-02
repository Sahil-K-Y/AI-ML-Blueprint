import json
import re
import os

def load_playlists():
    playlists_path = r"d:\repos\AI-ML-Blueprint\scratch\extracted_playlists.json"
    if not os.path.exists(playlists_path):
        return []
    with open(playlists_path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_best_match(topic_title, subtopics, playlists):
    # Normalize query
    query = (topic_title + " " + " ".join(subtopics)).lower()
    
    best_match = None
    best_score = 0
    
    # Keyword weights
    keywords = ["sigmoid", "logistic regression", "softmax", "decision tree", "random forest", "xgboost", "lightgbm", 
                "catboost", "svm", "knn", "naive bayes", "k-means", "dbscan", "pca", "fastapi", "docker", "streamlit", 
                "gradio", "mlflow", "dvc", "shap", "lime", "cross-validation", "optuna", "calibration", "imbalanced"]
    
    matched_keywords = [kw for kw in keywords if kw in query]
    
    for item in playlists:
        item_title = item["title"].lower()
        score = 0
        
        # Exact title match
        if topic_title.lower() in item_title or item_title in topic_title.lower():
            score += 10
            
        # Keyword matches
        for kw in matched_keywords:
            if kw in item_title:
                score += 5
                
        # Word overlap
        query_words = set(re.findall(r'\w+', query))
        title_words = set(re.findall(r'\w+', item_title))
        overlap = len(query_words.intersection(title_words))
        score += overlap
        
        if score > best_score:
            best_score = score
            best_match = item
            
    if best_score > 3: # threshold
        return best_match
    return None

def enrich_side_quest(quest_text, day_num):
    quest_text = quest_text.strip().strip('*')
    if "SQL" in quest_text:
        sql_topics = [
            "SELECT, WHERE, ORDER BY, LIMIT basics",
            "Aggregate functions (COUNT, SUM, AVG, MIN, MAX)",
            "GROUP BY & HAVING filters",
            "INNER JOIN & OUTER JOIN (LEFT/RIGHT)",
            "Self Joins & Subqueries",
            "Common Table Expressions (CTEs)",
            "Window Functions (ROW_NUMBER, RANK, DENSE_RANK)",
            "Window Functions (LAG, LEAD, SUM OVER)",
            "String operations (CONCAT, SUBSTRING, REPLACE)",
            "CASE WHEN conditional logic",
            "LeetCode SQL 50: Problems 1-5",
            "LeetCode SQL 50: Problems 6-10",
            "LeetCode SQL 50: Problems 11-15",
            "LeetCode SQL 50: Problems 16-20",
            "LeetCode SQL 50: Problems 21-25",
            "LeetCode SQL 50: Problems 26-30"
        ]
        topic = sql_topics[day_num % len(sql_topics)]
        return f"SQL: Practice **{topic}** on LeetCode/Hackerrank"
    elif "DSA" in quest_text:
        dsa_topics = [
            "Arrays: Two Sum / Contains Duplicate",
            "Arrays: Best Time to Buy and Sell Stock",
            "Arrays: Product of Array Except Self",
            "Two Pointers: Valid Palindrome / Two Sum II",
            "Sliding Window: Longest Substring Without Repeating Characters",
            "Stack: Valid Parentheses",
            "Binary Search: Search in Rotated Sorted Array",
            "Linked List: Reverse Linked List / Merge Two Sorted Lists",
            "Trees: Maximum Depth of Binary Tree",
            "Trees: Invert Binary Tree / Same Tree",
            "Trees: Binary Tree Level Order Traversal",
            "Heaps: Kth Largest Element in an Array",
            "Graphs: Number of Islands",
            "Dynamic Programming: Climbing Stairs",
            "Dynamic Programming: Coin Change"
        ]
        topic = dsa_topics[day_num % len(dsa_topics)]
        return f"DSA: Solve **{topic}** on LeetCode"
    elif "Interview" in quest_text:
        interview_topics = [
            "Linear Regression assumptions & multicollinearity",
            "L1 vs L2 regularization (sparsity vs weight reduction)",
            "Logistic Regression derivation & log loss",
            "Precision vs Recall vs F1 (business applications)",
            "ROC-AUC vs PR-AUC curves & imbalanced evaluation",
            "Bias-Variance tradeoff & overfitting controls",
            "Decision Trees splitting criteria (Gini vs Entropy)",
            "Random Forests bagging vs boosting difference",
            "XGBoost mathematical advantages & regularization",
            "Support Vector Machines kernel trick & dual problem",
            "K-Means clustering optimal K selection & scaling",
            "PCA dimensionality reduction & variance explained",
            "Feature Engineering: handling outliers and missingness",
            "MLOps: model drift, monitoring, and pipeline triggers"
        ]
        topic = interview_topics[day_num % len(interview_topics)]
        return f"Interview Prep: Prepare answers for **{topic}** interview questions"
    return quest_text

def enrich_day_block(day_num, date_str, status_str, main_topic, subtopics, practice_task, deliverable, side_quest, playlists):
    # Find matching video
    match = find_best_match(main_topic, subtopics, playlists)
    videos_md = ""
    if match and match["videos"]:
        clean_vids = []
        for vid in match["videos"]:
            cleaned = vid.strip().replace('⁏', '').replace(' \u007f', '').replace('\u007f', '')
            clean_vids.append(cleaned)
        videos_md = "- **Playlist Videos:**\n" + "".join([f"  - {vid}\n" for vid in clean_vids])
    else:
        videos_md = "- **Playlist Videos:** None (use official docs/implementation practice)\n"
        
    # Enrich subtopics if they are thin
    if len(subtopics) <= 1:
        # Default subtopics based on common themes
        if "logistic regression" in main_topic.lower():
            subtopics = [
                "Sigmoid mapping & decision boundaries",
                "Log Loss formula & binary cross-entropy minimization",
                "Regularization parameters (C parameter, L1 vs L2 penalty)",
                "Interpretation of coefficients (odds ratios)"
            ]
        elif "decision tree" in main_topic.lower():
            subtopics = [
                "Information Gain, Gini Impurity, and Entropy calculations",
                "Splitting criteria for classification vs regression trees",
                "Overfitting diagnostics and pruning techniques (max_depth, min_samples_split)",
                "Visualizing trees with plot_tree & export_text"
            ]
        elif "xgboost" in main_topic.lower():
            subtopics = [
                "Regularized objective function formulation",
                "Tree pruning & learning rate scheduling",
                "Tuning colsample_bytree, subsample, and max_depth",
                "Early stopping to prevent overfitting"
            ]
            
    # Enrich practice tasks with concrete steps
    task_detail = practice_task
    if "logistic regression" in main_topic.lower() and "implement" in practice_task.lower():
        task_detail = (
            "Load the breast cancer dataset using `load_breast_cancer`. Scale features using `StandardScaler`. "
            "Train a `LogisticRegression` model. Extract and print coefficients and intercept. "
            "Evaluate metrics (`accuracy_score`, `classification_report`) on test set and compare `predict` vs `predict_proba` outputs."
        )
    elif "metrics" in main_topic.lower():
        task_detail = (
            "Calculate confusion matrix components (TP, FP, TN, FN) from predictions. "
            "Manually compute Accuracy, Precision, Recall, and F1-Score in Python code cells. "
            "Verify your calculations using scikit-learn metrics functions."
        )
    elif "roc-auc" in main_topic.lower():
        task_detail = (
            "Compute probability scores using `predict_proba`. "
            "Use `roc_curve` and `precision_recall_curve` to calculate thresholds, TPR, FPR, Precision, and Recall. "
            "Plot both ROC Curve and Precision-Recall Curve. Compare metrics under different decision thresholds."
        )
        
    # Enrich side quest
    enriched_quest = enrich_side_quest(side_quest, day_num)
    
    # Reassemble markdown day block
    status_icon = "✅ Completed" if "DONE" in status_str or "x" in status_str else "⏳ Scheduled"
    status_checkbox = "[x]" if "DONE" in status_str or "x" in status_str else "[ ]"
    
    block_lines = [
        f"#### {status_checkbox} Day {day_num:03d} — {date_str} — {status_icon}",
        f"- **Main Topic:** {main_topic}",
        "- **Subtopics to learn:**"
    ]
    for sub in subtopics:
        block_lines.append(f"  - {sub.strip('- ')}")
    block_lines.append(f"- **Practice Task:** {task_detail}")
    block_lines.append(f"- **Deliverable:** `{deliverable}`")
    block_lines.append(f"- **Side Quest:** *{enriched_quest}*")
    block_lines.append(videos_md.strip('\n'))
    
    return "\n".join(block_lines)

def process_file(file_path, playlists):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    lines = content.split("\n")
    output = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a Day block
        if line.startswith("#### ") and "Day " in line:
            # Parse block
            # Format: #### [ ] Day 032 — 02 Jul 2026, Thursday — ⏳ Scheduled
            header = line
            checkbox_match = re.search(r'\[(.*?)\]', header)
            status_checkbox = checkbox_match.group(1) if checkbox_match else " "
            
            day_match = re.search(r'Day\s+(\d+)', header)
            day_num = int(day_match.group(1)) if day_match else 0
            
            date_match = re.search(r'—\s*(.*?)\s*—', header)
            date_str = date_match.group(1).strip() if date_match else ""
            
            status_match = header.split("—")[-1].strip()
            
            # Read inner lines
            main_topic = ""
            subtopics = []
            practice_task = ""
            deliverable = ""
            side_quest = ""
            
            i += 1
            while i < len(lines) and not lines[i].startswith("#### ") and not lines[i].startswith("---") and not lines[i].startswith("### "):
                inner = lines[i].strip()
                if inner.startswith("- **Main Topic:**"):
                    main_topic = inner.replace("- **Main Topic:**", "").strip()
                elif inner.startswith("- **Subtopics to learn:**"):
                    i += 1
                    while i < len(lines) and lines[i].startswith("  -"):
                        subtopics.append(lines[i].strip().replace("  -", "").strip())
                        i += 1
                    i -= 1 # adjust
                elif inner.startswith("- **Practice Task:**"):
                    practice_task = inner.replace("- **Practice Task:**", "").strip()
                elif inner.startswith("- **Deliverable:**"):
                    deliverable = inner.replace("- **Deliverable:**", "").replace("`", "").strip()
                elif inner.startswith("- **Side Quest:**"):
                    side_quest = inner.replace("- **Side Quest:**", "").replace("*", "").strip()
                i += 1
                
            # Enrich and format
            enriched_block = enrich_day_block(day_num, date_str, status_checkbox, main_topic, subtopics, practice_task, deliverable, side_quest, playlists)
            output.append(enriched_block)
            output.append("\n---")
            i -= 1 # adjust back for the outer loop to process boundary lines
        else:
            if not line.startswith("---") or (output and not output[-1].endswith("---")):
                output.append(line)
        i += 1
        
    # Write back
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output))
    print("Enriched final.md successfully.")

if __name__ == "__main__":
    playlists = load_playlists()
    process_file(r"d:\repos\AI-ML-Blueprint\final.md", playlists)
