import re
import os

def parse_roadmap_to_markdown(txt_path, md_path):
    if not os.path.exists(txt_path):
        print(f"Error: {txt_path} not found.")
        return
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    lines = content.split('\n')
    md_content = []
    
    # Title & Metadata
    md_content.append("# 🏁 Production-Grade AI/ML Engineer Master Roadmap (Beginner-Friendly)")
    md_content.append("\n**Timeline:** 1 June 2026 to 2 January 2027  ")
    md_content.append("**Duration:** 216 Days | **Status:** In Progress 🚀\n")
    md_content.append("---")
    
    # Parse status, rules, monthly targets
    mode = 'header'
    day_block = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Detect sections
        if "DAY BY DAY ROADMAP" in line:
            mode = 'roadmap'
            md_content.append("\n## 📅 Detailed Daily Roadmap Schedule\n")
            i += 2
            continue
        elif "PROJECT BUILD ORDER" in line:
            mode = 'projects'
            md_content.append("\n## 🛠️ Project Build Order\n")
            i += 2
            continue
        elif "FINAL CHECKLIST" in line:
            mode = 'checklist'
            md_content.append("\n## ✅ Final Checklist (By Dec 31, 2026)\n")
            i += 2
            continue
            
        if mode == 'header':
            if line.startswith("CURRENT STATUS"):
                md_content.append("### 📊 Current Status")
                i += 1
                while i < len(lines) and lines[i].strip().startswith("-"):
                    md_content.append(lines[i].strip())
                    i += 1
                md_content.append("")
                continue
            elif line.startswith("DAILY STUDY RULE"):
                md_content.append("### 📝 Daily Study Rules")
                i += 1
                while i < len(lines) and lines[i].strip().startswith("-"):
                    md_content.append(lines[i].strip())
                    i += 1
                md_content.append("")
                continue
            elif line.startswith("MONTHLY TARGETS"):
                md_content.append("### 🎯 Monthly Targets")
                i += 1
                while i < len(lines) and lines[i].strip().startswith("-"):
                    md_content.append(lines[i].strip())
                    i += 1
                md_content.append("")
                continue
            elif line.startswith("RECENT UPDATE NOTES"):
                md_content.append("### 🔄 Recent Updates")
                i += 1
                while i < len(lines) and lines[i].strip().startswith("-"):
                    md_content.append(lines[i].strip())
                    i += 1
                md_content.append("")
                continue
                
        elif mode == 'roadmap':
            # Handle month separators
            if line.startswith("# ") and " 202" in line:
                md_content.append(f"\n### {line.replace('# ', '📅 ').strip()}\n")
                i += 2
                continue
                
            # Parse day blocks
            if line.startswith("Day "):
                # Day block start
                day_header = line.strip()
                # Extract day, date, status
                # Format: Day 001 | 01 Jun 2026, Monday | Status: DONE
                parts = day_header.split('|')
                day_num = parts[0].strip()
                date_str = parts[1].strip() if len(parts) > 1 else ""
                status_str = parts[2].strip() if len(parts) > 2 else ""
                
                # Determine checkbox status
                status_icon = "✅ Completed" if "DONE" in status_str else "⏳ Scheduled"
                status_checkbox = "[x]" if "DONE" in status_str else "[ ]"
                
                day_info = f"#### {status_checkbox} {day_num} — {date_str} — {status_icon}"
                md_content.append(day_info)
                
                i += 1
                while i < len(lines) and not lines[i].startswith("Day ") and not lines[i].startswith("# ") and not lines[i].startswith("====="):
                    inner_line = lines[i]
                    if inner_line.startswith("Main Topic:"):
                        topic = inner_line.replace("Main Topic:", "").strip()
                        md_content.append(f"- **Main Topic:** {topic}")
                    elif inner_line.startswith("Subtopics to learn:"):
                        md_content.append("- **Subtopics to learn:**")
                        i += 1
                        while i < len(lines) and lines[i].startswith("  -"):
                            md_content.append(f"  {lines[i].strip()}")
                            i += 1
                        i -= 1 # adjust back
                    elif inner_line.startswith("Practice Task:"):
                        task = inner_line.replace("Practice Task:", "").strip()
                        md_content.append(f"- **Practice Task:** {task}")
                    elif inner_line.startswith("Deliverable:"):
                        deliv = inner_line.replace("Deliverable:", "").strip()
                        md_content.append(f"- **Deliverable:** `{deliv}`")
                    elif inner_line.startswith("Side Quest:"):
                        quest = inner_line.replace("Side Quest:", "").strip()
                        md_content.append(f"- **Side Quest:** *{quest}*")
                    i += 1
                md_content.append("\n---")
                i -= 1 # adjust back
                
        elif mode == 'projects':
            if line.strip() and not line.startswith("==="):
                # Format project list
                md_content.append(line.strip())
                
        elif mode == 'checklist':
            if line.strip() and not line.startswith("===") and not "END OF ROADMAP" in line:
                md_content.append(f"- [ ] {line.strip()}")
                
        i += 1
        
    # Write to final.md
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_content))
    print(f"Successfully wrote parsed markdown to {md_path}")

if __name__ == "__main__":
    txt_path = r"d:\repos\AI-ML-Blueprint\roadmap (1).txt"
    md_path = r"d:\repos\AI-ML-Blueprint\final.md"
    
    # Backup old final.md first
    backup_path = r"d:\repos\AI-ML-Blueprint\final_old_backup.md"
    if os.path.exists(md_path) and not os.path.exists(backup_path):
        os.rename(md_path, backup_path)
        print(f"Backed up old final.md to {backup_path}")
        
    parse_roadmap_to_markdown(txt_path, md_path)
