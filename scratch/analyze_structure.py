import re

def analyze_roadmap():
    with open(r"d:\repos\AI-ML-Blueprint\roadmap (1).txt", "r", encoding="utf-8") as f:
        content = f.read()
        
    lines = content.split("\n")
    current_month = "Unknown"
    month_days = {}
    
    for line in lines:
        if line.startswith("# ") and "202" in line:
            current_month = line.strip("# ")
            month_days[current_month] = []
        elif line.startswith("Day ") and "|" in line:
            day_num = line.split("|")[0].strip()
            month_days[current_month].append(day_num)
            
    for month, days in month_days.items():
        if days:
            print(f"{month}: {days[0]} to {days[-1]} ({len(days)} days)")
        else:
            print(f"{month}: 0 days")

if __name__ == "__main__":
    analyze_roadmap()
