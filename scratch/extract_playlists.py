import json
import re

def extract_metadata():
    with open(r"d:\repos\AI-ML-Blueprint\parsed_roadmap.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    extracted = []
    
    # Regex to find video identifiers like ML001 or DSMP001
    video_pattern = re.compile(r'([A-Z]+[0-9]+:.*?(?=(?:[A-Z]+[0-9]+:)|(?:Learn:)|$))', re.DOTALL)
    
    for phase in data:
        phase_num = phase.get("phase_num")
        phase_title = phase.get("phase_title")
        for day in phase.get("days", []):
            day_num = day.get("day_num")
            topic_str = day.get("topic", "")
            
            # Find all video patterns in the topic string
            videos = []
            matches = video_pattern.findall(topic_str)
            for m in matches:
                # Clean up match
                cleaned = m.strip()
                if cleaned:
                    videos.append(cleaned)
            
            # Also extract clean topic title (everything before "Study" or "  ")
            title_part = topic_str.split("  ")[0].strip()
            
            extracted.append({
                "day_num": day_num,
                "title": title_part,
                "videos": videos
            })
            
    with open(r"d:\repos\AI-ML-Blueprint\scratch\extracted_playlists.json", "w", encoding="utf-8") as f:
        json.dump(extracted, f, indent=2)
        
    print(f"Extracted metadata for {len(extracted)} days.")

if __name__ == "__main__":
    extract_metadata()
