import os
import subprocess
import shutil

def run_git_cmd(cmd):
    try:
        res = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Executed: {cmd}\nOutput: {res.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing: {cmd}\nError: {e.stderr.strip()}")
        return False

def main():
    repo_path = r"d:\repos\AI-ML-Blueprint"
    os.chdir(repo_path)
    print(f"Switched directory to: {os.getcwd()}")
    
    # 1. Rename directories if they exist
    rename_mappings = {
        "Phase-1 - EDA & Feature Engineering": "01 - June - Foundations",
        "Phase-2 - Regression & Streamlit": "02 - June - Regression",
        "Phase-3 - Logistic Regression Deep Dive": "03 - July - Classification"
    }
    
    for old_name, new_name in rename_mappings.items():
        old_dir = os.path.join(repo_path, old_name)
        new_dir = os.path.join(repo_path, new_name)
        
        if os.path.exists(old_dir):
            print(f"Found directory: {old_name}. Renaming to: {new_name}")
            # Use git mv to preserve history
            success = run_git_cmd(f'git mv "{old_name}" "{new_name}"')
            if not success:
                # Fallback to standard shutil move if git mv fails (e.g. untracked files)
                print("git mv failed, trying standard directory move...")
                if os.path.exists(new_dir):
                    shutil.copytree(old_dir, new_dir, dirs_exist_ok=True)
                    shutil.rmtree(old_dir)
                else:
                    shutil.move(old_dir, new_dir)
                run_git_cmd(f'git add "{new_name}"')
        else:
            print(f"Directory {old_name} not found, skipping rename.")

    # 2. Identify and remove useless files
    useless_files = [
        "aiml.txt",
        "final_detailed_roadmap_from_1_june.md",
        "final_old_backup.md",
        "last_roadmap.docx",
        "pdf_text.txt",
        "1knk30Gi9dpUvEk2gfqtH5sS-xeSZRDhw-xSh7erMOE.pdf",
        "K5ysrlRxSnIzqL3eSoTUh15bbU7qmm-EWJ54YsCRsKs.pdf"
    ]
    
    for filename in useless_files:
        filepath = os.path.join(repo_path, filename)
        if os.path.exists(filepath):
            print(f"Removing useless file: {filename}")
            # Try git rm first
            success = run_git_cmd(f'git rm -f "{filename}"')
            if not success:
                try:
                    os.remove(filepath)
                    print(f"Removed via os.remove: {filename}")
                except Exception as e:
                    print(f"Error removing {filename}: {e}")
        else:
            print(f"File {filename} not found, skipping removal.")

if __name__ == "__main__":
    main()
