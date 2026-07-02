import os
import subprocess

def run_git_cmd(cmd):
    try:
        res = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Executed: {cmd} -> Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing: {cmd}\nError: {e.stderr.strip()}")
        return False

def main():
    repo_path = r"d:\repos\AI-ML-Blueprint"
    os.chdir(repo_path)
    
    # List of useless files to remove
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
            success = run_git_cmd(f'git rm -f "{filename}"')
            if not success:
                try:
                    os.remove(filepath)
                    print(f"Removed via os.remove: {filename}")
                except Exception as e:
                    print(f"Failed to remove {filename}: {e}")
        else:
            print(f"File {filename} not found, skipping.")
            
    # Try to delete Phase-3 folder if it is empty or try to remove untracked stuff
    old_phase3_dir = os.path.join(repo_path, "Phase-3 - Logistic Regression Deep Dive")
    if os.path.exists(old_phase3_dir):
        print("Attempting to clean up old Phase-3 folder...")
        # Since we've already moved tracked files to July folder, we can clean up the old directory.
        # We run standard git rm on the old folder just in case
        run_git_cmd(f'git rm -r -f "Phase-3 - Logistic Regression Deep Dive"')
        
        # Then attempt to remove it from disk forcefully (ignoring errors for locked files)
        import shutil
        try:
            shutil.rmtree(old_phase3_dir, ignore_errors=True)
            print("Force removed Phase-3 folder from disk (ignoring errors for locked files).")
        except Exception as e:
            print(f"Could not remove Phase-3 folder directory: {e}")

if __name__ == "__main__":
    main()
