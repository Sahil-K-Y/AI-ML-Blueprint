import zipfile
import xml.etree.ElementTree as ET
import os

def read_docx(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    try:
        with zipfile.ZipFile(file_path) as docx:
            xml_content = docx.read('word/document.xml')
            root = ET.fromstring(xml_content)
            
            # Namespaces for OOXML
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            paragraphs = []
            for para in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
                text = "".join([node.text for node in para.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if node.text])
                if text.strip():
                    paragraphs.append(text)
            
            return "\n".join(paragraphs)
    except Exception as e:
        print(f"Error reading docx: {e}")
        return None

if __name__ == "__main__":
    docx_path = r"d:\repos\AI-ML-Blueprint\last_roadmap.docx"
    text = read_docx(docx_path)
    if text:
        # Write to a text file in scratch to inspect
        out_path = r"d:\repos\AI-ML-Blueprint\scratch\last_roadmap_text.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Successfully extracted {len(text)} characters to {out_path}")
