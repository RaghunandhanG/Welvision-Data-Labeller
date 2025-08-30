#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Script to fix the corrupted emoji in the tab text

def fix_tab_text():
    file_path = r"C:\Users\raghu\OneDrive\Desktop\WELVISION DATA LABELLER\yolo_labeler_app.py"
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Replace the corrupted line
    old_line = 'self.notebook.add(self.preview_frame, text="�  Dataset Preview")'
    new_line = 'self.notebook.add(self.preview_frame, text="  📂  DATASET PREVIEW  ")'
    
    # Make the replacement
    content = content.replace(old_line, new_line)
    
    # Also fix any other potential encoding issues
    content = content.replace("�", "📂")
    
    # Write back the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed tab text encoding issues!")

if __name__ == "__main__":
    fix_tab_text()
