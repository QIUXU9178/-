from docx import Document
from pathlib import Path
import csv

md_path = Path('..') / '论文.md'
doc_path = Path('..') / '论文.docx'

text = md_path.read_text(encoding='utf-8')

doc = Document()
for line in text.splitlines():
    if line.startswith('# '):
        doc.add_heading(line[2:].strip(), level=1)
    elif line.startswith('## '):
        doc.add_heading(line[3:].strip(), level=2)
    elif line.startswith('### '):
        doc.add_heading(line[4:].strip(), level=3)
    elif line.startswith('|') and '---' not in line:
        # skip markdown table rows; use CSV table instead
        continue
    else:
        doc.add_paragraph(line)

# 追加模型对比表格
csv_path = Path('..') / 'output' / 'model_compare.csv'
if csv_path.exists():
    with csv_path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if rows:
        doc.add_page_break()
        doc.add_heading('模型对比表', level=2)
        table = doc.add_table(rows=1, cols=len(rows[0]))
        hdr_cells = table.rows[0].cells
        for i, heading in enumerate(rows[0]):
            hdr_cells[i].text = heading
        for row in rows[1:]:
            row_cells = table.add_row().cells
            for i, value in enumerate(row):
                row_cells[i].text = value

doc.save(doc_path)
print('Saved', doc_path)
