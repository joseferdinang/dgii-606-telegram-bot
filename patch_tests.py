import re
with open('/mnt/c/Users/Jose/Downloads/Proyectos/tests/test_ocr_extraction.py', 'r') as f:
    content = f.read()
content = re.sub(r'assertEqual\(InvoiceOCREngine\._extract_([a-zA-Z0-9_]+)\((.*?)\), (.*?)\)', r'assertEqual(getattr(InvoiceOCREngine._extract_\1(\2), \'value\', None), \3)', content)
with open('/mnt/c/Users/Jose/Downloads/Proyectos/tests/test_ocr_extraction.py', 'w') as f:
    f.write(content)
