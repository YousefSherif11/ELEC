from pathlib import Path
path = Path('streamlitapp.py')
text = path.read_text(encoding='utf-8')
text = text.replace('use_container_width=True', "width='stretch'")
path.write_text(text, encoding='utf-8')
print('replaced', text.count("width='stretch'"))
