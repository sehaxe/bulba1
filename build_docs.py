#!/usr/bin/env python3
"""
Сборка всего кода в один markdown файл для документации.
"""
import os
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
EXCLUDE = {'.venv', 'venv', '__pycache__', '.git', 'node_modules', '.pytest_cache', 'backups'}
MARKDOWN = "CODE.md"

def should_include(path: Path) -> bool:
    name = path.name
    if name.startswith('.') or name.startswith('__'):
        return False
    if any(ex in path.parts for ex in EXCLUDE):
        return False
    return path.suffix in ('.py', '.yaml', '.yml', '.md', '.txt', '.sh', '.toml') and path.is_file()

def get_files():
    files = []
    for f in sorted(ROOT.rglob('*')):
        if should_include(f):
            rel = f.relative_to(ROOT)
            if rel.parts[0] in ('bulba1', 'scripts', 'configs', 'tools', 'tests', 'services', 'docs'):
                files.append(rel)
    return files

def code_block(content: str, lang: str = 'python') -> str:
    return f"```{lang}\n{content.strip()}\n```\n"

def build():
    files = get_files()
    total = 0
    
    md = f"""# Bulba 1 — Complete Codebase

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Contents

"""
    for f in files:
        md += f"- {f}\n"
    
    md += "\n---\n\n"
    
    for f in files:
        path = ROOT / f
        content = path.read_text(encoding='utf-8', errors='ignore')
        total += len(content)
        
        ext = f.suffix
        lang = {'py': 'python', 'yaml': 'yaml', 'yml': 'yaml', 'sh': 'bash', 'toml': 'ini'}.get(ext, '')
        
        md += f"## {f}\n\n"
        
        if lang:
            md += code_block(content, lang)
        else:
            md += content + "\n\n"
    
    md += f"\n---\n\n*Total: {total/1024:.1f} KB of code*\n"
    
    out = ROOT / MARKDOWN
    out.write_text(md, encoding='utf-8')
    print(f"✅ Written to {MARKDOWN} ({total/1024:.1f} KB, {len(files)} files)")

if __name__ == '__main__':
    build()